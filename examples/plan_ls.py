from __future__ import annotations

from copy import deepcopy
from typing import Literal

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import viser.transforms as vtf
import yourdfpy
import warnings
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int, Bool
import trimesh
from typing import Dict, List, Tuple, Optional, cast
import tyro

from jaxmp.collision import SphereSDF, MeshSDF
from jaxmp.collbody import Spheres
from jaxmp.urdf import JaxUrdfwithSphereCollision
import jaxls


def main(
    robot_name: Literal["yumi", "ur5", "allegro_hand"] = "yumi",
    pos_weight: float = 5.00,
    rot_weight: float = 0.50,
    coll_weight: float = 5.0,
    rest_weight: float = 0.001,
    limit_weight: float = 100.0,
):
    """Small test script to visualize the Yumi robot, and the collision spheres, in viser."""
    from pathlib import Path
    import viser
    import viser.extras

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    # Load the robot description.
    if robot_name == "yumi":
        yourdf = load_robot_description("yumi_description")
        target_joint_list = ["yumi_joint_6_r", "yumi_joint_6_l"]
        self_coll_ignore = [
            ("gripper_l_finger_r", "gripper_l_finger_l"),
            ("gripper_r_finger_r", "gripper_r_finger_l"),
        ]
    elif robot_name == "ur5":
        yourdf = load_robot_description("ur5_description")
        target_joint_list = ["ee_fixed_joint"]
        self_coll_ignore = []
    elif robot_name == "allegro_hand":
        yourdf = load_robot_description("allegro_hand_description")
        raise NotImplementedError("Allegro hand not implemented.")

    # Set rest pose as all zeros.
    rest_pose = onp.array([0.0] * yourdf.num_dofs)

    # Set up the viser server.
    server = viser.ViserServer()
    urdf = viser.extras.ViserUrdf(server, yourdf, root_node_name="/urdf")
    urdf.update_cfg(rest_pose)

    # Create collision-aware robot model!
    jax_urdf = JaxUrdfwithSphereCollision.from_urdf(yourdf, self_coll_ignore=self_coll_ignore)

    # Create handles + visualization for target joints.
    target_tf_list: list[viser.TransformControlsHandle] = []
    pred_frame_list: list[viser.FrameHandle] = []   
    for idx in range(len(target_joint_list)):
        handle = server.scene.add_transform_controls(f"target_{idx}", scale=0.2)
        target_tf_list.append(handle)
        handle = server.scene.add_frame(f"pred_{idx}", axes_length=0.1, axes_radius=0.01)
        pred_frame_list.append(handle)

    # Create a variable for the robot's joint positions
    class RobotVar(
        jaxls.Var[jax.Array],
        default=jnp.zeros(yourdf.num_dofs),
    ): ...

    def ik_for_joint(
        jax_urdf: JaxUrdfwithSphereCollision,
        target_joint_name: str,
    ):
        assert target_joint_name in jax_urdf.joint_names
        joint_idx = jax_urdf.joint_names.index(target_joint_name)

        def loss_fn(
            vals: jaxls.VarValues,
            var: RobotVar,
            init: jaxlie.SE3,
        ) -> Array:
            joint_angles = vals[var]
            pose_loss = (
                jaxlie.SE3(jax_urdf.forward_kinematics(joint_angles)[joint_idx]).inverse()
                @ init
            ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3)
            return pose_loss
        return loss_fn

    pose_vars = [RobotVar(id=0)]

    self_coll_value = server.gui.add_number("self_coll", 0.001)

    while True:
        factors = [
            jaxls.Factor.make(
                lambda vals, var: (vals[var] - jax_urdf.limits_lower).clip(max=0.0) * limit_weight,
                (pose_vars[0],)
            ),  # Lower joint limits.
            jaxls.Factor.make(
                lambda vals, var: (vals[var] - jax_urdf.limits_upper).clip(min=0.0) * limit_weight,
                (pose_vars[0],)
            ),  # Upper joint limits.
            jaxls.Factor.make(
                lambda vals, var: (vals[var] - rest_pose) * rest_weight,
                (pose_vars[0],)
            ),  # Bias solution to be centered around rest pose.
            jaxls.Factor.make(
                # lambda vals, var: jnp.maximum(jax_urdf.d_self(cfg=vals[var]) + 0.1, 0.0) * coll_weight,
                lambda vals, var: jnp.maximum(jax_urdf.d_self(cfg=vals[var]) + 0.1, 0.0)**2 * coll_weight,
                (pose_vars[0],)
            ),  # Avoid self-collision.
            *[
                jaxls.Factor.make(
                    ik_for_joint(jax_urdf, target_joint_list[idx]),
                    (pose_vars[0], jaxlie.SE3(jnp.array([*target_tf_list[idx].wxyz, *target_tf_list[idx].position]))),
                )  # Position constraints.
                for idx in range(len(target_joint_list))
            ]
        ]

        graph = jaxls.FactorGraph.make(factors, pose_vars, verbose=False)
        solution = graph.solve(
            initial_vals=jaxls.VarValues.make(pose_vars, [jnp.array(rest_pose)]),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=0.5),
            termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
            verbose=False,
        )

        urdf.update_cfg(onp.array(solution[pose_vars[0]]))

        # joint_idx = jax_urdf.joint_names.index("yumi_joint_6_r")
        # calc_pose = jaxlie.SE3(jax_urdf.forward_kinematics(solution[pose_vars[0]])[joint_idx])
        # calc_handle.position = onp.array(calc_pose.wxyz_xyz[4:])
        # calc_handle.wxyz = onp.array(calc_pose.wxyz_xyz[:4])

        # joint_idx = jax_urdf.joint_names.index("yumi_joint_6_l")
        # calc_pose = jaxlie.SE3(jax_urdf.forward_kinematics(solution[pose_vars[0]])[joint_idx])
        # calc_handle_1.position = onp.array(calc_pose.wxyz_xyz[4:])
        # calc_handle_1.wxyz = onp.array(calc_pose.wxyz_xyz[:4])

        self_coll_value.value = float(jax_urdf.d_self(cfg=solution[pose_vars[0]]).max())

        spheres = jax_urdf.spheres(solution[pose_vars[0]])
        server.scene.add_mesh_trimesh("spheres", spheres.to_trimesh())


if __name__ == "__main__":
    tyro.cli(main)
