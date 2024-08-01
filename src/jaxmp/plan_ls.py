from __future__ import annotations

from copy import deepcopy

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

def main():
    """Small test script to visualize the Yumi robot, and the collision spheres, in viser."""
    from pathlib import Path
    import viser
    import viser.extras

    from robot_descriptions.loaders.yourdfpy import load_robot_description
    yourdf = load_robot_description("yumi_description")
    # yourdf = load_robot_description("ur5_description")
    # yourdf = load_robot_description("allegro_hand_description")

    rest_pose = onp.array([0.0] * yourdf.num_dofs)
    jax_urdf = JaxUrdfwithSphereCollision.from_urdf(yourdf)

    server = viser.ViserServer()
    urdf = viser.extras.ViserUrdf(server, yourdf, root_node_name="/urdf")
    urdf.update_cfg(rest_pose)

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
            ).log() * jnp.array([1.0] * 3 + [0.1] * 3)
            return pose_loss
        return loss_fn

    pose_vars = [RobotVar(id=0)]
    target_handle = server.scene.add_transform_controls("target", scale=0.2)
    calc_handle = server.scene.add_frame("calc", axes_length=0.1, axes_radius=0.01)

    curr_joint = server.gui.add_dropdown("target_joint", jax_urdf.joint_names)

    ball = trimesh.creation.icosphere(radius=0.1)
    ball.vertices += onp.array([0.4, 0.0, 0.2])
    server.scene.add_mesh_trimesh("ball", ball)
    ball_sphere = Spheres(1, jnp.array([[0.4, 0.0, 0.2]]), jnp.array([0.1]))

    while True:
        target_pose = jaxlie.SE3(jnp.array([*target_handle.wxyz, *target_handle.position]))
        curr_joint_name = curr_joint.value
        factors = [
            jaxls.Factor.make(
                ik_for_joint(
                    jax_urdf, curr_joint_name
                ),
                (pose_vars[0], target_pose),
            ),  # Position constraints.
            jaxls.Factor.make(
                lambda vals, var: (vals[var] - jax_urdf.limits_lower).clip(max=0.0) * 10.0,
                (pose_vars[0],)
            ),  # Lower joint limits.
            jaxls.Factor.make(
                lambda vals, var: (vals[var] - jax_urdf.limits_upper).clip(min=0.0) * 10.0,
                (pose_vars[0],)
            ),  # Upper joint limits.
            jaxls.Factor.make(
                lambda vals, var: (vals[var] - rest_pose) * 0.001,
                (pose_vars[0],)
            ),  # Bias solution to be centered around rest pose.
            # jaxls.Factor.make(
            #     lambda vals, var: jnp.maximum(jax_urdf.d_self(cfg=vals[var])[None], 0.0) * 0.001,
            #     (pose_vars[0],)
            # ),  # Avoid self-collision.
            # jaxls.Factor.make(
            #     lambda vals, var: jnp.maximum(jax_urdf.d_world(cfg=vals[var], other=ball_sphere)[None], 0.0) * 10.0,
            #     (pose_vars[0],)
            # ),  # Avoid self-collision.
        ]
        graph = jaxls.FactorGraph.make(factors, pose_vars)
        solution = graph.solve(
            initial_vals=jaxls.VarValues.make(pose_vars, [jnp.zeros(yourdf.num_dofs)]),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
            termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        )

        urdf.update_cfg(onp.array(solution[pose_vars[0]]))
        # print(jax_urdf.d_self(cfg=solution[pose_vars[0]]))

        joint_idx = jax_urdf.joint_names.index(curr_joint_name)
        calc_pose = jaxlie.SE3(jax_urdf.forward_kinematics(solution[pose_vars[0]])[joint_idx])
        calc_handle.position = onp.array(calc_pose.wxyz_xyz[4:])
        calc_handle.wxyz = onp.array(calc_pose.wxyz_xyz[:4])

        # spheres = jax_urdf.spheres(solution[pose_vars[0]])
        # server.scene.add_mesh_trimesh("spheres", spheres.to_trimesh())


if __name__ == "__main__":
    tyro.cli(main)
