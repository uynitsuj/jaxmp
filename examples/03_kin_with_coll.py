""" 03_kin_with_coll.py
Similar to 01_kinematics.py, but with collision detection.
"""

import time

from robot_descriptions.loaders.yourdfpy import load_robot_description

import tyro

import jax.numpy as jnp
import jaxlie
import numpy as onp

import jaxls

import viser
import viser.extras

from jaxmp.collision_sdf import dist_signed
from jaxmp.collision_types import HalfSpaceColl, RobotColl
from jaxmp.kinematics import JaxKinTree, sort_joint_map
from jaxmp.robot_factors import RobotFactors

def main(
    robot_description: str = "yumi_description",
    pos_weight: float = 5.0,
    rot_weight: float = 0.5,
    limit_weight: float = 100.0,
    rest_weight: float = 0.1,
    coll_weight: float = 2.0,
    world_coll_weight: float = 10.0,
):
    urdf = load_robot_description(robot_description)
    urdf = sort_joint_map(urdf)

    robot_coll = RobotColl.from_urdf(
        urdf,
        self_coll_ignore=[
            ("gripper_l_finger_l", "gripper_l_finger_r"),
            ("gripper_r_finger_l", "gripper_r_finger_r"),
        ]
    )
    kin = JaxKinTree.from_urdf(urdf)
    robot_factors = RobotFactors(kin, coll=robot_coll)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    target_tf_handle = server.scene.add_transform_controls("target transform", scale=0.2)
    target_frame_handle = server.scene.add_frame("target", axes_length=0.1)

    # Show target joint name, and current joint positions.
    visualize_spheres = server.gui.add_checkbox("Show Collbody", initial_value=False)
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0]
    )

    # Create ground plane as an obstacle (world collision)!
    obstacle = HalfSpaceColl(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0]))
    server.scene.add_mesh_trimesh("ground_plane", obstacle.to_trimesh())
    self_coll_value = server.gui.add_number("max. coll dist (self)", 0.0, step=0.01, disabled=True)
    world_coll_value = server.gui.add_number("max. coll dist (world)", 0.0, step=0.01, disabled=True)

    # Create factor graph.
    JointVar = robot_factors.get_var_class(default_val=rest_pose)

    collbody_handle = None
    def solve_ik():
        nonlocal collbody_handle
        joint_vars = [JointVar(id=0)]

        target_joint_idx = kin.joint_names.index(target_name_handle.value)
        target_pose = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))

        graph = jaxls.FactorGraph.make(
            [
                jaxls.Factor.make(
                    robot_factors.ik_cost,
                    (
                        joint_vars[0],
                        target_pose,
                        target_joint_idx,
                        jnp.array([pos_weight] * 3 + [rot_weight] * 3),
                    ),
                ),
                jaxls.Factor.make(
                    robot_factors.limit_cost,
                    (
                        joint_vars[0],
                        jnp.array([limit_weight] * kin.num_actuated_joints),
                    ),
                ),
                jaxls.Factor.make(
                    robot_factors.self_coll_cost,
                    (joint_vars[0], jnp.array([coll_weight] * len(robot_coll))),
                ),
                jaxls.Factor.make(
                    robot_factors.world_coll_cost,
                    (
                        joint_vars[0],
                        obstacle,
                        jnp.array([world_coll_weight] * len(robot_coll)),
                    ),
                ),
                jaxls.Factor.make(
                    robot_factors.rest_cost,
                    (
                        joint_vars[0],
                        jnp.array([rest_weight] * kin.num_actuated_joints),
                    ),
                ),
            ],
            joint_vars,
            verbose=False,
        )
        solution = graph.solve(
            initial_vals=jaxls.VarValues.make(joint_vars, [JointVar.default]),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
            termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
            verbose=False,
        )
        joints = solution[joint_vars[0]]
        T_target_world = jaxlie.SE3(  # pylint: disable=invalid-name
            kin.forward_kinematics(joints)[target_joint_idx]
        ).wxyz_xyz

        # Update visualization.
        urdf_vis.update_cfg(onp.array(joints))
        target_frame_handle.position = onp.array(T_target_world)[4:]
        target_frame_handle.wxyz = onp.array(T_target_world)[:4]

        coll = robot_coll.transform(jaxlie.SE3(kin.forward_kinematics(joints)))
        self_coll_value.value = dist_signed(coll, coll).max().item()
        world_coll_value.value = dist_signed(coll, obstacle).max().item()
        if visualize_spheres.value:
            collbody_handle = server.scene.add_mesh_trimesh(
                "coll",
                robot_coll.transform(jaxlie.SE3(kin.forward_kinematics(joints))).to_trimesh()
            )
        elif collbody_handle is not None:
            collbody_handle.remove()

    while True:
        solve_ik()
        time.sleep(0.01)


if __name__ == "__main__":
    tyro.cli(main)
