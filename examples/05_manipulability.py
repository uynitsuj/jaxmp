""" 05_manipulability.py.
Similar to 01_kinematics.py, but including manipulability as the cost function!

Hard to verify if it works, but the cost does make the robot pose different.
"""

from robot_descriptions.loaders.yourdfpy import load_robot_description

import time
import tyro
import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp

import jaxls

import viser
import viser.extras

from jaxmp.kinematics import JaxKinTree, sort_joint_map
from jaxmp.robot_factors import RobotFactors

def main(
    robot_description: str = "yumi_description",
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    manipulability_weight: float = 0.01,
):
    urdf = load_robot_description(robot_description)
    urdf = sort_joint_map(urdf)
    kin = JaxKinTree.from_urdf(urdf)
    robot_factors = RobotFactors(kin)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/urdf")
    urdf_vis_no_manip = viser.extras.ViserUrdf(server, urdf, root_node_name="/no_manip", mesh_color_override=(220, 100, 100))
    target_tf_handle = server.scene.add_transform_controls("target transform", scale=0.2)
    target_frame_handle = server.scene.add_frame("target", axes_length=0.1)
    manip_norm_handle = server.gui.add_number("manip (opt)", initial_value=0.01, disabled=True)
    manip_norm_no_handle = server.gui.add_number("manip (no)", initial_value=0.01, disabled=True)

    # Show target joint name, and current joint positions.
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0]
    )

    # Create factor graph.
    JointVar = robot_factors.get_var_class(default_val=rest_pose)

    def solve_ik():
        joint_vars = [JointVar(id=0)]

        target_joint_idx = kin.joint_names.index(target_name_handle.value)
        target_pose = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
        factors = [
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
                robot_factors.rest_cost,
                (
                    joint_vars[0],
                    jnp.array([rest_weight] * kin.num_actuated_joints),
                ),
            ),
            jaxls.Factor.make(
                robot_factors.manipulability_cost,
                (
                    joint_vars[0],
                    target_joint_idx,
                    jnp.array([manipulability_weight] * kin.num_actuated_joints),
                ),
            ),
        ]

        # Solve with manipulability cost.
        graph = jaxls.FactorGraph.make(
            factors,
            joint_vars,
            verbose=False,
        )
        solution = graph.solve(
            initial_vals=jaxls.VarValues.make(joint_vars, [rest_pose]),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
            termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
            verbose=False,
        )
        joints = solution[joint_vars[0]]

        # Solve again, but without the manipulation cost.
        graph = jaxls.FactorGraph.make(
            factors[:-1],
            joint_vars,
            verbose=False,
        )
        solution = graph.solve(
            initial_vals=jaxls.VarValues.make(joint_vars, [rest_pose]),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
            termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
            verbose=False,
        )
        joints_no_manip = solution[joint_vars[0]]

        urdf_vis.update_cfg(onp.array(joints))
        urdf_vis_no_manip.update_cfg(onp.array(joints_no_manip))

        manip_norm_handle.value = robot_factors.manipulability(joints, target_joint_idx).item()
        manip_norm_no_handle.value = robot_factors.manipulability(joints_no_manip, target_joint_idx).item()

        T_target_world = jaxlie.SE3(kin.forward_kinematics(joints)[target_joint_idx]).wxyz_xyz
        target_frame_handle.position = onp.array(T_target_world)[4:]
        target_frame_handle.wxyz = onp.array(T_target_world)[:4]

    while True:
        solve_ik()


if __name__ == "__main__":
    tyro.cli(main)
