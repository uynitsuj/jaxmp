""" 01_kinematics.py
Tests robot forward + inverse kinematics using JaxMP.
"""

import time
import tyro

from robot_descriptions.loaders.yourdfpy import load_robot_description

import jax.numpy as jnp
import jax_dataclasses as jdc
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
):
    urdf = load_robot_description(robot_description)
    urdf = sort_joint_map(urdf)

    kin = JaxKinTree.from_urdf(urdf)
    robot_factors = RobotFactors(kin)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    target_tf_handle = server.scene.add_transform_controls("target transform", scale=0.2)
    target_frame_handle = server.scene.add_frame("target", axes_length=0.1)
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Timing info.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    # Show target joint name, and current joint positions.
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0]
    )

    # Create factor graph.
    JointVar = robot_factors.get_var_class(default_val=rest_pose)

    @jdc.jit
    def solve_ik(target_pose: jaxlie.SE3, target_joint_idx: jdc.Static[int]) -> jnp.ndarray:
        joint_vars = [JointVar(id=0)]

        start_time = time.time()
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
        ]
        graph = jaxls.FactorGraph.make(
            factors,
            joint_vars,
            verbose=False,
            use_onp=False,
        )
        solution = graph.solve(
            initial_vals=jaxls.VarValues.make(joint_vars, [JointVar.default]),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        )

        # Update timing info.
        timing_handle.value = (time.time() - start_time) * 1000

        # Update visualization.
        joints = solution[joint_vars[0]]
        return joints

    while True:
        target_joint_idx = kin.joint_names.index(target_name_handle.value)
        target_pose = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))

        joints = solve_ik(target_pose, target_joint_idx)

        urdf_vis.update_cfg(onp.array(joints))
        T_target_world = kin.forward_kinematics(joints)[target_joint_idx]
        target_frame_handle.position = onp.array(T_target_world)[4:]
        target_frame_handle.wxyz = onp.array(T_target_world)[:4]


if __name__ == "__main__":
    tyro.cli(main)
