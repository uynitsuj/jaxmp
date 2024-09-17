""" 01_kinematics.py
Tests robot forward + inverse kinematics using JaxMP.
"""

from typing import Optional
from pathlib import Path
import time
import tyro
import yourdfpy

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
    robot_urdf_path: Optional[Path] = None,
):
    if robot_urdf_path is not None:
        def filename_handler(fname: str) -> str:
            return str(robot_urdf_path.parent / fname)
        urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)
    else:
        urdf = load_robot_description(robot_description)
    urdf = sort_joint_map(urdf)

    kin = JaxKinTree.from_urdf(urdf)
    robot_factors = RobotFactors(kin)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    urdf_vis.update_cfg(onp.array(rest_pose))
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    add_joint_button = server.gui.add_button("Add joint!")
    target_name_handles: list[viser.GuiDropdownHandle] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.FrameHandle] = []

    @add_joint_button.on_click
    def _(_):
        # Show target joint name.
        idx = len(target_name_handles)
        target_name_handle = server.gui.add_dropdown(
            f"target joint {idx}",
            list(urdf.joint_names),
            initial_value=urdf.joint_names[0]
        )
        target_tf_handle = server.scene.add_transform_controls(f"target_transform_{idx}", scale=0.2)
        target_frame_handle = server.scene.add_frame(f"target_{idx}", axes_length=0.1)

        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    # Create factor graph.
    JointVar = robot_factors.get_var_class(default_val=rest_pose)

    @jdc.jit
    def solve_ik(
        target_pose: jaxlie.SE3,
        target_joint_indices: jdc.Static[tuple[int]],
    ) -> jnp.ndarray:
        joint_vars = [JointVar(id=0)]

        start_time = time.time()
        factors: list[jaxls.Factor] = [
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

        for idx, target_joint_idx in enumerate(target_joint_indices):
            factors.append(
                jaxls.Factor.make(
                    robot_factors.ik_cost,
                    (
                        joint_vars[0],
                        jaxlie.SE3(target_pose.wxyz_xyz[idx]),
                        target_joint_idx,
                        jnp.array([pos_weight] * 3 + [rot_weight] * 3),
                    ),
                )
            )

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
            verbose=False,
        )

        # Update timing info.
        timing_handle.value = (time.time() - start_time) * 1000

        # Update visualization.
        joints = solution[joint_vars[0]]
        return joints

    while True:
        if len(target_name_handles) == 0:
            time.sleep(0.1)
            continue

        target_joint_indices = [
            kin.joint_names.index(target_name_handle.value)
            for target_name_handle in target_name_handles
        ]
        target_pose_list = [
            jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
            for target_tf_handle in target_tf_handles
        ]

        target_poses = jaxlie.SE3(jnp.stack([pose.wxyz_xyz for pose in target_pose_list]))
        joints = solve_ik(
            target_poses,
            tuple[int](target_joint_indices)
        )

        urdf_vis.update_cfg(onp.array(joints))

        for target_frame_handle, target_joint_idx in zip(target_frame_handles, target_joint_indices):
            T_target_world = kin.forward_kinematics(joints)[target_joint_idx]
            target_frame_handle.position = onp.array(T_target_world)[4:]
            target_frame_handle.wxyz = onp.array(T_target_world)[:4]


if __name__ == "__main__":
    tyro.cli(main)
