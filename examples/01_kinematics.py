""" 01_kinematics.py
Tests robot forward + inverse kinematics using JaxMP.
"""

from typing import Optional
from pathlib import Path
import time
import tyro
import yourdfpy
import viser
import viser.extras

from robot_descriptions.loaders.yourdfpy import load_robot_description

import jax
# Set to cpu.
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import jaxls

from jaxmp.kinematics import JaxKinTree, sort_joint_map
from jaxmp.robot_factors import RobotFactors


@jdc.jit
def solve_ik(
    kin: JaxKinTree,
    target_pose: jaxlie.SE3,
    target_joint_indices: jdc.Static[tuple[int]],
    pos_weight: float,
    rot_weight: float,
    rest_weight: float,
    limit_weight: float,
    rest_pose: jnp.ndarray,
    freeze_base_xyz_xyz: jnp.ndarray,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    """
    Solve IK for the robot.
    Args:
        target_pose: Desired pose of the target joint, SE3 has batch axes (n_target,).
        target_joint_indices: Indices of the target joints, length n_target.
        freeze_base_xyz_xyz: 6D vector indicating which axes to freeze in the base frame.
    Returns:
        Base pose (jaxlie.SE3)
        Joint angles (jnp.ndarray)
    """
    JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * freeze_base_xyz_xyz
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default=jaxlie.SE3.identity(),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    joint_vars = [JointVar(0), ConstrainedSE3Var(0)]

    factors: list[jaxls.Factor] = [
        jaxls.Factor.make(
            RobotFactors.limit_cost,
            (
                kin,
                JointVar(0),
                jnp.array([limit_weight] * kin.num_actuated_joints),
            ),
        ),
        jaxls.Factor.make(
            RobotFactors.rest_cost,
            (
                JointVar(0),
                jnp.array([rest_weight] * kin.num_actuated_joints),
            ),
        ),
    ]

    for idx, target_joint_idx in enumerate(target_joint_indices):
        factors.append(
            jaxls.Factor.make(
                RobotFactors.ik_cost,
                (
                    kin,
                    joint_vars[0],
                    jaxlie.SE3(target_pose.wxyz_xyz[idx]),
                    target_joint_idx,
                    jnp.array([pos_weight] * 3 + [rot_weight] * 3),
                    ConstrainedSE3Var(0),
                ),
            )
        )

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(joint_vars),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        verbose=False,
    )

    # Update visualization.
    base_pose = solution[ConstrainedSE3Var(0)]
    joints = solution[JointVar(0)]
    return base_pose, joints


def main(
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    robot_description: Optional[str] = "yumi",
    robot_urdf_path: Optional[Path] = None,
):
    """
    Test robot inverse kinematics using JaxMP.
    Args:
        pos_weight: Weight for position error in IK.
        rot_weight: Weight for rotation error in IK.
        rest_weight: Weight for rest pose in IK.
        limit_weight: Weight for joint limits in IK.
        robot_description: Name of the robot description to load.
        robot_urdf_path: Path to the robot URDF file.
    """
    # Load robot description.
    if robot_urdf_path is not None:
        def filename_handler(fname: str) -> str:
            base_path = robot_urdf_path.parent
            return yourdfpy.filename_handler_magic(fname, dir=base_path)
        urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)
    elif robot_description is not None:
        robot_description = robot_description + "_description"
        urdf = load_robot_description(robot_description)
    else:
        raise ValueError("Must provide either robot_description or robot_urdf_path.")

    # Sort joint map in topological order.
    urdf = sort_joint_map(urdf)

    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Add base-frame freezing logic.
    with server.gui.add_folder("Base frame"):
        freeze_base_x = server.gui.add_checkbox("Freeze x", initial_value=True)
        freeze_base_y = server.gui.add_checkbox("Freeze y", initial_value=True)
        freeze_base_z = server.gui.add_checkbox("Freeze z", initial_value=True)
        freeze_base_rx = server.gui.add_checkbox("Freeze rx", initial_value=True)
        freeze_base_ry = server.gui.add_checkbox("Freeze ry", initial_value=True)
        freeze_base_rz = server.gui.add_checkbox("Freeze rz", initial_value=True)

    def get_freeze_base_xyz_xyz() -> jnp.ndarray:
        return jnp.array([
            freeze_base_x.value,
            freeze_base_y.value,
            freeze_base_z.value,
            freeze_base_rx.value,
            freeze_base_ry.value,
            freeze_base_rz.value,
        ]).astype(jnp.float32)

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    tf_size_handle = server.gui.add_slider(
        "Gizmo size", min=0.01, max=0.4, step=0.01, initial_value=0.2
    )
    set_frames_to_current_pose = server.gui.add_button("Set frames to current pose")
    add_joint_button = server.gui.add_button("Add joint!")

    target_name_handles: list[viser.GuiDropdownHandle] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.FrameHandle] = []

    # Put robot to rest pose :-)
    base_pose = jaxlie.SE3.identity()
    joints = rest_pose

    urdf_base_frame.position = onp.array(base_pose.translation())
    urdf_base_frame.wxyz = onp.array(base_pose.rotation().wxyz)
    urdf_vis.update_cfg(onp.array(joints))

    # Add joints.
    @add_joint_button.on_click
    def _(_):
        idx = len(target_name_handles)
        target_name_handle = server.gui.add_dropdown(
            f"target joint {idx}",
            list(urdf.joint_names),
            initial_value=urdf.joint_names[0]
        )
        target_tf_handle = server.scene.add_transform_controls(
            f"target_transform_{idx}", scale=tf_size_handle.value
        )
        target_frame_handle = server.scene.add_frame(
            f"target_{idx}",
            axes_length=0.5 * tf_size_handle.value,
            axes_radius=0.05 * tf_size_handle.value,
            origin_radius=0.1 * tf_size_handle.value,
        )
        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    # Let the user change the size of the transformcontrol gizmo.
    @tf_size_handle.on_update
    def _(_):
        for target_tf_handle in target_tf_handles:
            target_tf_handle.scale = tf_size_handle.value
        for target_frame_handle in target_frame_handles:
            target_frame_handle.axes_length = 0.5 * tf_size_handle.value
            target_frame_handle.axes_radius = 0.05 * tf_size_handle.value
            target_frame_handle.origin_radius = 0.1 * tf_size_handle.value

    # Set target frames to where it is on the currently displayed robot.
    # We need to put them in world frame (since our goal is to match joint-to-world).
    @set_frames_to_current_pose.on_click
    def _(_):
        nonlocal joints
        base_pose = jnp.array(urdf_base_frame.wxyz.tolist() + urdf_base_frame.position.tolist())

        for target_frame_handle, target_name_handle, target_tf_handle in zip(
            target_frame_handles, target_name_handles, target_tf_handles
        ):
            target_joint_idx = kin.joint_names.index(target_name_handle.value)
            T_target_world = (
                jaxlie.SE3(base_pose)
                @ jaxlie.SE3(kin.forward_kinematics(joints)[target_joint_idx])
            )

            target_frame_handle.position = onp.array(T_target_world.translation())
            target_frame_handle.wxyz = onp.array(T_target_world.rotation().wxyz)
            target_tf_handle.position = onp.array(T_target_world.translation())
            target_tf_handle.wxyz = onp.array(T_target_world.rotation().wxyz)

    while True:
        # Don't do anything if there are no target joints...
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
        target_poses = jaxlie.SE3(
            jnp.stack([pose.wxyz_xyz for pose in target_pose_list])
        )

        # Solve!
        start_time = time.time()
        base_pose, joints = solve_ik(
            kin,
            target_poses,
            tuple[int](target_joint_indices),
            pos_weight,
            rot_weight,
            rest_weight,
            limit_weight,
            rest_pose,
            jnp.ones(6) - get_freeze_base_xyz_xyz(),
        )
        timing_handle.value = (time.time() - start_time) * 1000

        # Update visualizations.
        urdf_base_frame.position = onp.array(base_pose.translation())
        urdf_base_frame.wxyz = onp.array(base_pose.rotation().wxyz)
        urdf_vis.update_cfg(onp.array(joints))
        for target_frame_handle, target_joint_idx in zip(
            target_frame_handles, target_joint_indices
        ):
            T_target_world = base_pose @ jaxlie.SE3(
                kin.forward_kinematics(joints)[target_joint_idx]
            )
            target_frame_handle.position = onp.array(T_target_world.translation())
            target_frame_handle.wxyz = onp.array(T_target_world.rotation().wxyz)


if __name__ == "__main__":
    tyro.cli(main)
