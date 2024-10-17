"""
Test to see if we can batch IK -- by solving IK for multiple target poses at once.
The target poses come from antipodal grasps on a box.
"""

from typing import Optional
from pathlib import Path
import time
import tyro
import viser
import viser.extras
import trimesh

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp

from jaxmp.extras.urdf_loader import load_urdf
from jaxmp.kinematics import JaxKinTree
from jaxmp.jaxls.solve_ik import solve_ik
from jaxmp.extras.grasp_antipodal import AntipodalGrasps

# set device to cpu
jax.config.update("jax_platform_name", "cpu")


def main(
    pos_weight: float = 10.0,
    rot_weight: float = 2.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    robot_description: Optional[str] = "panda",
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
    urdf = load_urdf(robot_description, robot_urdf_path)

    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2
    rest_pose = rest_pose.at[-2:].set(kin.limits_upper[-2:])

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    tf_size_handle = server.gui.add_slider(
        "Gizmo size", min=0.01, max=0.4, step=0.01, initial_value=0.2
    )

    # Put robot to rest pose :-)
    base_pose = jaxlie.SE3.identity()
    joints = rest_pose

    urdf_base_frame.position = onp.array(base_pose.translation())
    urdf_base_frame.wxyz = onp.array(base_pose.rotation().wxyz)
    urdf_vis.update_cfg(onp.array(joints))

    # Add joints.
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0],
    )
    target_tf_handle = server.scene.add_transform_controls(
        "target_transform", scale=tf_size_handle.value
    )
    target_frame_handle = server.scene.add_frame(
        "target",
        axes_length=0.5 * tf_size_handle.value,
        axes_radius=0.05 * tf_size_handle.value,
        origin_radius=0.1 * tf_size_handle.value,
    )

    # Let the user change the size of the transformcontrol gizmo.
    @tf_size_handle.on_update
    def _(_):
        target_tf_handle.scale = tf_size_handle.value
        target_frame_handle.axes_length = 0.5 * tf_size_handle.value
        target_frame_handle.axes_radius = 0.05 * tf_size_handle.value
        target_frame_handle.origin_radius = 0.1 * tf_size_handle.value

    # Create a graspable object!
    workspace_obj = trimesh.creation.box(extents=[0.05]*3)
    obj_grasps = AntipodalGrasps.from_sample_mesh(workspace_obj, max_samples=20)
    server.scene.add_mesh_trimesh("target_transform/mesh", workspace_obj)
    grasp_handles = []
    
    # TODO(cmk) remove hardcoding for `along_axis` (panda grasp axis is along y.)
    for idx in range(len(obj_grasps)):
        grasp_handles.append(
            server.scene.add_mesh_trimesh(
                f"target_transform/grasp_{idx}",
                obj_grasps.to_trimesh(indices=(idx,), axes_height=0.2, along_axis="y"),
                visible=False
            ),
        )
    grasp_idx_handle = server.gui.add_slider(
        "Grasp index", min=0, max=len(obj_grasps)*2 - 1, step=1, initial_value=0
    )
    @grasp_idx_handle.on_update
    def _(_):
        for idx, handle in enumerate(grasp_handles):
            handle.visible = idx == grasp_idx_handle.value % len(obj_grasps) 

    while True:
        target_joint_idx = kin.joint_names.index(target_name_handle.value)
        T_obj_world = jaxlie.SE3(
            jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position])
        )
        T_grasp_obj = jaxlie.SE3(
            jnp.concatenate([
                obj_grasps.to_se3(along_axis="y").wxyz_xyz,
                obj_grasps.to_se3(along_axis="y", flip_axis=True).wxyz_xyz,
            ])
        )
        T_grasp_world = T_obj_world @ T_grasp_obj
        T_grasp_world = jaxlie.SE3(
            T_grasp_world.wxyz_xyz.reshape(-1, 1, 7)
        )

        # Solve!
        start_time = time.time()
        solve_ik_batch = jax.vmap(
            solve_ik, in_axes=(None, 0, None, None, None, None, None, None, None)
        )
        
        # TODO(cmk) consider collisions here!!
        # TODO(cmk) how to bias towards gripper facing down? -- maybe collision will address this.
        base_pose, joints = solve_ik_batch(
            kin,
            T_grasp_world,
            (target_joint_idx,),
            pos_weight,
            rot_weight,
            rest_weight,
            limit_weight,
            rest_pose,
            jnp.array([1, 1, 1, 1, 0, 1]),
        )
        timing_handle.value = (time.time() - start_time) * 1000

        # Update visualizations.
        grasp_idx = grasp_idx_handle.value
        urdf_base_frame.position = onp.array(base_pose.translation())[grasp_idx]
        urdf_base_frame.wxyz = onp.array(base_pose.rotation().wxyz)[grasp_idx]
        urdf_vis.update_cfg(onp.array(joints[grasp_idx]))

        T_target_world = base_pose @ jaxlie.SE3(
            kin.forward_kinematics(joints[grasp_idx])[target_joint_idx]
        )
        target_frame_handle.position = onp.array(T_target_world.translation())[grasp_idx]
        target_frame_handle.wxyz = onp.array(T_target_world.rotation().wxyz)[grasp_idx]


if __name__ == "__main__":
    tyro.cli(main)
