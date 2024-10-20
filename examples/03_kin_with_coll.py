"""03_kin_with_coll.py
Similar to 01_kinematics.py, but with collision detection.
"""

from typing import Optional
from pathlib import Path
import time
import jax
from jaxmp.coll import collide

import tyro

import jax.numpy as jnp
import jaxlie
import numpy as onp
import jax_dataclasses as jdc

import jaxls

import viser
import viser.extras

from jaxmp.kinematics import JaxKinTree
from jaxmp.jaxls.robot_factors import RobotFactors
from jaxmp.extras.urdf_loader import load_urdf
from jaxmp.coll import RobotColl, Plane


def main(
    robot_description: str = "yumi_description",
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    coll_weight: float = 5.0,
    world_coll_weight: float = 100.0,
    robot_urdf_path: Optional[Path] = None,
):
    urdf = load_urdf(robot_description, robot_urdf_path)
    robot_coll = RobotColl.from_urdf(urdf)
    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    urdf_vis.update_cfg(onp.array(rest_pose))
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Create ground plane as an obstacle (world collision)!
    obstacle = Plane.from_point_and_normal(
        jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    )
    server.scene.add_mesh_trimesh("ground_plane", obstacle.to_trimesh())
    server.scene.add_grid(
        "ground", width=3, height=3, cell_size=0.1, position=(0.0, 0.0, 0.001)
    )
    self_coll_value = server.gui.add_number(
        "max. coll (self)", 0.0, step=0.01, disabled=True
    )
    world_coll_value = server.gui.add_number(
        "max. coll (world)", 0.0, step=0.01, disabled=True
    )
    visualize_coll = server.gui.add_checkbox("Visualize spheres", False)

    # Add GUI elements, to let user interact with the robot joints.
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
            initial_value=urdf.joint_names[0],
        )
        target_tf_handle = server.scene.add_transform_controls(
            f"target_transform_{idx}", scale=0.2
        )
        target_frame_handle = server.scene.add_frame(f"target_{idx}", axes_length=0.1)

        target_name_handles.append(target_name_handle)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    # Create factor graph.
    JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)

    collbody_handle = None

    @jdc.jit
    def solve_ik(
        target_pose: jaxlie.SE3,
        target_joint_indices: jax.Array,
    ) -> jnp.ndarray:
        joint_vars = [JointVar(id=0)]

        factors: list[jaxls.Factor] = [
            jaxls.Factor.make(
                RobotFactors.limit_cost,
                (
                    kin,
                    joint_vars[0],
                    jnp.array([limit_weight] * kin.num_actuated_joints),
                ),
            ),
            jaxls.Factor.make(
                RobotFactors.self_coll_cost,
                (
                    kin,
                    robot_coll,
                    joint_vars[0],
                    0.01,
                    jnp.full(robot_coll.coll.get_batch_axes(), coll_weight),
                ),
            ),
            jaxls.Factor.make(
                RobotFactors.world_coll_cost,
                (
                    kin,
                    robot_coll,
                    joint_vars[0],
                    obstacle,
                    0.05,
                    jnp.full(robot_coll.coll.get_batch_axes(), world_coll_weight),
                ),
            ),
            jaxls.Factor.make(
                RobotFactors.rest_cost,
                (
                    joint_vars[0],
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
                    ),
                )
            )

        graph = jaxls.FactorGraph.make(
            factors,
            joint_vars,
            use_onp=False,
        )
        solution = graph.solve(
            linear_solver="conjugate_gradient",
            initial_vals=jaxls.VarValues.make(joint_vars),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
            termination=jaxls.TerminationConfig(
                gradient_tolerance=1e-5, parameter_tolerance=1e-5
            ),
            verbose=False,
        )

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

        target_poses = jaxlie.SE3(
            jnp.stack([pose.wxyz_xyz for pose in target_pose_list])
        )

        start = time.time()
        joints = solve_ik(target_poses, tuple[int](target_joint_indices))
        # Update timing info.
        timing_handle.value = (time.time() - start) * 1000

        urdf_vis.update_cfg(onp.array(joints))

        for target_frame_handle, target_joint_idx in zip(
            target_frame_handles, target_joint_indices
        ):
            T_target_world = kin.forward_kinematics(joints)[target_joint_idx]
            target_frame_handle.position = onp.array(T_target_world)[4:]
            target_frame_handle.wxyz = onp.array(T_target_world)[:4]

        urdf_vis.update_cfg(onp.array(joints))

        coll = robot_coll.coll.transform(
            jaxlie.SE3(
                kin.forward_kinematics(joints)[..., robot_coll.link_joint_idx, :]
            )
        )
        self_coll_value.value = (
            (
                collide(coll, coll.reshape(-1, 1)).dist.squeeze()
                * robot_coll.self_coll_matrix
            )
            .min()
            .item()
        )
        world_coll_value.value = collide(coll, obstacle).dist.min().item()
        if visualize_coll.value:
            collbody_handle = server.scene.add_mesh_trimesh(
                "coll",
                robot_coll.coll.transform(
                    jaxlie.SE3(
                        kin.forward_kinematics(joints)[
                            ..., robot_coll.link_joint_idx, :
                        ]
                    )
                ).to_trimesh(),
            )
        elif collbody_handle is not None:
            collbody_handle.remove()


if __name__ == "__main__":
    tyro.cli(main)
