"""
Test to see if we can batch IK.
"""

from typing import Optional
from pathlib import Path
import time
import tyro
import yourdfpy

from robot_descriptions.loaders.yourdfpy import load_robot_description

import jax
# set to cpu
jax.config.update("jax_platform_name", "cpu")

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
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0],
    )

    urdf_handles: list[viser.extras.ViserUrdf] = []
    target_tf_handles: list[viser.TransformControlsHandle] = []
    target_frame_handles: list[viser.FrameHandle] = []
    add_tf_button = server.gui.add_button("Add tf!")

    def create_target_tf(_):
        # Show target joint name.
        idx = len(target_tf_handles)

        urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name=f"/robot_{idx}")
        urdf_vis.update_cfg(onp.array(rest_pose))

        target_tf_handle = server.scene.add_transform_controls(
            f"/robot_{idx}/target_transform", scale=0.2
        )
        target_frame_handle = server.scene.add_frame(f"robot_{idx}/target", axes_length=0.1)

        urdf_handles.append(urdf_vis)
        target_tf_handles.append(target_tf_handle)
        target_frame_handles.append(target_frame_handle)

    add_tf_button.on_click(create_target_tf)

    # Create factor graph.
    JointVar = robot_factors.get_var_class(default_val=rest_pose)

    @jdc.jit
    def solve_ik(
        target_pose_wxyz_xyz: jax.Array,
        target_joint_idx: jdc.Static[int],
    ) -> jnp.ndarray:
        joint_vars = [JointVar(id=0)]

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

        factors.append(
            jaxls.Factor.make(
                robot_factors.ik_cost,
                (
                    joint_vars[0],
                    jaxlie.SE3(target_pose_wxyz_xyz),
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
            termination=jaxls.TerminationConfig(
                gradient_tolerance=1e-5, parameter_tolerance=1e-5
            ),
            verbose=False,
        )


        # Update visualization.
        joints = solution[joint_vars[0]]
        return joints

    while True:
        if len(target_tf_handles) == 0:
            time.sleep(0.1)
            continue

        target_joint_idx = kin.joint_names.index(target_name_handle.value)

        target_pose_list = [
            jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
            for target_tf_handle in target_tf_handles
        ] * 200
        target_poses = jnp.stack([pose.wxyz_xyz for pose in target_pose_list])

        start_time = time.time()
        joints = jax.vmap(solve_ik, in_axes=(0, None))(target_poses, target_joint_idx)
        timing_handle.value = (time.time() - start_time) * 1000

        for idx, target_frame_handle in enumerate(target_frame_handles):
            curr_joints = joints[idx]
            urdf_handles[idx].update_cfg(onp.array(curr_joints))
            T_target_world = kin.forward_kinematics(curr_joints)[target_joint_idx]
            target_frame_handle.position = onp.array(T_target_world)[4:]
            target_frame_handle.wxyz = onp.array(T_target_world)[:4]


if __name__ == "__main__":
    tyro.cli(main)
