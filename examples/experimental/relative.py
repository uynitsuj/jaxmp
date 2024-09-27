""" relative.py
Tries to set the relative pose/orientation between the two yumi grippers.
Gripper names are:
- yumi_link_7_r
- yumi_link_7_l
"""

from typing import Optional
from pathlib import Path
import time
import tyro
import yourdfpy

from robot_descriptions.loaders.yourdfpy import load_robot_description

import jax
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
    # rest_pose = (kin.limits_upper + kin.limits_lower) / 2
    rest_pose = jnp.array([
        1.21442839,
        -1.03205606,
        -1.10072738,
        0.2987352,
        -1.85257716,
        1.25363652,
        -2.42181893,
        -1.24839656,
        -1.09802876,
        1.06634394,
        0.31386161,
        1.90125141,
        1.3205139,
        2.43563939,
        0.0,
        0.0,
    ])

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    urdf_vis.update_cfg(onp.array(rest_pose))
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    # Create a relative pose between the two grippers.
    right_idx, left_idx = kin.joint_names.index("yumi_link_7_r_joint"), kin.joint_names.index("yumi_link_7_l_joint")
    server.scene.add_frame(
        "_world",
        position=onp.array([0, 0.5, 0]),
        # wxyz=onp.array(jaxlie.SO3.from_x_radians(jnp.pi).wxyz),
        axes_length=0.2,
    )
    target_tf_rel_handle = server.scene.add_transform_controls(
        "_world/relative_pose",
        scale=0.2
    )
    target_frame_handle_l = server.scene.add_frame("left", axes_length=0.1)
    target_frame_handle_r = server.scene.add_frame("right", axes_length=0.1)

    rel_frame_handle = server.scene.add_frame("right_pred", axes_length=0.1)
    server.scene.add_label("right_pred/label", "target")

    target_tf_left_handle = server.scene.add_transform_controls(
        "target_left",
        scale=0.2
    )

    # Create factor graph.
    JointVar = robot_factors.get_var_class(default_val=rest_pose)

    def rel_pose(
        vals: jaxls.VarValues,
        val: jaxls.Var,
        target_rel_pose: jaxlie.SE3,
        weight: jax.Array,
    ) -> jax.Array:
        """Get the relative pose between the two grippers."""
        joints = vals[val]
        T_r_world = kin.forward_kinematics(joints)[right_idx]
        T_l_world = kin.forward_kinematics(joints)[left_idx]

        # From right to left.
        rel_pose = jaxlie.SE3(T_l_world).inverse() @ jaxlie.SE3(T_r_world)

        return (target_rel_pose.inverse() @ rel_pose).log() * weight

    # it _is_ actually pretty noticably slow (9ms -> 60ms? -- even considering it's two frames.)
    @jdc.jit
    def solve_ik(
        target_left_pose: jaxlie.SE3,
        target_rel_pose: jaxlie.SE3,
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

        # Compute the target pose for the right gripper based on the relative pose
        factors.append(
            jaxls.Factor.make(
                rel_pose,
                (
                    joint_vars[0],
                    target_rel_pose,
                    jnp.array([pos_weight] * 3 + [rot_weight] * 3),
                ),
            )
        )

        # Compute the target pose for the left gripper
        factors.append(
            jaxls.Factor.make(
                robot_factors.ik_cost,
                (
                    joint_vars[0],
                    target_left_pose,
                    left_idx,
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

        # Update visualization.
        joints = solution[joint_vars[0]]
        return joints

    while True:
        # Assuming the first target pose is the relative pose
        relative_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.array(target_tf_rel_handle.wxyz)),
            target_tf_rel_handle.position
        )
        left_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.array(target_tf_left_handle.wxyz)),
            target_tf_left_handle.position
        )

        start_time = time.time()
        joints = solve_ik(left_pose, relative_pose)
        timing_handle.value = (time.time() - start_time) * 1000

        urdf_vis.update_cfg(onp.array(joints))

        T_right_world = kin.forward_kinematics(joints)[right_idx]
        target_frame_handle_r.position = onp.array(T_right_world)[4:]
        target_frame_handle_r.wxyz = onp.array(T_right_world)[:4]

        T_left_world = kin.forward_kinematics(joints)[left_idx]
        target_frame_handle_l.position = onp.array(T_left_world)[4:]
        target_frame_handle_l.wxyz = onp.array(T_left_world)[:4]

        T_right_world = jaxlie.SE3(T_left_world) @ relative_pose
        rel_frame_handle.position = onp.array(T_right_world.translation())
        rel_frame_handle.wxyz = onp.array(T_right_world.rotation().wxyz)


if __name__ == "__main__":
    tyro.cli(main)
