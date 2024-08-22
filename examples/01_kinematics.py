""" 01_kinematics.py
Tests robot forward + inverse kinematics using JaxMP.
"""

import time

from robot_descriptions.loaders.yourdfpy import load_robot_description

import viser
import viser.extras

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp

import jaxls

from jaxmp.kinematics import JaxKinTree

def main(
    robot_description: str = "yumi_description",
    pos_weight: float = 2.0,
    rot_weight: float = 0.5,
    limit_weight: float = 100.0,
):
    urdf = load_robot_description(robot_description)
    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    target_tf_handle = server.scene.add_transform_controls("target transform", scale=0.2)
    target_frame_handle = server.scene.add_frame("target", axes_length=0.1)

    # Show target joint name, and current joint positions.
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0]
    )

    # Create factor graph.
    class JointVar(jaxls.Var[jax.Array], default=rest_pose): ...

    def ik_to_joint(vals: jaxls.VarValues, var: JointVar, target_pose: jaxlie.SE3, target_joint_idx: int):
        joint_cfg: jax.Array = vals[var]
        pose_res = (
            jaxlie.SE3(kin.forward_kinematics(joint_cfg)[target_joint_idx]).inverse()
            @ target_pose
        ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3)
        return pose_res

    def limit_cost(vals, var):
        joint_cfg: jax.Array = vals[var]
        return (
            jnp.maximum(0.0, joint_cfg - kin.limits_upper) +
            jnp.maximum(0.0, kin.limits_lower - joint_cfg)
        ) * limit_weight

    def solve_ik():
        joint_vars = [JointVar(id=0)]

        target_joint_idx = kin.joint_names.index(target_name_handle.value)
        target_pose = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))

        graph = jaxls.FactorGraph.make(
            [
                jaxls.Factor.make(ik_to_joint, (joint_vars[0], target_pose, target_joint_idx)),
                jaxls.Factor.make(limit_cost, (joint_vars[0],)),
            ],
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
        urdf_vis.update_cfg(onp.array(joints))
        T_target_world = jaxlie.SE3(kin.forward_kinematics(joints)[target_joint_idx]).wxyz_xyz
        target_frame_handle.position = onp.array(T_target_world)[4:]
        target_frame_handle.wxyz = onp.array(T_target_world)[:4]

    while True:
        solve_ik()
        time.sleep(0.01)


if __name__ == "__main__":
    main()