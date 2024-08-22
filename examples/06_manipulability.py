""" 06_manipulability.py.
Similar to 01_kinematics.py, but including manipulability as the cost function!

Hard to verify if it works, but the cost does make the robot pose different.
"""

from robot_descriptions.loaders.yourdfpy import load_robot_description

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp

import jaxls

from jaxmp.kinematics import JaxKinTree

def main(
    pos_weight: float = 5.0,
    rot_weight: float = 0.5,
    limit_weight: float = 100.0,
    manipulability_weight: float = 0.1,
):
    urdf = load_robot_description("yumi_description")
    # urdf = load_robot_description("panda_description")
    # urdf = load_robot_description("ur5_description")
    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    import viser
    import viser.extras
    import time

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/urdf")
    urdf_vis_no_manip = viser.extras.ViserUrdf(server, urdf, root_node_name="/no_manip", mesh_color_override=(220, 100, 100))
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

    # New cost: Manipulability.
    def manipulability_cost(vals, var, target_joint_idx: int):
        joint_cfg: jax.Array = vals[var]

        # Jacobian between wxyz_xyz, and dof.
        jacobian = jax.jacfwd(kin.forward_kinematics)(joint_cfg)[target_joint_idx]
        norm = jnp.linalg.norm(jacobian, ord='nuc')

        return jnp.maximum(7.0 - norm[None], 0.0) * manipulability_weight

    def solve_ik():
        joint_vars = [JointVar(id=0)]

        target_joint_idx = kin.joint_names.index(target_name_handle.value)
        target_pose = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
        factors = [
            jaxls.Factor.make(ik_to_joint, (joint_vars[0], target_pose, target_joint_idx)),
            jaxls.Factor.make(limit_cost, (joint_vars[0],)),
            jaxls.Factor.make(manipulability_cost, (joint_vars[0], target_joint_idx)),
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

        T_target_world = jaxlie.SE3(kin.forward_kinematics(joints)[target_joint_idx]).wxyz_xyz
        target_frame_handle.position = onp.array(T_target_world)[4:]
        target_frame_handle.wxyz = onp.array(T_target_world)[:4]

    while True:
        solve_ik()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
