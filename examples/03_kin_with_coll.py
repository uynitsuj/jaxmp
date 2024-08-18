""" 03_kin_with_coll.py
Similar to 01_kinematics.py, but with collision detection.
"""

from robot_descriptions.loaders.yourdfpy import load_robot_description

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp

import jaxls

from jaxmp.collbody import sdf_to_colldist
from jaxmp.kinematics import JaxCollKinematics

def main(
    pos_weight: float = 5.0,
    rot_weight: float = 0.5,
    limit_weight: float = 100.0,
    coll_weight: float = 1.0,
):
    urdf = load_robot_description("yumi_description")
    kin = JaxCollKinematics.from_urdf(
        urdf,
        self_coll_ignore=[
            ("gripper_l_finger_l", "gripper_l_finger_r"),
            ("gripper_r_finger_l", "gripper_r_finger_r"),
        ]
    )  # JaxKinematics -> JaxCollKinematics!
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    import viser
    import viser.extras
    import time

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    target_tf_handle = server.scene.add_transform_controls("target transform", scale=0.2)
    target_frame_handle = server.scene.add_frame("target", axes_length=0.1)

    # Show target joint name, and current joint positions.
    visualize_spheres = server.gui.add_checkbox("Show spheres", initial_value=False)
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

    # New cost, for collision detection.
    def coll_self(vals, var):
        return sdf_to_colldist(kin.d_self(cfg=vals[var])) * coll_weight

    sphere_handle = None
    def solve_ik():
        nonlocal sphere_handle
        joint_vars = [JointVar(id=0)]

        target_joint_idx = kin.joint_names.index(target_name_handle.value)
        target_pose = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))

        graph = jaxls.FactorGraph.make(
            [
                jaxls.Factor.make(ik_to_joint, (joint_vars[0], target_pose, target_joint_idx)),
                jaxls.Factor.make(limit_cost, (joint_vars[0],)),
                jaxls.Factor.make(coll_self, (joint_vars[0],)),
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
        T_target_world = jaxlie.SE3(  # pylint: disable=invalid-name
            kin.forward_kinematics(joints)[target_joint_idx]
        ).wxyz_xyz

        # Update visualization.
        urdf_vis.update_cfg(onp.array(joints))
        target_frame_handle.position = onp.array(T_target_world)[4:]
        target_frame_handle.wxyz = onp.array(T_target_world)[:4]

        if visualize_spheres.value:
            sphere_handle = server.scene.add_mesh_trimesh(
                "spheres",
                kin.spheres(joints).to_trimesh()
            )
        elif sphere_handle is not None:
            sphere_handle.remove()

    while True:
        solve_ik()
        time.sleep(0.01)


if __name__ == "__main__":
    main()