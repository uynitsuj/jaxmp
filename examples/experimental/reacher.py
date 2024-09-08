# """ 08_reacher.py
# Reach a target with the specified end-effector -- while avoiding collisions with it.
# This is difficult, since 1) EE should be close to target, while 2) avoiding collisions.
# """

# import time
# from robot_descriptions.loaders.yourdfpy import load_robot_description

# import jax
# import jax.numpy as jnp
# import jaxlie
# import numpy as onp
# import viser
# import viser.extras

# import jaxls

# from jaxmp.collision import SphereColl, sdf_to_colldist
# from jaxmp.kinematics import JaxCollKinematics

# def main(
#     pos_weight: float = 2.0,
#     rot_weight: float = 0.5,
#     limit_weight: float = 100.0,
#     coll_weight: float = 1.0,
#     world_coll_weight: float = 10.0,
# ):
#     yourdf = load_robot_description("yumi_description")
#     kin = JaxCollKinematics.from_urdf(
#         yourdf,
#         self_coll_ignore=[
#             ("gripper_l_finger_l", "gripper_l_finger_r"),
#             ("gripper_r_finger_l", "gripper_r_finger_r"),
#         ]
#     )
#     rest_pose = (kin.limits_upper + kin.limits_lower) / 2

#     server = viser.ViserServer()
#     urdf_vis = viser.extras.ViserUrdf(server, yourdf)
#     target_tf_handle = server.scene.add_transform_controls("target transform", scale=0.2)
#     target_frame_handle = server.scene.add_frame("target", axes_length=0.1)

#     visualize_spheres = server.gui.add_checkbox("Show spheres", initial_value=False)
#     target_name_handle = server.gui.add_dropdown(
#         "target joint",
#         list(yourdf.joint_names),
#         initial_value=yourdf.joint_names[0]
#     )

#     target_obj_coll = SphereColl(
#         centers=jnp.array([[0.0, 0.0, 0.0]]),
#         radii=jnp.array([0.01])
#     )
#     target_obj_tf_handle = server.scene.add_transform_controls("target_obj", scale=0.2)

#     class JointVar(jaxls.Var[jax.Array], default=rest_pose): ...

#     def ik_to_joint(vals: jaxls.VarValues, var: JointVar, target_pose: jaxlie.SE3, target_joint_idx: int):
#         joint_cfg: jax.Array = vals[var]
#         pose_res = (
#             jaxlie.SE3(kin.forward_kinematics(joint_cfg)[target_joint_idx]).inverse()
#             @ (target_pose @ jaxlie.SE3.from_translation(jnp.array([0.0, 0.0, -0.12])))
#         ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3)
#         return pose_res

#     def limit_cost(vals, var):
#         joint_cfg: jax.Array = vals[var]
#         return (
#             jnp.maximum(0.0, joint_cfg - kin.limits_upper) +
#             jnp.maximum(0.0, kin.limits_lower - joint_cfg)
#         ) * limit_weight

#     # New cost, for collision detection.
#     def coll_self(vals, var):
#         return sdf_to_colldist(kin.d_self(cfg=vals[var])) * coll_weight

#     def coll_world(vals, var, target_tf):
#         _target_coll = target_obj_coll.transform(target_tf)
#         dist = sdf_to_colldist(kin.d_world(cfg=vals[var], other=_target_coll))
#         weighted_dist = dist * kin.coll_weight(
#             {
#                 "gripper_l_finger_l": 0.01,
#                 "gripper_l_finger_r": 0.01,
#                 "gripper_r_finger_l": 0.01,
#                 "gripper_r_finger_r": 0.01,
#             }
#         )
#         return weighted_dist

#     # Something to make sure that the middle pose is actually the end effector pose
#     def in_grasp(vals, var, target_tf):
#         pass

#     sphere_handle = None
#     def solve_ik():
#         nonlocal sphere_handle
#         joint_vars = [JointVar(id=0)]

#         target_joint_idx = kin.joint_names.index(target_name_handle.value)
#         # target_pose = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
#         target_pose = jaxlie.SE3(jnp.array([*target_obj_tf_handle.wxyz, *target_obj_tf_handle.position]))
#         target_coll_pose = jaxlie.SE3(jnp.array([*target_obj_tf_handle.wxyz, *target_obj_tf_handle.position]))

#         graph = jaxls.FactorGraph.make(
#             [
#                 jaxls.Factor.make(ik_to_joint, (joint_vars[0], target_pose, target_joint_idx)),
#                 jaxls.Factor.make(limit_cost, (joint_vars[0],)),
#                 jaxls.Factor.make(coll_self, (joint_vars[0],)),
#                 jaxls.Factor.make(coll_world, (joint_vars[0], target_coll_pose)),
#                 jaxls.Factor.make(lambda vals, var: (vals[var] - rest_pose) * 0.1, (joint_vars[0],)),
#             ],
#             joint_vars,
#             verbose=False,
#         )
#         solution = graph.solve(
#             initial_vals=jaxls.VarValues.make(joint_vars, [rest_pose]),
#             trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
#             termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
#             verbose=False,
#         )
#         joints = solution[joint_vars[0]]
#         T_target_world = jaxlie.SE3(  # pylint: disable=invalid-name
#             kin.forward_kinematics(joints)[target_joint_idx]
#         ).wxyz_xyz

#         # Update visualization.
#         urdf_vis.update_cfg(onp.array(joints))
#         target_frame_handle.position = onp.array(T_target_world)[4:]
#         target_frame_handle.wxyz = onp.array(T_target_world)[:4]

#         server.scene.add_mesh_trimesh("target_mesh", target_obj_coll.transform(target_coll_pose).to_trimesh())

#         if visualize_spheres.value:
#             sphere_handle = server.scene.add_mesh_trimesh(
#                 "spheres",
#                 kin.collbody(joints).to_trimesh()
#             )
#         elif sphere_handle is not None:
#             sphere_handle.remove()

#     while True:
#         solve_ik()
#         time.sleep(0.01)

# if __name__ == "__main__":
#     main()
