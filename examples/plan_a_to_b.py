from __future__ import annotations

from copy import deepcopy
from typing import Literal

from robot_descriptions.loaders.yourdfpy import load_robot_description

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import viser.transforms as vtf
import yourdfpy
import warnings
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int, Bool
import trimesh
from typing import Dict, List, Tuple, Optional, cast
import tyro

from jaxmp.collision import SphereSDF, MeshSDF
from jaxmp.collbody import Spheres
from jaxmp.urdf import JaxUrdfwithSphereCollision
import jaxls

import viser
import viser.extras
import time


YUMI_REST_POSE = [
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
]

self_coll_ignore = [
    ("gripper_l_finger_r", "gripper_l_finger_l"),
    ("gripper_r_finger_r", "gripper_r_finger_l"),

    ("base_link", "yumi_link_1_l"),
    ("base_link", "yumi_link_2_l"),
    ("base_link", "yumi_link_3_l"),
    ("base_link", "yumi_link_4_l"),
    ("base_link", "yumi_link_5_l"),

    ("base_link", "yumi_link_1_r"),
    ("base_link", "yumi_link_2_r"),
    ("base_link", "yumi_link_3_r"),
    ("base_link", "yumi_link_4_r"),
    ("base_link", "yumi_link_5_r"),

    ("yumi_link_1_l", "yumi_link_3_l"),
    ("yumi_link_1_l", "yumi_link_4_l"),
    ("yumi_link_1_l", "yumi_link_5_l"),
    ("yumi_link_1_l", "yumi_link_1_r"),

    ("yumi_link_1_r", "yumi_link_3_r"),
    ("yumi_link_1_r", "yumi_link_4_r"),
    ("yumi_link_1_r", "yumi_link_5_r"),
    ("yumi_link_1_r", "yumi_link_1_l"),

    ("yumi_link_2_l", "yumi_link_4_l"),
    ("yumi_link_2_l", "yumi_link_5_l"),
    ("yumi_link_2_l", "yumi_link_1_r"),

    ("yumi_link_2_r", "yumi_link_4_r"),
    ("yumi_link_2_r", "yumi_link_5_r"),
    ("yumi_link_2_r", "yumi_link_1_l"),

    ("yumi_link_3_l", "yumi_link_5_l"),
    ("yumi_link_3_l", "yumi_link_6_l"),
    ("yumi_link_3_l", "yumi_link_7_l"),
    ("yumi_link_3_l", "gripper_l_base"),

    ("yumi_link_3_r", "yumi_link_5_r"),
    ("yumi_link_3_r", "yumi_link_6_r"),
    ("yumi_link_3_r", "yumi_link_7_r"),
    ("yumi_link_3_r", "gripper_r_base"),

    ("yumi_link_4_l", "yumi_link_6_l"),
    ("yumi_link_4_l", "yumi_link_7_l"),
    ("yumi_link_4_l", "gripper_l_base"),

    ("yumi_link_4_r", "yumi_link_6_r"),
    ("yumi_link_4_r", "yumi_link_7_r"),
    ("yumi_link_4_r", "gripper_r_base"),

    ("yumi_link_5_r", "yumi_link_7_r"),
    ("yumi_link_5_r", "gripper_r_base"),

    ("yumi_link_5_l", "yumi_link_7_l"),
    ("yumi_link_5_l", "gripper_l_base"),

    ("yumi_link_6_l", "gripper_l_base"),
    ("yumi_link_6_r", "gripper_r_base"),
]


def main():
    """
    Shortest path possible, but prioritizng limiting joint angles and avoiding self-collision.
    """
    pos_weight: float = 50.0
    rot_weight: float = 0.0
    coll_weight: float = 50.0
    rest_weight: float = 0.001
    limit_weight: float = 100_000.0
    smoothness_weight: float = 1.0

    timesteps = 20

    # This is so sensitive to n_waypoints, and the weights...

    yourdf = load_robot_description("yumi_description")
    jax_urdf = JaxUrdfwithSphereCollision.from_urdf(yourdf, self_coll_ignore=self_coll_ignore)
    # rest_pose = onp.array([0.0] * yourdf.num_dofs)
    rest_pose = onp.array(YUMI_REST_POSE)

    target_idx_r = jax_urdf.joint_names.index("yumi_joint_6_r")
    target_idx_l = jax_urdf.joint_names.index("yumi_joint_6_l")

    pose_r_start = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi),
        translation=jnp.array([0.4, 0.0, 0.4])
    )
    pose_r_end = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi),
        translation=jnp.array([0.4, 0.0, 0.2])
    )
    pose_l_start = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi),
        translation=jnp.array([0.2, 0.0, 0.3])
    )
    pose_l_end = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi),
        translation=jnp.array([0.6, -0.0, 0.3])
    )

    # Create a variable for the robot's joint positions.
    class JointVar(
        jaxls.Var[jax.Array],
        default=jnp.zeros(yourdf.num_dofs),
    ): ...

    joint_vars = [JointVar(id=idx) for idx in range(timesteps)]

    # Define the factors.
    def joint_angle_prior_factor(vals, var):
        return (
            (vals[var] - jax_urdf.limits_lower).clip(max=0.0) * limit_weight +
            (vals[var] - jax_urdf.limits_upper).clip(min=0.0) * limit_weight +
            (vals[var] - rest_pose) * rest_weight
        )

    def self_coll_factor(vals: jaxls.VarValues, var: JointVar, var_t: Optional[JointVar] = None):
        if var_t is None:
            return (
                jnp.maximum(jax_urdf.d_self(cfg=vals[var]) + 0.05, 0.0)**2
            ) * coll_weight / timesteps

        joint = vals[var]
        joint_t = vals[var_t]
        coll_loss = jnp.concatenate([
            jnp.maximum(jax_urdf.d_self(cfg=(joint*t + joint_t*(1-t))), 0.0)**2
            for t in jnp.linspace(0, 1, 5)
        ]) / 5
        coll_loss = coll_loss * coll_weight / timesteps
        return coll_loss

    def ik_on_joint(vals, var, target_pose, target_joint_idx):
        joint_angles = vals[var]
        pose_loss = (
            (
                jaxlie.SE3(jax_urdf.forward_kinematics(joint_angles)[target_joint_idx]) @
                jaxlie.SE3.from_translation(jnp.array([0.0, 0.0, 0.12]))
            ).inverse()
            @ target_pose
        ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3) / timesteps
        return pose_loss

    def smoothness_joint_factor(vals, var, var_t):
        return (vals[var] - vals[var_t]) * smoothness_weight / (timesteps - 1)

    # First solve for IK to get the initial and goal positions, in a collision-free manner.
    endpoint_vars = [joint_vars[0], joint_vars[-1]]
    factors: List[jaxls.Factor] = [
        jaxls.Factor.make(joint_angle_prior_factor, (joint_vars[0],)),  # Lower joint limits.
        jaxls.Factor.make(self_coll_factor, (joint_vars[0],)),  # Avoid self-collision.
        jaxls.Factor.make(ik_on_joint, (joint_vars[0], pose_r_start, target_idx_r,)),  # Position constraints.
        jaxls.Factor.make(ik_on_joint, (joint_vars[0], pose_l_start, target_idx_l,)),  # Position constraints.
        jaxls.Factor.make(joint_angle_prior_factor, (joint_vars[1],)),  # Lower joint limits.
        jaxls.Factor.make(self_coll_factor, (joint_vars[1],)),  # Avoid self-collision.
        jaxls.Factor.make(ik_on_joint, (joint_vars[1], pose_r_end, target_idx_r,)),  # Position constraints.
        jaxls.Factor.make(ik_on_joint, (joint_vars[1], pose_l_end, target_idx_l,)),  # Position constraints.
    ]
    graph = jaxls.FactorGraph.make(factors, endpoint_vars)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(endpoint_vars, [jnp.array(rest_pose) for _ in range(len(endpoint_vars))]),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        verbose=False,
    )

    start_joint = solution[endpoint_vars[0]]
    end_joint = solution[endpoint_vars[1]]
    # init_joint = [start_joint + (end_joint - start_joint) * i / (timesteps - 1) for i in range(timesteps)]
    init_joint = [start_joint + (rest_pose - start_joint) * i / (timesteps // 2 - 1) for i in range(timesteps//2)]
    init_joint += [rest_pose + (end_joint - start_joint) * i / (timesteps // 2 - 1) for i in range(timesteps//2)]
    
    # Assemble the factor graph!
    factors: List[jaxls.Factor] = []

    factors.extend([
        jaxls.Factor.make(ik_on_joint, (joint_vars[0], pose_r_start, target_idx_r,)),  # Position constraints.
        jaxls.Factor.make(ik_on_joint, (joint_vars[-1], pose_r_end, target_idx_r,)),  # Position constraints.
        jaxls.Factor.make(ik_on_joint, (joint_vars[0], pose_l_start, target_idx_l,)),  # Position constraints.
        jaxls.Factor.make(ik_on_joint, (joint_vars[-1], pose_l_end, target_idx_l,)),  # Position constraints.
    ])

    for tstep, pose_var in enumerate(joint_vars):
        factors.extend(
            [
                jaxls.Factor.make(joint_angle_prior_factor, (pose_var,)),  # Lower joint limits.
            ]
        )
        if tstep > 0:
            factors.extend(
                [
                    jaxls.Factor.make(smoothness_joint_factor, (pose_var, joint_vars[tstep - 1])),
                    jaxls.Factor.make(self_coll_factor, (pose_var, joint_vars[tstep - 1])),  # Avoid self-collision.
                ]
            )

    start = time.time()
    graph = jaxls.FactorGraph.make(factors, joint_vars)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(joint_vars, init_joint),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    print("Elapsed time (jit + exec): ", time.time() - start)

    start = time.time()
    graph = jaxls.FactorGraph.make(factors, joint_vars)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(joint_vars, init_joint),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    traj = onp.array([solution[pose_var] for pose_var in joint_vars])
    print("Elapsed time (exec): ", time.time() - start)

    # Assemble the factor graph!
    factors: List[jaxls.Factor] = []
    factors.extend([
        jaxls.Factor.make(ik_on_joint, (joint_vars[0], pose_r_start, target_idx_r,)),  # Position constraints.
        jaxls.Factor.make(ik_on_joint, (joint_vars[-1], pose_r_end, target_idx_r,)),  # Position constraints.
        jaxls.Factor.make(ik_on_joint, (joint_vars[0], pose_l_start, target_idx_l,)),  # Position constraints.
        jaxls.Factor.make(ik_on_joint, (joint_vars[-1], pose_l_end, target_idx_l,)),  # Position constraints.
    ])
    for tstep, pose_var in enumerate(joint_vars):
        factors.extend(
            [
                jaxls.Factor.make(joint_angle_prior_factor, (pose_var,)),  # Lower joint limits.
            ]
        )
        if tstep > 0:
            factors.extend(
                [
                    jaxls.Factor.make(smoothness_joint_factor, (pose_var, joint_vars[tstep - 1])),
                    # jaxls.Factor.make(self_coll_factor, (pose_var, joint_vars[tstep - 1])),  # Avoid self-collision.
                ]
            )
    start = time.time()
    graph = jaxls.FactorGraph.make(factors, joint_vars)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(joint_vars, init_joint),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    print("Elapsed time (jit + exec): ", time.time() - start)
    traj_no_coll = onp.array([solution[pose_var] for pose_var in joint_vars])

    server = viser.ViserServer()
    urdf_orig = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf_orig"
    )
    urdf_no_coll = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf_orig_no_coll", mesh_color_override=(220, 100, 100),
    )
    server.scene.add_frame("r_start", wxyz=pose_r_start.wxyz_xyz[:4], position=pose_r_start.wxyz_xyz[4:], axes_length=0.05, axes_radius=0.002)
    server.scene.add_frame("r_end", wxyz=pose_r_end.wxyz_xyz[:4], position=pose_r_end.wxyz_xyz[4:], axes_length=0.05, axes_radius=0.002)
    server.scene.add_frame("l_start", wxyz=pose_l_start.wxyz_xyz[:4], position=pose_l_start.wxyz_xyz[4:], axes_length=0.05, axes_radius=0.002)
    server.scene.add_frame("l_end", wxyz=pose_l_end.wxyz_xyz[:4], position=pose_l_end.wxyz_xyz[4:], axes_length=0.05, axes_radius=0.002)

    # Visualize!
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )

    @slider.on_update
    def _(_) -> None:
        urdf_orig.update_cfg(traj[slider.value])
        urdf_no_coll.update_cfg(traj_no_coll[slider.value])
        server.scene.add_mesh_trimesh("spheres", jax_urdf.spheres(cfg=traj[slider.value]).to_trimesh())

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        time.sleep(1.0 / 10.0)

main()