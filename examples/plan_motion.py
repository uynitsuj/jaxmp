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


def main():
    pos_weight: float = 50.00
    rot_weight: float = 0.50
    coll_weight: float = 100.0
    rest_weight: float = 0.001
    limit_weight: float = 100.0
    smoothness_weight: float = 0.5

    traj_npy = "trajectory.npy"

    yourdf = load_robot_description("yumi_description")
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

    # rest_pose = onp.array([0.0] * yourdf.num_dofs)
    rest_pose = onp.array(YUMI_REST_POSE)
    jax_urdf = JaxUrdfwithSphereCollision.from_urdf(yourdf, self_coll_ignore=self_coll_ignore)

    # Load the trajectory.
    raw_traj = onp.load(traj_npy)
    traj = onp.zeros((raw_traj.shape[0], 16))
    traj[:, :7] = raw_traj[:, 7:14]
    traj[:, 7:14] = raw_traj[:, :7]
    del raw_traj

    # Forward kinematics with loaded trajectory.
    print("Running FK...")
    start = time.time()
    timesteps = traj.shape[0]
    orig_Ts_world_joint = jax_urdf.forward_kinematics(jnp.array(traj))
    assert orig_Ts_world_joint.shape == (timesteps, jax_urdf.num_joints, 7)
    print("Done, elapsed time: ", time.time() - start)

    # Create a variable for the robot's joint positions.
    class RobotVar(
        jaxls.Var[jax.Array],
        default=jnp.zeros(yourdf.num_dofs),
    ): ...

    pose_vars = [RobotVar(id=idx) for idx in range(timesteps)]

    # Define the factors.
    def joint_angle_prior_factor(vals, var):
        return (
            (vals[var] - jax_urdf.limits_lower).clip(max=0.0) * limit_weight +
            (vals[var] - jax_urdf.limits_upper).clip(min=0.0) * limit_weight +
            (vals[var] - rest_pose) * rest_weight
        )

    def self_coll_factor(vals, var):
        return jnp.maximum(jax_urdf.d_self(cfg=vals[var]) + 0.05, 0.0)**2 * coll_weight / timesteps

    def ik_on_joint(vals, var, target_pose, target_joint_idx):
        joint_angles = vals[var]
        pose_loss = (
            jaxlie.SE3(jax_urdf.forward_kinematics(joint_angles)[target_joint_idx]).inverse()
            @ target_pose
        ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3) / timesteps
        return pose_loss

    def smoothness_joint_factor(vals, var, var_t):
        return (vals[var].inverse @ vals[var_t]).log() * smoothness_weight / (timesteps - 1)

    def smoothness_pose_factor(vals, var, var_t):
        return (vals[var] - vals[var_t]) * smoothness_weight / (timesteps - 1)
    
    # Assemble the factor graph!
    factors: List[jaxls.Factor] = []
    for tstep, pose_var in enumerate(pose_vars):
        # target_pose_r = jaxlie.SE3(orig_Ts_world_joint[tstep, 7] + jnp.array([0, 0, 0, 0, 0, 0.2, 0]))
        target_joint_r = jax_urdf.joint_names.index("yumi_joint_6_r")
        target_joint_l = jax_urdf.joint_names.index("yumi_joint_6_l")
        target_pose_r = jaxlie.SE3(orig_Ts_world_joint[tstep, target_joint_r])
        target_pose_l = jaxlie.SE3(orig_Ts_world_joint[tstep, target_joint_l])
        factors.extend(
            [
                jaxls.Factor.make(joint_angle_prior_factor, (pose_var,)),  # Lower joint limits.
                jaxls.Factor.make(self_coll_factor, (pose_var,)),  # Avoid self-collision.
                jaxls.Factor.make(ik_on_joint, (pose_var, target_pose_r, target_joint_r,)),  # Position constraints.
                jaxls.Factor.make(ik_on_joint, (pose_var, target_pose_l, target_joint_l,)),  # Position constraints.
            ]
        )
        if tstep > 0:
            factors.append(
                jaxls.Factor.make(smoothness_joint_factor, (pose_var, pose_vars[tstep - 1])),
                jaxls.Factor.make(smoothness_pose_factor, (pose_var, pose_vars[tstep - 1]))
            )

    start = time.time()
    graph = jaxls.FactorGraph.make(factors, pose_vars, verbose=False)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(pose_vars, [jnp.array(rest_pose) for _ in range(timesteps)]),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        verbose=False,
    )
    print("Elapsed time (jit + exec): ", time.time() - start)

    start = time.time()
    graph = jaxls.FactorGraph.make(factors, pose_vars, verbose=False)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(pose_vars, [jnp.array(rest_pose) for _ in range(timesteps)]),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        verbose=False,
    )
    print("Elapsed time (exec): ", time.time() - start)
    breakpoint()

    opt_traj = onp.array([solution[pose_var] for pose_var in pose_vars])

    factors: List[jaxls.Factor] = []
    for tstep, pose_var in enumerate(pose_vars):
        target_pose_r = jaxlie.SE3(orig_Ts_world_joint[tstep, 7] + jnp.array([0, 0, 0, 0, 0, 0.3, 0]))
        # target_pose_r = jaxlie.SE3(orig_Ts_world_joint[tstep, 7])
        target_pose_l = jaxlie.SE3(orig_Ts_world_joint[tstep, 14])
        factors.extend(
            [
                jaxls.Factor.make(lower_joint_factor, (pose_var,)),  # Lower joint limits.
                jaxls.Factor.make(upper_joint_factor, (pose_var,)),  # Upper joint limits.
                jaxls.Factor.make(rest_pose_factor, (pose_var,)),  # Bias solution to be centered around rest pose.
                jaxls.Factor.make(ik_on_ee_r, (pose_var, target_pose_r),),  # Position constraints.
                jaxls.Factor.make(ik_on_ee_l, (pose_var, target_pose_l),),  # Position constraints.
            ]
        )
        if tstep > 0:
            factors.append(
                jaxls.Factor.make(smoothness_factor, (pose_var, pose_vars[tstep - 1]))
            )

    graph = jaxls.FactorGraph.make(factors, pose_vars, verbose=True)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(pose_vars, [jnp.array(rest_pose) for _ in range(timesteps)]),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        # verbose=False,
    )
    opt_traj_without_coll = onp.array([solution[pose_var] for pose_var in pose_vars])

    server = viser.ViserServer()
    urdf_orig = viser.extras.ViserUrdf(
        server, yourdf, mesh_color_override=(220, 100, 100), root_node_name="/urdf_orig"
    )
    urdf_smoothed = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf_smoothed"
    )
    urdf_no_coll = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf_smoothed_no_coll"
    )

    # Visualize!
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )

    @slider.on_update
    def _(_) -> None:
        urdf_orig.update_cfg(traj[slider.value])
        urdf_smoothed.update_cfg(opt_traj[slider.value])
        urdf_no_coll.update_cfg(opt_traj_without_coll[slider.value])

        Ts_world_joint = onp.array(jax_urdf.forward_kinematics(traj[slider.value]))
        for i in range(Ts_world_joint.shape[0]):
            server.scene.add_frame(
                f"/joints/{i}",
                wxyz=Ts_world_joint[i, :4],
                position=Ts_world_joint[i, 4:7],
                axes_length=0.1,
                axes_radius=0.01,
            )
        spheres = jax_urdf.spheres(opt_traj[slider.value])
        server.scene.add_mesh_trimesh("spheres", spheres.to_trimesh())

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        time.sleep(1.0 / 10.0)
    breakpoint()

main()