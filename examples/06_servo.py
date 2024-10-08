"""
06_servo.py

Tests robot servoing using jaxmp; similar to 01_kinematics.py,
but instead of reaching the position directly, the robot solves for the velocity instead.

It should respect joint velocity limits and singularity avoidance.
"""

import time
import tyro

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
from jaxmp.jaxls.robot_factors import RobotFactors


def pos_limit_cost(vals: jaxls.VarValues, var: jaxls.Var[jnp.ndarray], curr_joints: jnp.ndarray, kin: JaxKinTree, limit_weight: float, dt: float) -> jnp.ndarray:
    joint_vel = vals[var]
    joint_pos = curr_joints + joint_vel * dt
    residual = (
        jnp.maximum(0.0, joint_pos - kin.limits_upper) +
        jnp.maximum(0.0, kin.limits_lower - joint_pos)
    )
    return (residual * jnp.array([limit_weight] * kin.num_actuated_joints)).flatten()

def vel_limit_cost(vals: jaxls.VarValues, var: jaxls.Var[jnp.ndarray], kin: JaxKinTree, vel_limit_weight: float) -> jnp.ndarray:
    joint_vel = vals[var]
    residual = jnp.maximum(jnp.abs(joint_vel) - kin.joint_vel_limit, 0.0)
    return (residual * jnp.array([vel_limit_weight] * kin.num_actuated_joints)).flatten()

def rest_cost(vals: jaxls.VarValues, var: jaxls.Var[jnp.ndarray], curr_joints: jnp.ndarray, kin: JaxKinTree, rest_weight: float, dt: float) -> jnp.ndarray:
    joint_vel = vals[var]
    joint_pos = curr_joints + joint_vel * dt
    return (joint_pos * jnp.array([rest_weight] * kin.num_actuated_joints)).flatten()

def ik_cost(vals: jaxls.VarValues, var: jaxls.Var[jnp.ndarray], pose: jaxlie.SE3, target_idx: int, curr_joints: jnp.ndarray, kin: JaxKinTree, gain: float, dt: float, pos_weight: float, rot_weight: float) -> jnp.ndarray:
    """
    PBS is formulated as:
    v = beta * (T_target^-1 * T_current).log()
    """
    joint_vel = vals[var]
    Ts_joint_world = kin.forward_kinematics(curr_joints + joint_vel * dt)
    residual = (
        (jaxlie.SE3(Ts_joint_world[target_idx])).inverse()
        @ (pose)
    ).log()
    return (residual * gain * jnp.array([pos_weight] * 3 + [rot_weight] * 3)).flatten()

def manip_cost(vals: jaxls.VarValues, var: jaxls.Var[jnp.ndarray], curr_joints: jnp.ndarray, kin: JaxKinTree, target_idx: int, dt: float, manip_weight: float) -> jnp.ndarray:
    joint_vel = vals[var]
    curr_joints = curr_joints + joint_vel * dt
    J = jax.jacfwd(lambda joints: kin.forward_kinematics(joints)[target_idx, 4:])(curr_joints)
    eigval, _ = jnp.linalg.eigh(J @ J.T)
    # want to maximize these!
    return manip_weight * jnp.minimum(1.0-eigval, 1.0).flatten()

@jdc.jit
def solve_ik(
    kin: JaxKinTree,
    curr_joints: jnp.ndarray,
    target_pose: jaxlie.SE3,
    target_joint_idx: jdc.Static[int],
    pos_weight: float,
    rot_weight: float,
    rest_weight: float,
    limit_weight: float,
    vel_limit_weight: float,
    manip_weight: float,
    gain: float,
    dt: float,
) -> jnp.ndarray:
    """
    Solve for the _joint velocity_ that moves the robot from the current pose to the target pose.
    """
    JointVelVar = RobotFactors.get_var_class(kin, default_val=jnp.zeros_like(curr_joints))

    # What's the best way to fix up robotfactor?
    joint_vars = [JointVelVar(0)]

    factors: list[jaxls.Factor] = [
        jaxls.Factor.make(pos_limit_cost, (JointVelVar(0), curr_joints, kin, limit_weight, dt)),
        jaxls.Factor.make(vel_limit_cost, (JointVelVar(0), kin, vel_limit_weight)),
        jaxls.Factor.make(rest_cost, (JointVelVar(0), curr_joints, kin, rest_weight, dt)),
    ]

    factors.extend([
        jaxls.Factor.make(
            ik_cost,
            (
                JointVelVar(0),
                jaxlie.SE3(target_pose.wxyz_xyz),
                target_joint_idx,
                curr_joints,
                kin,
                gain,
                dt,
                pos_weight,
                rot_weight
            ),
        ),
        jaxls.Factor.make(
            manip_cost,
            (
                JointVelVar(0),
                curr_joints,
                kin,
                target_joint_idx,
                dt,
                manip_weight,
            ),
        )
    ])

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(joint_vars),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        # verbose=False,
    )

    # Update visualization.
    joint_vel = solution[JointVelVar(0)]
    return joint_vel

def main(
    robot_description: str = "yumi_description",
    pos_weight: float = 5.0,
    rot_weight: float = 2.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    velocity_limit_weight: float = 100.0,
    manip_weight: float = 0.01,
    gain: float = 1,
    dt: float = 0.2,
):
    urdf = load_robot_description(robot_description)
    urdf = sort_joint_map(urdf)

    kin = JaxKinTree.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    target_tf_handle = server.scene.add_transform_controls("target transform", scale=0.2)
    target_frame_handle = server.scene.add_frame("target", axes_length=0.1)
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Timing info.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    # Show joint names + limits.
    slider_handles = {}
    with server.gui.add_folder("Joint position control"):
        for joint_name in urdf.joint_names:
            lower = kin.limits_lower[kin.joint_names.index(joint_name)].item()
            upper = kin.limits_upper[kin.joint_names.index(joint_name)].item()
            slider = server.gui.add_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=1e-3,
                initial_value=(lower + upper) / 2.0,
                disabled=True,
            )
            slider_handles[joint_name] = slider

    # Show target joint name, and current joint positions.
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0]
    )
    current_joints = rest_pose
    urdf_vis.update_cfg(onp.array(current_joints))

    while True:
        target_pose = jaxlie.SE3(
            jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position])
        )
        target_joint_idx = kin.joint_names.index(target_name_handle.value)

        # Generate a trajectory from the current joint angles to the target pose.
        start_time = time.time()
        joint_vel = solve_ik(
            kin,
            current_joints,
            target_pose,
            target_joint_idx,
            pos_weight,
            rot_weight,
            rest_weight,
            limit_weight,
            velocity_limit_weight,
            manip_weight,
            gain,
            dt,
        )
        end_time = time.time()
        timing_handle.value = (end_time - start_time) * 1000
        
        current_joints = current_joints + joint_vel * dt
        urdf_vis.update_cfg(onp.array(current_joints))
        for joint_name, slider in slider_handles.items():
            slider.value = onp.array(current_joints[kin.joint_names.index(joint_name)]).item()

        target_frame_handle.position = tuple(target_pose.translation())
        target_frame_handle.wxyz = tuple(target_pose.rotation().wxyz)

if __name__ == "__main__":
    tyro.cli(main)
