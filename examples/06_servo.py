"""
06_servo.py

Tests robot servoing using jaxmp; similar to 01_kinematics.py,
but instead of reaching the position directly,
the robot slowly moves to the target position.

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
from jaxmp.robot_factors import RobotFactors
from jaxmp.collision_types import RobotColl

def main(
    robot_description: str = "yumi_description",
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    smoothness_weight: float = 1.0,
    velocity_limit_weight: float = 10.0,
    start_pose_weight: float = 10.0,
    self_collision_weight: float = 1.0,
    dt: float = 0.1,
):
    urdf = load_robot_description(robot_description)
    urdf = sort_joint_map(urdf)

    kin = JaxKinTree.from_urdf(urdf)
    coll = RobotColl.from_urdf(urdf)

    robot_factors = RobotFactors(kin, coll)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    target_tf_handle = server.scene.add_transform_controls("target transform", scale=0.2)
    target_frame_handle = server.scene.add_frame("target", axes_length=0.1)
    current_frame_handle = server.scene.add_frame("current", axes_length=0.1)
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    update_pose_queue_handle = server.gui.add_button("Update pose queue")

    # Timing info.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    # Show target joint name, and current joint positions.
    target_name_handle = server.gui.add_dropdown(
        "target joint",
        list(urdf.joint_names),
        initial_value=urdf.joint_names[0]
    )

    # Create factor graph.
    JointVar = robot_factors.get_var_class(default_val=rest_pose)

    @jdc.jit
    def get_joints_for_pose(
        target_pose: jaxlie.SE3,
        target_joint_idx: jdc.Static[int],
    ) -> jax.Array:
        joint_vars = [JointVar(id=0)]
        factors = [
            jaxls.Factor.make(
                robot_factors.ik_cost,
                (
                    joint_vars[0],  # Apply target pose to the last waypoint
                    target_pose,
                    target_joint_idx,
                    jnp.array([pos_weight] * 3 + [rot_weight] * 3),
                ),
            ),
            jaxls.Factor.make(
                robot_factors.limit_cost,
                (
                    joint_vars[0],
                    jnp.array(
                        [limit_weight] * kin.num_actuated_joints
                    ),
                ),
            ),
            jaxls.Factor.make(
                robot_factors.rest_cost,
                (
                    joint_vars[0],
                    jnp.array(
                        [rest_weight] * kin.num_actuated_joints
                    ),
                ),
            ),
            jaxls.Factor.make(
                robot_factors.self_coll_cost,
                (
                    joint_vars[0],
                    0.05,
                    jnp.array(
                        [self_collision_weight] * len(coll)
                    ),
                ),
            ),
        ]

        # Create the factor graph
        graph = jaxls.FactorGraph.make(
            factors,
            joint_vars,
            verbose=False,
            use_onp=False,
        )

        # Solve the factor graph
        solution = graph.solve(
            initial_vals=jaxls.VarValues.make(joint_vars, [JointVar.default]),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        )

        return solution[joint_vars[0]]

    @jdc.jit
    def path_to_pose(
        start_joint_angles: jax.Array,
        target_pose: jaxlie.SE3,
        target_joint_idx: jdc.Static[int],
        num_waypoints: jdc.Static[int],
    ) -> list[jax.Array]:
        """
        Generates a trajectory of poses from a start pose to a target pose.
        Args:
            start_joint_angles (jnp.ndarray): The starting joint angles.
            target_pose (jaxlie.SE3): The target pose.
            target_joint_idx (int): The index of the target joint.
            num_waypoints (int): The number of waypoints in the trajectory.
        Returns:
            list[jaxlie.SE3]: The trajectory of joint angles from start_pose to target_pose.
        """
        # Calculate joint variable for the final waypoint
        final_joints = get_joints_for_pose(target_pose, target_joint_idx)

        # Create joint variables for each waypoint
        joint_vars = [JointVar(id=i) for i in range(num_waypoints)]

        # Create factors for each waypoint
        factors = []

        # Add factors for the initial and final waypoints
        factors.extend(
            [
                jaxls.Factor.make(
                    lambda vals, val: start_pose_weight
                    * jnp.abs(vals[val] - start_joint_angles),
                    (joint_vars[0],),
                ),
                jaxls.Factor.make(
                    robot_factors.ik_cost,
                    (
                        joint_vars[-1],  # Apply target pose to the last waypoint
                        target_pose,
                        target_joint_idx,
                        jnp.array([pos_weight] * 3 + [rot_weight] * 3),
                    ),
                ),
            ]
        )

        for i in range(num_waypoints):
            factors.extend([
                jaxls.Factor.make(
                    robot_factors.limit_cost,
                    (
                        joint_vars[i],
                        jnp.array(
                            [limit_weight / num_waypoints] * kin.num_actuated_joints
                        ),
                    ),
                ),
                jaxls.Factor.make(
                    robot_factors.rest_cost,
                    (
                        joint_vars[i],
                        jnp.array(
                            [rest_weight / num_waypoints] * kin.num_actuated_joints
                        ),
                    ),
                ),
                jaxls.Factor.make(
                    robot_factors.self_coll_cost,
                    (
                        joint_vars[i],
                        0.05,
                        jnp.array(
                            [self_collision_weight / num_waypoints] * len(coll)
                        ),
                    ),
                ),
            ])

        # Add smoothness and joint velocity limit factors between consecutive waypoints
        for i in range(num_waypoints - 1):
            factors.append(
                jaxls.Factor.make(
                    robot_factors.smoothness_cost,
                    (
                        joint_vars[i],
                        joint_vars[i + 1],
                        jnp.array(
                            [smoothness_weight / num_waypoints]
                            * kin.num_actuated_joints
                        ),
                    ),
                )
            )
            factors.append(
                jaxls.Factor.make(
                    robot_factors.joint_limit_vel_cost,
                    (
                        joint_vars[i],
                        joint_vars[i + 1],
                        dt,
                        jnp.array(
                            [velocity_limit_weight / num_waypoints]
                            * kin.num_actuated_joints
                        ),
                    ),
                )
            )

        # Create the factor graph
        graph = jaxls.FactorGraph.make(
            factors,
            joint_vars,
            verbose=False,
            use_onp=False,
        )

        # Initial vals using linear interpolation between start and final joint angles
        initial_vals = [
            start_joint_angles + (final_joints - start_joint_angles) * i / (num_waypoints - 1)
            for i in range(num_waypoints)
        ]

        # Solve the factor graph
        solution = graph.solve(
            initial_vals=jaxls.VarValues.make(joint_vars, initial_vals),
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        )

        traj = [solution[joint_var] for joint_var in joint_vars][1:]
        return traj

    current_joints = rest_pose
    urdf_vis.update_cfg(onp.array(current_joints))

    pose_queue = []

    from threading import Lock
    jax_lock = Lock()

    @target_tf_handle.on_update
    def _(_):
        nonlocal current_joints, pose_queue
        update_pose_queue_handle.disabled = True

        target_pose = jaxlie.SE3(jnp.array([*target_tf_handle.wxyz, *target_tf_handle.position]))
        target_joint_idx = kin.joint_names.index(target_name_handle.value)

        # Generate a trajectory from the current joint angles to the target pose.
        # with server.atomic():
        with jax_lock:
            start_time = time.time()
            pose_queue = path_to_pose(current_joints, target_pose, target_joint_idx, 10)
            timing_handle.value = (time.time() - start_time) * 1000

        target_frame_handle.position = tuple(target_pose.translation())
        target_frame_handle.wxyz = tuple(target_pose.rotation().wxyz)

        update_pose_queue_handle.disabled = False

    while True:
        if pose_queue:
            current_joints = pose_queue.pop(0)

            target_joint_idx = kin.joint_names.index(target_name_handle.value)
            curr_pose = jaxlie.SE3(kin.forward_kinematics(current_joints)[target_joint_idx])
            current_frame_handle.position = tuple(curr_pose.translation())
            current_frame_handle.wxyz = tuple(curr_pose.rotation().wxyz)

            urdf_vis.update_cfg(onp.array(current_joints))

        time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)
