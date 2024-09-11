"""
constrain_grasp_axis.py
This is more of a motion planning setting.
"""

import time
import tyro
from pathlib import Path

from yourdfpy import URDF

import jax.numpy as jnp
import jaxlie
import numpy as onp
import jaxls

import viser
import viser.extras

from jaxmp.kinematics import JaxKinTree, sort_joint_map, freeze_joints
from jaxmp.collision_types import RobotColl, CapsuleColl, HalfSpaceColl
from jaxmp.robot_factors import RobotFactors

def main(
    timesteps: int = 40,
    dt: float = 0.1,
    pos_weight: float = 10.0,
    rot_weight: float = 1.0,
    rest_weight: float = 0.01,
    limit_weight: float = 100.0,
    limit_vel_weight: float = 10.0,
    self_coll_weight: float = 0.1,
    world_coll_weight: float = 20.0,
    smoothness_weight: float = 1.0,
):
    ur5_urdf_path = Path(__file__).parent / "../assets/ur5_robotiq/urdf/ur5_robotiq_85.urdf"
    def filename_handler(fname):
        return ur5_urdf_path.parent / fname
    urdf = URDF.load(ur5_urdf_path, filename_handler=filename_handler)
    urdf = sort_joint_map(urdf)
    urdf = freeze_joints(urdf, list(range(7, len(urdf.robot.joints))))

    kin = JaxKinTree.from_urdf(urdf)
    robot_coll = RobotColl.from_urdf(urdf)
    robot_factors = RobotFactors(kin, coll=robot_coll)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    start_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_x_radians(jnp.pi) @ jaxlie.SO3.from_z_radians(jnp.pi / 2),
        jnp.array([0.5, 0.3, 0.2])
    )
    end_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_x_radians(jnp.pi) @ jaxlie.SO3.from_z_radians(jnp.pi / 2),
        jnp.array([0.5, -0.3, 0.2])
    )

    grasp_joint = "grasp_joint"
    target_joint_idx = kin.joint_names.index(grasp_joint)
    rot_weight_list = [rot_weight] * 3

    server = viser.ViserServer()

    # Visualize robot, target joint pose, and desired joint pose.
    urdf_vis = viser.extras.ViserUrdf(server, urdf)
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Add some obstacles / collision logic.
    ground_coll = HalfSpaceColl(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0]))
    server.scene.add_grid("ground", width=3, height=3, cell_size=0.1, position=(0.0, 0.0, 0.001))
    visualize_spheres = server.gui.add_checkbox("Show Collbody", initial_value=False)

    wall_intervals = jnp.arange(start=0.3, stop=0.7, step=0.05)
    translation = jnp.concatenate( [wall_intervals.reshape(-1, 1), jnp.zeros((wall_intervals.shape[0], 2))], axis=1,)
    # translation = translation.at[:, 1].set(-0.1)
    wall_coll = CapsuleColl(
        radii=jnp.full((translation.shape[0],), 0.05),
        heights=jnp.full((translation.shape[0],), 0.7),
        tf=jaxlie.SE3.from_translation(translation),
    )
    server.scene.add_mesh_trimesh("wall", wall_coll.to_trimesh() + ground_coll.to_trimesh())

    server.scene.add_frame(
        "start_pose",
        position=start_pose.translation(),
        wxyz=start_pose.rotation().wxyz,
        axes_length=0.05,
        axes_radius=0.01,
    )
    server.scene.add_frame(
        "end_pose",
        position=end_pose.translation(),
        wxyz=end_pose.rotation().wxyz,
        axes_length=0.05,
        axes_radius=0.01,
    )

    # Create factor graph.
    JointVar = robot_factors.get_var_class(default_val=rest_pose)

    world_coll_weight_arr = jnp.array([world_coll_weight] * len(robot_coll))
    world_coll_weight_arr = world_coll_weight_arr.at[
        robot_coll.link_names.index("robotiq_arg2f_base_link") :
    ].set(
        0.00
    )  # ignore robotiq from collision.
    self_coll_weight_arr = jnp.array([self_coll_weight] * len(robot_coll))
    self_coll_weight_arr = self_coll_weight_arr.at[
        robot_coll.link_names.index("robotiq_arg2f_base_link") :
    ].set(
        0.00
    )  # ignore robotiq from collision.

    # 1. First solve the start + end poses.
    factors = []
    traj_vars = [JointVar(id=idx) for idx in range(2)]
    for t, traj_var in enumerate(traj_vars):
        factors.extend([
            jaxls.Factor.make(
                robot_factors.rest_cost,
                (
                    traj_var,
                    jnp.array([rest_weight] * kin.num_actuated_joints) / timesteps,
                ),
            ),
            jaxls.Factor.make(
                robot_factors.limit_cost,
                (
                    traj_var,
                    jnp.array([limit_weight] * kin.num_actuated_joints) / timesteps,
                ),
            ),
            jaxls.Factor.make(
                robot_factors.world_coll_cost,
                (
                    traj_var,
                    ground_coll,
                    0.05,
                    world_coll_weight_arr / timesteps,
                ),
            ),
            jaxls.Factor.make(
                robot_factors.world_coll_cost,
                (
                    traj_var,
                    wall_coll,
                    0.1,
                    world_coll_weight_arr / timesteps,
                ),
            ),
            jaxls.Factor.make(
                robot_factors.self_coll_cost,
                (
                    traj_var,
                    0.05,
                    self_coll_weight_arr / timesteps,
                ),
            )
        ])
    factors.extend([
        jaxls.Factor.make(
            robot_factors.ik_cost,
            (
                traj_vars[0],
                start_pose,
                target_joint_idx,
                jnp.array([pos_weight]*3 + rot_weight_list),
            )
        ),
        jaxls.Factor.make(
            robot_factors.ik_cost,
            (
                traj_vars[-1],
                end_pose,
                target_joint_idx,
                jnp.array([pos_weight]*3 + rot_weight_list),
            )
        ),
    ])
    graph = jaxls.FactorGraph.make(
        factors,
        traj_vars,
    )

    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(traj_vars, [JointVar.default]*2),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    start_joints, end_joints = [solution[var] for var in traj_vars]

    # Now, solve for the full trajectory.
    traj_vars = [JointVar(id=idx) for idx in range(timesteps)]
    factors = []
    for t, traj_var in enumerate(traj_vars):
        factors.extend([
            jaxls.Factor.make(
                robot_factors.rest_cost,
                (
                    traj_var,
                    jnp.array([rest_weight] * kin.num_actuated_joints) / timesteps,
                ),
            ),
            jaxls.Factor.make(
                robot_factors.limit_cost,
                (
                    traj_var,
                    jnp.array([limit_weight] * kin.num_actuated_joints) / timesteps,
                ),
            ),
            jaxls.Factor.make(
                robot_factors.world_coll_cost,
                (
                    traj_var,
                    ground_coll,
                    0.05,
                    world_coll_weight_arr / timesteps,
                ),
            ),
            jaxls.Factor.make(
                robot_factors.world_coll_cost,
                (
                    traj_var,
                    wall_coll,
                    0.05,
                    world_coll_weight_arr / timesteps,
                ),
            ),
            jaxls.Factor.make(
                robot_factors.self_coll_cost,
                (
                    traj_var,
                    0.05,
                    self_coll_weight_arr / timesteps,
                ),
            )
        ])
        if t > 0:
            factors.extend([
                jaxls.Factor.make(
                    robot_factors.smoothness_cost,
                    (
                        traj_vars[t - 1],
                        traj_var,
                        jnp.array([smoothness_weight] * kin.num_actuated_joints) / (timesteps - 1),
                    ),
                ),
                jaxls.Factor.make(
                    robot_factors.joint_limit_vel_cost,
                    (
                        traj_vars[t],
                        traj_vars[t - 1],
                        dt,
                        jnp.array([limit_vel_weight] * kin.num_actuated_joints) / (timesteps - 1),
                    ),
                ),
            ])
    factors.extend([
        jaxls.Factor.make(
            robot_factors.ik_cost,
            (
                traj_vars[0],
                start_pose,
                target_joint_idx,
                jnp.array([pos_weight]*3 + rot_weight_list),
            )
        ),
        jaxls.Factor.make(
            robot_factors.ik_cost,
            (
                traj_vars[-1],
                end_pose,
                target_joint_idx,
                jnp.array([pos_weight]*3 + rot_weight_list),
            )
        ),
    ])
    graph = jaxls.FactorGraph.make(
        factors,
        traj_vars,
    )

    initial_joints = [
        start_joints + (end_joints - start_joints) * t / (timesteps - 1)
        for t in range(timesteps)
    ]
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(traj_vars, initial_joints),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    traj = onp.array([solution[var] for var in traj_vars])

    # Visualize!
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )

    sphere_handle = None
    @slider.on_update
    def _(_) -> None:
        nonlocal sphere_handle
        urdf_vis.update_cfg(traj[slider.value])
        if visualize_spheres.value:
            sphere_handle = server.scene.add_mesh_trimesh(
                "sph",
                robot_coll.transform(
                    jaxlie.SE3(kin.forward_kinematics(traj[slider.value]))
                ).to_trimesh(),
            )
        elif sphere_handle is not None:
            sphere_handle.remove()

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
