""" 07_trajopt_manipulability.py
Similar to 04_trajopt.py, but including manipulability as the cost function!
"""

from pathlib import Path
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
    smoothness_weight: float = 1.0,
    coll_weight: float = 10.0,
    manipulability_weight: float = 0.1,
):
    yourdf = load_robot_description("yumi_description")
    kin = JaxCollKinematics.from_urdf(yourdf, self_coll_ignore=[
        ('gripper_l_finger_l', 'gripper_l_finger_r'),
        ('gripper_r_finger_l', 'gripper_r_finger_r'),
    ])
    trajectory = onp.load(
        Path(__file__).parent / "assets/yumi_trajectory.npy",
        allow_pickle=True
    ).item()  # {'joint_name': [time, wxyz_xyz]}
    timesteps = list(trajectory.values())[0].shape[0]

    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    class JointVar(jaxls.Var[jax.Array], default=rest_pose): ...

    def ik_to_joint_with_offset(
        vals: jaxls.VarValues,
        var: JointVar,
        target_pose: jaxlie.SE3,
        offset_var: jaxls.SE3Var,
        target_joint_idx: int
    ):
        joint_cfg: jax.Array = vals[var]
        offset_pose: jaxlie.SE3 = vals[offset_var]
        pose_res = (
            jaxlie.SE3(kin.forward_kinematics(joint_cfg)[target_joint_idx]).inverse()
            @ (offset_pose @ target_pose)
        ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3)
        return pose_res

    def ik_to_joint(
        vals: jaxls.VarValues,
        var: JointVar,
        target_pose: jaxlie.SE3,
        target_joint_idx: int
    ):
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

    def smoothness_cost(vals, var_curr, var_prev):
        return (vals[var_curr] - vals[var_prev]) * smoothness_weight

    def self_coll_cost(vals, var):
        return sdf_to_colldist(kin.d_self(cfg=vals[var])) * coll_weight / timesteps

    # New cost: Manipulability.
    def manipulability_cost(vals, var, target_joint_idx: int):
        joint_cfg: jax.Array = vals[var]

        # Jacobian between wxyz_xyz, and dof.
        jacobian = jax.jacfwd(kin.forward_kinematics)(joint_cfg)[target_joint_idx]
        norm = jnp.linalg.norm(jacobian, ord='nuc')

        return jnp.maximum(7.0 - norm[None], 0.0) * manipulability_weight

    traj_vars = [JointVar(id=i) for i in range(timesteps)] + [jaxls.SE3Var(id=timesteps)]

    factors = []
    list_target_joint_idx = [kin.joint_names.index(joint_name) for joint_name in trajectory.keys()]
    for tstep in range(timesteps):
        for idx, joint_name in zip(list_target_joint_idx, trajectory.keys()):
            factors.extend([
                jaxls.Factor.make(
                    ik_to_joint_with_offset, (traj_vars[tstep], jaxlie.SE3(trajectory[joint_name][tstep]), traj_vars[-1], idx),
                ),
                jaxls.Factor.make(manipulability_cost, (traj_vars[tstep], idx)),
            ])
        factors.extend([
            jaxls.Factor.make(limit_cost, (traj_vars[tstep],),),
            jaxls.Factor.make(self_coll_cost, (traj_vars[tstep],)),
        ])
        if tstep > 0:
            factors.append(jaxls.Factor.make( smoothness_cost, (traj_vars[tstep], traj_vars[tstep - 1]),))
    factors.append(
        jaxls.Factor.make(
            lambda vals, var: (
                vals[var] @ jaxlie.SE3(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3),
            (traj_vars[-1],)
        )
    )

    # Run once, with mainpulation cost.
    graph = jaxls.FactorGraph.make(
        factors,
        traj_vars,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(
            traj_vars,
            [rest_pose] * timesteps + [jaxlie.SE3(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))]
        ),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    traj = onp.array([solution[var] for var in traj_vars[:-1]])

    # Run again, but without manipulability cost.
    traj_vars = [JointVar(id=i) for i in range(timesteps)]
    factors = []
    list_target_joint_idx = [kin.joint_names.index(joint_name) for joint_name in trajectory.keys()]
    for tstep in range(timesteps):
        for idx, joint_name in zip(list_target_joint_idx, trajectory.keys()):
            factors.extend([
                jaxls.Factor.make(
                    ik_to_joint, (traj_vars[tstep], jaxlie.SE3(trajectory[joint_name][tstep]), idx),
                ),
            ])
        factors.extend([
            jaxls.Factor.make(
                limit_cost, (traj_vars[tstep],),
            )
        ])
        if tstep > 0:
            factors.append(jaxls.Factor.make( smoothness_cost, (traj_vars[tstep], traj_vars[tstep - 1]),))
    graph = jaxls.FactorGraph.make(
        factors,
        traj_vars,
    )
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(
            traj_vars,
            [rest_pose] * timesteps
        ),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    traj_no_manip = onp.array([solution[var] for var in traj_vars])

    # Visualization.
    import viser
    import viser.extras
    import time

    server = viser.ViserServer()
    urdf_orig = viser.extras.ViserUrdf(server, yourdf, root_node_name="/urdf")
    urdf_no_manip = viser.extras.ViserUrdf(server, yourdf, root_node_name="/urdf_no_manip", mesh_color_override=(220, 100, 100))

    # Visualize!
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    for joint_name, joint_pose_traj in trajectory.items():
        server.scene.add_batched_axes(
            joint_name,
            batched_positions=joint_pose_traj[:, 4:],
            batched_wxyzs=joint_pose_traj[:, :4],
            axes_length=0.02,
            axes_radius=0.002
        )

    @slider.on_update
    def _(_) -> None:
        urdf_orig.update_cfg(traj[slider.value])
        urdf_no_manip.update_cfg(traj_no_manip[slider.value])

        Ts_world_joint = onp.array(kin.forward_kinematics(traj[slider.value]))
        for idx, joint_name in zip(list_target_joint_idx, trajectory.keys()):
            server.scene.add_frame(
                f"/joints/{joint_name}",
                wxyz=Ts_world_joint[idx, :4],
                position=Ts_world_joint[idx, 4:7],
                axes_length=0.1,
                axes_radius=0.01,
            )

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    main()