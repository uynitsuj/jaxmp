""" 05_path_planning.py
Simple point-to-point motion planning.
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
    coll_weight: float = 10.0,
    limit_weight: float = 100.0,
    smoothness_weight: float = 10.0,
    timesteps: int = 20,
):
    yourdf = load_robot_description("yumi_description")
    kin = JaxCollKinematics.from_urdf(
        yourdf,
        self_coll_ignore=[
            ('gripper_l_finger_l', 'gripper_l_finger_r'),
            ('gripper_r_finger_l', 'gripper_r_finger_r'),
        ]
    )
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    start_pose_r = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi),
        translation=jnp.array([0.4, 0.0, 0.5])
    )
    end_pose_r = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi),
        translation=jnp.array([0.4, 0.0, 0.1])
    )
    start_pose_l = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi),
        translation=jnp.array([0.3, 0.0, 0.3])
    )
    end_pose_l = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi),
        translation=jnp.array([0.5, -0.0, 0.3])
    )
    joint_r_idx, joint_l_idx = 7, 14

    class JointVar(jaxls.Var[jax.Array], default=rest_pose): ...

    def limit_cost(vals, var):
        joint_cfg: jax.Array = vals[var]
        return (
            jnp.maximum(0.0, joint_cfg - kin.limits_upper) +
            jnp.maximum(0.0, kin.limits_lower - joint_cfg)
        ) * limit_weight

    def ik_to_joint(vals: jaxls.VarValues, var: JointVar, target_pose: jaxlie.SE3, target_joint_idx: int):
        joint_cfg: jax.Array = vals[var]
        pose_res = (
            jaxlie.SE3(kin.forward_kinematics(joint_cfg)[target_joint_idx]).inverse()
            @ target_pose
        ).log() * jnp.array([pos_weight] * 3 + [rot_weight] * 3)
        return pose_res

    def smoothness_cost(vals, var_curr, var_prev):
        return (vals[var_curr] - vals[var_prev]) * smoothness_weight / (timesteps - 1)

    def coll_self(vals, var):
        return sdf_to_colldist(kin.d_self(cfg=vals[var])) * coll_weight / timesteps

    def coll_self_along_segment(vals, var_curr, var_prev, n_samples=5):
        prev_cfg, curr_cfg = vals[var_prev], vals[var_curr]
        return jnp.concatenate([
            sdf_to_colldist(kin.d_self(cfg=(prev_cfg*t + curr_cfg*(1-t))))
            for t in jnp.linspace(0, 1, n_samples)
        ]) * (coll_weight / n_samples / timesteps)

    # 1. First, calculate the start and end joint configurations.
    pose_vars = [JointVar(id=i) for i in range(2)]
    factors = []
    for pose_var in pose_vars:
        factors.extend([
            jaxls.Factor.make(coll_self, (pose_var,)),
            jaxls.Factor.make(limit_cost, (pose_var,)),
        ])
    factors.extend([
        jaxls.Factor.make(ik_to_joint, (pose_vars[0], start_pose_l, joint_l_idx)),
        jaxls.Factor.make(ik_to_joint, (pose_vars[1], end_pose_l, joint_l_idx)),
        jaxls.Factor.make(ik_to_joint, (pose_vars[0], start_pose_r, joint_r_idx)),
        jaxls.Factor.make(ik_to_joint, (pose_vars[1], end_pose_r, joint_r_idx)),
    ])
    graph = jaxls.FactorGraph.make(factors, pose_vars)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(pose_vars, [rest_pose] * 2),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    endpoint_cfgs = [solution[var] for var in pose_vars]

    # 2. Create a joint-space linear trajectory between the start and end joint configurations.
    traj_init_values = [
        endpoint_cfgs[0] + (endpoint_cfgs[1] - endpoint_cfgs[0]) * i / (timesteps - 1)
        for i in range(timesteps)
    ]

    # 3. Calculate the full trajectory!
    traj_vars = [JointVar(id=i) for i in range(timesteps)]
    factors = []
    for tstep, traj_var in enumerate(traj_vars):
        factors.extend([
            jaxls.Factor.make(limit_cost, (traj_var,)),
        ])
        if tstep > 0:
            factors.extend([
                jaxls.Factor.make(smoothness_cost, (traj_var, traj_vars[tstep - 1])),
                jaxls.Factor.make(coll_self_along_segment, (traj_var, traj_vars[tstep - 1])),
            ])
    factors.extend([
        jaxls.Factor.make(ik_to_joint, (traj_vars[0], start_pose_l, joint_l_idx)),
        jaxls.Factor.make(ik_to_joint, (traj_vars[-1], end_pose_l, joint_l_idx)),
        jaxls.Factor.make(ik_to_joint, (traj_vars[0], start_pose_r, joint_r_idx)),
        jaxls.Factor.make(ik_to_joint, (traj_vars[-1], end_pose_r, joint_r_idx)),
    ])

    graph = jaxls.FactorGraph.make(factors, traj_vars,)
    solution = graph.solve(
        initial_vals=jaxls.VarValues.make(traj_vars, traj_init_values),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
    )
    traj = onp.array([solution[var] for var in traj_vars])

    import viser
    import viser.extras
    import time

    server = viser.ViserServer()
    urdf_orig = viser.extras.ViserUrdf(server, yourdf, root_node_name="/urdf")

    server.scene.add_frame(
        "start_pose_l",
        position=onp.array(start_pose_l.translation()),
        wxyz=onp.array(start_pose_l.rotation().wxyz),
        axes_length=0.1,
        axes_radius=0.01
    )
    server.scene.add_frame(
        "start_pose_r",
        position=onp.array(start_pose_r.translation()),
        wxyz=onp.array(start_pose_r.rotation().wxyz),
        axes_length=0.1,
        axes_radius=0.01
    )
    server.scene.add_frame(
        "end_pose_l",
        position=onp.array(end_pose_l.translation()),
        wxyz=onp.array(end_pose_l.rotation().wxyz),
        axes_length=0.1,
        axes_radius=0.01
    )
    server.scene.add_frame(
        "end_pose_r",
        position=onp.array(end_pose_r.translation()),
        wxyz=onp.array(end_pose_r.rotation().wxyz),
        axes_length=0.1,
        axes_radius=0.01
    )

    visualize_spheres = server.gui.add_checkbox("Show spheres", initial_value=False)
    sphere_handle = None

    # Visualize!
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    @slider.on_update
    def _(_) -> None:
        nonlocal sphere_handle
        urdf_orig.update_cfg(traj[slider.value])

        if visualize_spheres.value:
            sphere_handle = server.scene.add_mesh_trimesh(
                "spheres",
                kin.spheres(traj[slider.value]).to_trimesh()
            )
        elif sphere_handle is not None:
            sphere_handle.remove()

    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps
        time.sleep(1.0 / 10.0)


if __name__ == '__main__':
    main()