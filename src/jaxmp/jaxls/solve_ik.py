from typing import Literal, Optional
import jax
import jaxlie
import jax_dataclasses as jdc
import jaxls
from jaxmp.jaxls.robot_factors import RobotFactors
from jaxmp.kinematics import JaxKinTree

import jax.numpy as jnp

@jdc.jit
def solve_ik(
    kin: JaxKinTree,
    target_pose: jaxlie.SE3,
    target_joint_indices: jax.Array,
    pos_weight: float,
    rot_weight: float,
    rest_weight: float,
    limit_weight: float,
    manipulability_weight: float,
    joint_vel_weight: float,
    rest_pose: jnp.ndarray,
    solver_type: jdc.Static[
        Literal["cholmod", "conjugate_gradient", "dense_cholesky"]
    ] = "conjugate_gradient",
    freeze_target_xyz_xyz: Optional[jnp.ndarray] = None,
    freeze_base_xyz_xyz: Optional[jnp.ndarray] = None,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    """
    Solve IK for the robot.
    Args:
        target_pose: Desired pose of the target joint, SE3 has batch axes (n_target,).
        target_joint_indices: Indices of the target joints, length n_target.
        freeze_target_xyz_xyz: 6D vector indicating which axes to freeze in the target frame.
        freeze_base_xyz_xyz: 6D vector indicating which axes to freeze in the base frame.
    Returns:
        Base pose (jaxlie.SE3)
        Joint angles (jnp.ndarray)
    """
    if freeze_target_xyz_xyz is None:
        freeze_target_xyz_xyz = jnp.ones(6)
    if freeze_base_xyz_xyz is None:
        freeze_base_xyz_xyz = jnp.ones(6)

    JointVar = RobotFactors.get_var_class(kin, default_val=rest_pose)

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * (1 - freeze_base_xyz_xyz)
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity(),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    joint_vars = [JointVar(0), ConstrainedSE3Var(0)]

    factors: list[jaxls.Factor] = [
        jaxls.Factor(
            RobotFactors.limit_cost,
            (
                kin,
                JointVar(0),
                jnp.array([limit_weight] * kin.num_actuated_joints),
            ),
        ),
        # jaxls.Factor(
        #     RobotFactors.joint_limit_vel_cost,
        #     (
        #         kin,
        #         JointVar(0),
        #         rest_pose,
        #         0.1,
        #         jnp.array([joint_vel_weight] * kin.num_actuated_joints),
        #     ),
        # ),
        jaxls.Factor(
            RobotFactors.rest_cost,
            (
                JointVar(0),
                jnp.array([rest_weight] * kin.num_actuated_joints),
            ),
        ),
    ]

    ik_weights = jnp.array([pos_weight] * 3 + [rot_weight] * 3)
    ik_weights = ik_weights * freeze_target_xyz_xyz
    for idx, target_joint_idx in enumerate(target_joint_indices):
        factors.extend([
            jaxls.Factor(
                RobotFactors.ik_cost,
                (
                    kin,
                    joint_vars[0],
                    jaxlie.SE3(target_pose.wxyz_xyz[idx]),
                    target_joint_idx,
                    ik_weights,
                    ConstrainedSE3Var(0),
                ),
            ),
            jaxls.Factor.make(
                RobotFactors.manipulability_cost,
                (
                    kin,
                    joint_vars[0],
                    target_joint_idx,
                    jnp.array([manipulability_weight] * kin.num_actuated_joints),
                ),
            )
        ])

    graph = jaxls.FactorGraph.make(
        factors,
        joint_vars,
        use_onp=False,
    )
    solution = graph.solve(
        linear_solver=solver_type,
        initial_vals=jaxls.VarValues.make(joint_vars),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        termination=jaxls.TerminationConfig(gradient_tolerance=1e-5, parameter_tolerance=1e-5),
        verbose=False,
    )

    # Update visualization.
    base_pose = solution[ConstrainedSE3Var(0)]
    joints = solution[JointVar(0)]
    return base_pose, joints
