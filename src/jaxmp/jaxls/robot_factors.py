"""Common `jaxls` factors for robot control, via wrapping `JaxKinTree` and `RobotColl`.
"""
from typing import Optional

import jax
from jax import Array
import jax_dataclasses as jdc
import jax.numpy as jnp
import jaxlie

import jaxls

from jaxmp.kinematics import JaxKinTree
from jaxmp.coll import RobotColl, CollGeom, collide, colldist_from_sdf

class RobotFactors:
    """Helper class for using `jaxls` factors with a `JaxKinTree` and `RobotColl`."""

    @staticmethod
    def get_var_class(
        kin: JaxKinTree, default_val: Optional[Array] = None
    ) -> type[jaxls.Var[Array]]:
        """Get the Variable class for this robot. Default value is mid-point of limits."""
        if default_val is None:
            default_val = (kin.limits_upper + kin.limits_lower) / 2

        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default_factory=lambda: default_val,
            tangent_dim=kin.num_actuated_joints,
            retract_fn=kin.get_retract_fn(),
        ): ...

        return JointVar

    @staticmethod
    def ik_cost(
        vals: jaxls.VarValues,
        kin: JaxKinTree,
        var: jaxls.Var[Array],
        target_pose: jaxlie.SE3,
        target_joint_idx: jdc.Static[int],
        weights: Array,
        base_tf_var: Optional[jaxls.Var] = None,
    ) -> Array:
        """Pose cost."""
        joint_cfg: jax.Array = vals[var]
        base_tf = jaxlie.SE3.identity() if base_tf_var is None else vals[base_tf_var]
        Ts_joint_world = kin.forward_kinematics(joint_cfg)
        residual = (
            (base_tf @ jaxlie.SE3(Ts_joint_world[target_joint_idx])).inverse()
            @ (target_pose)
        ).log()
        weights = jnp.broadcast_to(weights, residual.shape)
        assert residual.shape == weights.shape
        return (residual * weights).flatten()

    @staticmethod
    def limit_cost(
        vals: jaxls.VarValues,
        kin: JaxKinTree,
        var: jaxls.Var[Array],
        weights: Array,
    ) -> Array:
        """Limit cost."""
        joint_cfg: jax.Array = vals[var]
        residual = (
            jnp.maximum(0.0, joint_cfg - kin.limits_upper) +
            jnp.maximum(0.0, kin.limits_lower - joint_cfg)
        )
        assert residual.shape == weights.shape
        return residual * weights

    @staticmethod
    def joint_limit_vel_cost(
        vals: jaxls.VarValues,
        kin: JaxKinTree,
        var_curr: jaxls.Var[Array],
        var_prev: jaxls.Var[Array],
        dt: float,
        weights: Array,
    ) -> Array:
        """Joint limit velocity cost."""
        joint_vel = (vals[var_curr] - vals[var_prev]) / dt
        residual = jnp.maximum(0.0, jnp.abs(joint_vel) - kin.joint_vel_limit)
        assert residual.shape == weights.shape
        return residual * weights

    @staticmethod
    def rest_cost(
        vals: jaxls.VarValues,
        var: jaxls.Var[Array],
        weights: Array,
    ) -> Array:
        """Bias towards joints at rest pose, specified by `default`."""
        default = var.default_factory()
        assert default is not None
        assert default.shape == vals[var].shape and default.shape == weights.shape
        return (vals[var] - default) * weights

    @staticmethod
    def self_coll_cost(
        vals: jaxls.VarValues,
        kin: JaxKinTree,
        coll: RobotColl,
        var: jaxls.Var[Array],
        eta: float,
        weights: Array,
    ) -> Array:
        """Collision-scaled dist for self-collision."""
        joint_cfg = vals[var]
        sdf = coll.collide_self(kin, joint_cfg)
        weights = weights * coll.self_coll_matrix
        return (colldist_from_sdf(sdf, eta=eta) * weights).flatten()

    @staticmethod
    def world_coll_cost(
        vals: jaxls.VarValues,
        kin: JaxKinTree,
        coll: RobotColl,
        var: jaxls.Var[Array],
        other: CollGeom,
        eta: float,
        weights: Array,
    ) -> Array:
        """Collision-scaled dist for world collision."""
        joint_cfg = vals[var]
        coll = coll.transform(jaxlie.SE3(kin.forward_kinematics(joint_cfg)[..., coll.link_joint_idx, :]))
        sdf, _, _ = collide(other, coll)
        sdf = sdf[..., 0] * coll.self_coll_matrix
        return (colldist_from_sdf(sdf, eta=eta) * weights).flatten()

    @staticmethod
    def smoothness_cost(
        vals: jaxls.VarValues,
        var_curr: jaxls.Var[Array],
        var_past: jaxls.Var[Array],
        weights: Array,
    ) -> Array:
        """Smoothness cost, for trajectories etc."""
        residual = (vals[var_curr] - vals[var_past])
        assert residual.shape == weights.shape
        return residual * weights

    @staticmethod
    def manipulability_cost(
        vals: jaxls.VarValues,
        kin: JaxKinTree,
        var: jaxls.Var[Array],
        target_joint_idx: int,
        weights: Array,
    ):
        """Manipulability cost."""
        joint_cfg: jax.Array = vals[var]
        manipulability = RobotFactors.manipulability(kin, joint_cfg, target_joint_idx)
        return (1 / (manipulability + 1e-6)) * weights

    @staticmethod
    def manipulability(
        kin: JaxKinTree,
        cfg: Array,
        target_joint_idx: int,
    ) -> Array:
        """Manipulability, as the ratio of the largest to smallest singular value.
        Small -> close to losing rank -> bad manipulability."""
        jacobian = jax.jacfwd(
            lambda cfg: jaxlie.SE3(kin.forward_kinematics(cfg)).translation()
        )(cfg)[target_joint_idx]
        eigvals = jnp.linalg.svd(jacobian, compute_uv=False)  # in decreasing order
        return eigvals[-1] / (eigvals[0] + 1e-6)  # sig_N / sig_1
