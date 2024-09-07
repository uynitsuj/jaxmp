"""Common `jaxls` factors for robot control, via wrapping `JaxKinTree` and `RobotColl`.
"""
from typing import Optional
from dataclasses import dataclass

import jax
from jax import Array
import jax.numpy as jnp
import jaxlie

import jaxls

from jaxmp.kinematics import JaxKinTree
from jaxmp.collision_sdf import colldist_from_sdf, dist_signed
from jaxmp.collision_types import RobotColl, CollBody

@dataclass
class RobotFactors:
    """Common costs."""

    kin: JaxKinTree
    coll: Optional[RobotColl] = None

    def get_var_class(self, default_val: Optional[Array] = None) -> type[jaxls.Var[Array]]:
        """Get the Variable class for this robot. Default value is mid-point of limits."""
        if default_val is None:
            default_val = (self.kin.limits_upper + self.kin.limits_lower) / 2

        class JointVar(  # pylint: disable=missing-class-docstring
            jaxls.Var[Array],
            default=default_val,
            tangent_dim=self.kin.num_actuated_joints,
            retract_fn=self.kin.get_retract_fn(),
        ): ...

        return JointVar

    def ik_cost(
        self,
        vals: jaxls.VarValues,
        var: jaxls.Var[Array],
        target_pose: jaxlie.SE3,
        target_joint_idx: int,
        weights: Array,
    ) -> Array:
        """Pose cost."""
        joint_cfg: jax.Array = vals[var]
        Ts_joint_world = self.kin.forward_kinematics(joint_cfg)
        residual = (
            jaxlie.SE3(Ts_joint_world[target_joint_idx]).inverse()
            @ target_pose
        ).log()
        assert residual.shape == weights.shape
        return residual * weights

    def limit_cost(
        self,
        vals: jaxls.VarValues,
        var: jaxls.Var[Array],
        weights: Array,
    ) -> Array:
        """Limit cost."""
        joint_cfg: jax.Array = vals[var]
        residual = (
            jnp.maximum(0.0, joint_cfg - self.kin.limits_upper) +
            jnp.maximum(0.0, self.kin.limits_lower - joint_cfg)
        )
        assert residual.shape == weights.shape
        return residual * weights

    def rest_cost(
        self,
        vals: jaxls.VarValues,
        var: jaxls.Var[Array],
        weights: Array,
    ) -> Array:
        """Bias towards joints at rest pose, specified by `default`."""
        assert var.default is not None
        assert var.default.shape == vals[var].shape and var.default.shape == weights.shape
        return (vals[var] - var.default) * weights

    def self_coll_cost(
        self,
        vals: jaxls.VarValues,
        var: jaxls.Var[Array],
        weights: Array,
    ) -> Array:
        """Collision-scaled dist for self-collision."""
        assert self.coll is not None
        joint_cfg = vals[var]
        coll = self.coll.transform(jaxlie.SE3(self.kin.forward_kinematics(joint_cfg)))
        sdf = dist_signed(coll, coll)
        weights = weights[:, None] * weights[None, :]
        assert sdf.shape == weights.shape
        return (colldist_from_sdf(sdf) * weights).flatten()

    def world_coll_cost(
        self,
        vals: jaxls.VarValues,
        var: jaxls.Var[Array],
        other: CollBody,
        weights: Array,
    ) -> Array:
        """Collision-scaled dist for world collisio."""
        assert self.coll is not None
        joint_cfg = vals[var]
        coll = self.coll.transform(jaxlie.SE3(self.kin.forward_kinematics(joint_cfg)))
        sdf = dist_signed(coll, other).flatten()
        assert sdf.shape == weights.shape
        return colldist_from_sdf(sdf) * weights

    def smoothness_cost(
        self,
        vals: jaxls.VarValues,
        var_curr: jaxls.Var[Array],
        var_past: jaxls.Var[Array],
        weights: Array,
    ) -> Array:
        """Smoothness cost, for trajectories etc."""
        residual = (vals[var_curr] - vals[var_past])
        assert residual.shape == weights.shape
        return residual * weights

    def manipulability_cost(
        self,
        vals: jaxls.VarValues,
        var: jaxls.Var[Array],
        target_joint_idx: int,
        weights: Array,
    ):
        """Manipulability cost."""
        joint_cfg: jax.Array = vals[var]
        manipulability = self.manipulability(joint_cfg, target_joint_idx)
        return (1 / (manipulability + 1e-6)) * weights

    def manipulability(
        self,
        cfg: Array,
        target_joint_idx: int,
    ) -> Array:
        """Manipulability, as the ratio of the largest to smallest singular value.
        Small -> close to losing rank -> bad manipulability."""
        jacobian = jax.jacfwd(
            lambda cfg: jaxlie.SE3(self.kin.forward_kinematics(cfg)).translation()
        )(cfg)[target_joint_idx]
        eigvals = jnp.linalg.svd(jacobian, compute_uv=False)  # in decreasing order
        return eigvals[-1] / (eigvals[0] + 1e-6)  # sig_N / sig_1
