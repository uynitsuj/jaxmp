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
        coll = self.coll.transform(jaxlie.SE3(self.kin.forward_kinematics(vals[var])))
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
        coll = self.coll.transform(jaxlie.SE3(self.kin.forward_kinematics(vals[var])))
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
        """Manipulability cost, by increasing the nuclear norm of the Jacobian."""
        joint_cfg: jax.Array = vals[var]

        # Jacobian between wxyz_xyz, and dof.
        jacobian = jax.jacfwd(self.kin.forward_kinematics)(joint_cfg)[target_joint_idx]
        norm = jnp.linalg.norm(jacobian, ord='nuc')

        return jnp.maximum(6.0 - norm[None], 0.0) * weights
