import yaml
from pathlib import Path
import viser
import viser.extras
import viser.transforms as vtf
from robot_descriptions.loaders.yourdfpy import load_robot_description
import tyro
import jax
import jax.numpy as jnp
import numpy as onp
import trimesh.creation
import jax_dataclasses as jdc
from typing_extensions import override
from typing import cast

from jaxtyping import Float, Array, Int
import jaxfg
import jaxlie

from jaxmp.urdf import JaxUrdfwithCollision
from jaxmp.collision import SphereCollision

@jdc.jit
def get_factors_ik(
    urdf: JaxUrdfwithCollision,
    pose_target_from_joint_idx: dict[int, Float[Array, "7"]],
    rest_pose: Float[Array, "joints"],
    coll_env: SphereCollision,
    *,
    rest_prior_weight: float = 0.001,
    pos_weight: float = 50.0,
    ori_weight: float = 0.1,
    limits_weight: float = 100_000.0,
    limits_pad: float = 0.1,
) -> tuple[jaxfg.core.VariableBase, list[jaxfg.core.FactorBase]]:
    """"""

    Vec16Variable = jaxfg.core.RealVectorVariable[16]

    pose_dim = next(iter(pose_target_from_joint_idx.values())).shape
    assert pose_dim == (7,)  # wxyz, xyz
    q_var = Vec16Variable()

    factors = list[jaxfg.core.FactorBase]()

    # Per-joint cost terms.
    @jdc.pytree_dataclass
    class RegFactor(jaxfg.core.FactorBase):
        reg_pose: Float[Array, "num_joints"]
        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (q_t,) = variable_values
            return q_t - rest_pose

    @jdc.pytree_dataclass
    class UpperLimitFactor(jaxfg.core.FactorBase):
        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (q_t,) = variable_values
            return jnp.maximum(
                # We don't pad the gripper joint limits!
                0.0,
                q_t - (urdf.limits_upper.at[:14].add(-limits_pad)),
            )

    @jdc.pytree_dataclass
    class LowerLimitFactor(jaxfg.core.FactorBase):
        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (q_t,) = variable_values
            return jnp.minimum(
                # We don't pad the gripper joint limits!
                0.0,
                q_t - (urdf.limits_lower.at[:14].add(limits_pad)),
            )

    factors.extend(
        [
            RegFactor(
                variables=(q_var,),
                reg_pose=rest_pose,
                noise_model=jaxfg.noises.DiagonalGaussian(
                    jnp.ones(16) * rest_prior_weight
                ),
            ),
            UpperLimitFactor(
                variables=(q_var,),
                noise_model=jaxfg.noises.DiagonalGaussian(
                    jnp.ones(16) * limits_weight
                ),
            ),
            LowerLimitFactor(
                variables=(q_var,),
                noise_model=jaxfg.noises.DiagonalGaussian(
                    jnp.ones(16) * limits_weight
                ),
            ),
        ]
    )
    return (q_var, factors)


@jdc.jit
def ik(
    yumi_urdf: JaxUrdfwithCollision,
    pose_target_from_joint_idx: dict[int, Float[Array, "7"]],
    rest_pose: Float[Array, "16"],
    coll_env: SphereCollision,
    *,
    smooth_weight: float = 0.01,
    rest_prior_weight: float = 0.001,
    pos_weight: float = 50.0,
    ori_weight: float = 0.1,
    coll_weight: float = 10.0,
    limits_weight: float = 100_000.0,
    limits_pad: float = 0.1,
) -> Array:
    """Smooth a trajectory, while holding the output frames of some set of
    joints fixed."""
    Vec16Variable = jaxfg.core.RealVectorVariable[16]

    pose_dim = next(iter(pose_target_from_joint_idx.values())).shape
    assert pose_dim == (7,)  # wxyz, xyz
    q_var = Vec16Variable()

    factors = list[jaxfg.core.FactorBase]()

    @jdc.pytree_dataclass
    class CollisionFactor(jaxfg.core.FactorBase):
        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (q_t,) = variable_values
            return jnp.maximum(0.0, yumi_urdf.d_world(cfg=q_t, other=coll_env))

    factors.append(
        CollisionFactor(
            variables=(q_var,),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(16) * coll_weight),
        )
    )

    # Inverse kinematics term.
    @jdc.pytree_dataclass
    class InverseKinematicsFactor(jaxfg.core.FactorBase):
        joint_index: Int[Array, ""]
        pose_target: Float[Array, "7"]

        @override
        def compute_residual_vector(self, variable_values: tuple[Array]) -> Array:
            (joints,) = variable_values
            assert self.joint_index.shape == ()
            Ts_world_joint = yumi_urdf.forward_kinematics(joints)
            assert Ts_world_joint.shape == (yumi_urdf.num_joints, 7)
            assert self.pose_target.shape == (7,)
            return (
                jaxlie.SE3(Ts_world_joint[self.joint_index]).inverse()
                @ jaxlie.SE3(self.pose_target)
            ).log()

    ik_weight = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    for joint_idx, pose_target in pose_target_from_joint_idx.items():
        factors.append(
            InverseKinematicsFactor(
                variables=(q_var,),
                noise_model=jaxfg.noises.DiagonalGaussian(ik_weight),
                joint_index=jnp.array(joint_idx),
                pose_target=pose_target,
            ),
        )

    graph = jaxfg.core.StackedFactorGraph.make(factors, use_onp=False)
    solver = jaxfg.solvers.LevenbergMarquardtSolver(
        lambda_initial=0.1, gradient_tolerance=1e-5, parameter_tolerance=1e-5, verbose=False
    )
    assignments = jaxfg.core.VariableAssignments.make_from_dict({q_var: rest_pose})
    solved_assignments = solver.solve(graph, assignments)
    out = solved_assignments.get_value(q_var)
    assert out.shape == (16,)
    return cast(Array, out)