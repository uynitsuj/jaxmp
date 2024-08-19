"""
Differentiable robot kinematics model, implemented in JAX.
Includes:
 - URDF parsing
 - Forward kinematics
 - World- and self- collision detection (returns signed distance).
"""
# pylint: disable=invalid-name

from __future__ import annotations

from typing import Optional, cast, Union
import warnings
import trimesh

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import yourdfpy

from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int

from jaxmp.collbody import SphereColl, PlaneColl


@jdc.pytree_dataclass
class JaxKinematics:
    """A differentiable robot kinematics model."""

    num_joints: jdc.Static[int]
    joint_names: jdc.Static[tuple[str]]
    """List of joint names, in order."""

    num_actuated_joints: jdc.Static[int]
    idx_actuated_joint: Int[Array, "joints"]
    """Index of actuated joint in `act_joints`, if it is actuated. -1 otherwise."""

    joint_twists: Float[Array, "act_joints 6"]
    limits_lower: Float[Array, "act_joints"]
    limits_upper: Float[Array, "act_joints"]

    Ts_parent_joint: Float[Array, "joints 7"]
    parent_indices: Int[Array, "joints"]

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
    ) -> JaxKinematics:
        """Build a differentiable robot model from a URDF."""

        joint_from_child = {joint.child: joint for joint in urdf.joint_map.values()}

        # Get the parent indices + joint twist parameters.
        joint_twists = list[onp.ndarray]()
        Ts_parent_joint = list[onp.ndarray]()
        joint_lim_lower = list[float]()
        joint_lim_upper = list[float]()
        parent_indices = list[int]()
        joint_names = list[str]()
        idx_actuated_joint = list[int]()

        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            assert joint.origin.shape == (4, 4)
            joint_names.append(joint.name)

            # Check if this joint is a mimic joint.
            if joint.mimic is not None:
                mimicked_joint = urdf.joint_map[joint.mimic.joint]
                mimicked_joint_idx = urdf.actuated_joints.index(mimicked_joint)
                assert mimicked_joint_idx < joint_idx, "Code + fk `fori_loop` assumes this!"
                warnings.warn("Mimic joint detected, assuming multiplier=1.0, offset=0.0.")
                act_joint_idx = urdf.actuated_joints.index(mimicked_joint)
                idx_actuated_joint.append(act_joint_idx)

                # ... skip the twist info, since mimic joints are not actuated.
            elif joint in urdf.actuated_joints:
                assert joint.axis.shape == (3,)
                assert (
                    joint.limit is not None
                    and joint.limit.lower is not None
                    and joint.limit.upper is not None
                ), "We currently assume there are joint limits!"
                joint_lim_lower.append(joint.limit.lower)
                joint_lim_upper.append(joint.limit.upper)

                act_joint_idx = urdf.actuated_joints.index(joint)
                idx_actuated_joint.append(act_joint_idx)

                # We use twists in the (v, omega) convention.
                if joint.type == "revolute":
                    joint_twists.append(onp.concatenate([onp.zeros(3), joint.axis]))
                elif joint.type == "prismatic":
                    joint_twists.append(onp.concatenate([joint.axis, onp.zeros(3)]))
                else:
                    raise ValueError(f"Unsupported joint type {joint.type}!")
            else:
                idx_actuated_joint.append(-1)

            # Get the transform from the parent joint to the current joint.
            # Do this for all the joints.
            T_parent_joint = joint.origin
            if joint.parent not in joint_from_child:
                # Must be root node.
                parent_indices.append(-1)
            else:
                parent_joint = joint_from_child[joint.parent]
                parent_index = urdf.joint_names.index(parent_joint.name)
                if parent_index >= joint_idx:
                    warnings.warn(
                        f"Parent index {parent_index} >= joint index {joint_idx}!" +
                        "Assuming that parent is root."
                    )
                    if parent_joint.parent != urdf.scene.graph.base_frame:
                        raise ValueError("Parent index >= joint_index, but parent is not root!")
                    T_parent_joint = parent_joint.origin @ T_parent_joint  # T_root_joint.
                    parent_index = -1
                parent_indices.append(parent_index)

            Ts_parent_joint.append(jaxlie.SE3.from_matrix(T_parent_joint).wxyz_xyz)  # type: ignore

        joint_twists = jnp.array(joint_twists)
        limits_lower = jnp.array(joint_lim_lower)
        limits_upper = jnp.array(joint_lim_upper)
        Ts_parent_joint = jnp.array(Ts_parent_joint)
        parent_indices = jnp.array(parent_indices)
        idx_actuated_joint = jnp.array(idx_actuated_joint)
        joint_names = tuple[str](joint_names)

        num_joints = len(urdf.joint_map)
        num_actuated_joints = len(urdf.actuated_joints)

        assert idx_actuated_joint.shape == (len(urdf.joint_map),)
        assert joint_twists.shape == (num_actuated_joints, 6)
        assert limits_lower.shape == (num_actuated_joints,)
        assert limits_upper.shape == (num_actuated_joints,)
        assert Ts_parent_joint.shape == (num_joints, 7)
        assert parent_indices.shape == (num_joints,)
        assert idx_actuated_joint.max() == num_actuated_joints - 1

        return JaxKinematics(
            num_joints=num_joints,
            joint_names=joint_names,
            num_actuated_joints=num_actuated_joints,
            idx_actuated_joint=idx_actuated_joint,
            joint_twists=joint_twists,
            Ts_parent_joint=Ts_parent_joint,
            limits_lower=limits_lower,
            limits_upper=limits_upper,
            parent_indices=parent_indices,
        )

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch num_act_joints"],
    ) -> Float[Array, "*batch num_joints 7"]:
        """
        Run forward kinematics on the robot, in the provided configuration.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch num_act_joints)`.
        
        Returns:
            The SE(3) transforms of the joints, in the format `(*batch num_joints wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.num_actuated_joints)

        Ts_joint_child = jaxlie.SE3.exp(self.joint_twists * cfg[..., None]).wxyz_xyz
        assert Ts_joint_child.shape == (*batch_axes, self.num_actuated_joints, 7)

        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                self.parent_indices[i] == -1,
                jnp.broadcast_to(jaxlie.SE3.identity().wxyz_xyz, (*batch_axes, 7)),
                Ts_world_joint[..., self.parent_indices[i], :],
            )

            T_joint_child = jnp.where(
                self.idx_actuated_joint[i] != -1,
                Ts_joint_child[..., self.idx_actuated_joint[i], :],
                jnp.broadcast_to(jaxlie.SE3.identity().wxyz_xyz, (*batch_axes, 7)),
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent)
                    @ jaxlie.SE3(self.Ts_parent_joint[i])
                    @ jaxlie.SE3(T_joint_child)
                ).wxyz_xyz
            )

        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=self.num_joints,
            body_fun=compute_joint,
            init_val=jnp.zeros((*batch_axes, self.num_joints, 7)),
        )
        assert Ts_world_joint.shape == (*batch_axes, self.num_joints, 7)
        return Ts_world_joint


@jdc.pytree_dataclass
class JaxCollKinematics(JaxKinematics):
    """A differentiable, collision-aware robot kinematics model."""
    _spheres: SphereColl
    """Collision spheres for the robot."""

    _coll_link_to_joint: Int[Array, "spheres"]
    """Mapping from collision spheres to links."""

    _coll_mat: Int[Array, "spheres spheres"]
    """Collision matrix, defined on per-sphere basis."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        self_coll_ignore: Optional[list[tuple[str, str]]] = None,
        ignore_immediate_parent: bool = True,
    ):
        """
        Build a differentiable robot model from a URDF.
        This model also includes collision information for the robot.

        Args:
            urdf: The URDF object.
            self_coll_ignore: List of tuples of link names that are allowed to collide.
            ignore_immediate_parent: If True, ignore collisions between parent and child links.
        """
        # scrape the collision data.
        coll_link_to_joint = []
        coll_link_spheres: list[SphereColl] = []
        coll_link_names = []
        if self_coll_ignore is None:
            self_coll_ignore = []

        # First, create a urdf with the collision data!
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        urdf = yourdfpy.URDF(
            robot=urdf.robot,
            filename_handler=filename_handler,
            load_collision_meshes=True,
        )

        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            curr_link = joint.child
            coll_mesh_list = urdf.link_map[curr_link].collisions
            if len(coll_mesh_list) == 0:
                continue

            coll_link_mesh = trimesh.Trimesh()
            print(f"Found collision mesh for {curr_link}.")
            for coll in coll_mesh_list:
                # Handle different geometry types.
                coll_mesh: Optional[trimesh.Trimesh] = None
                geom = coll.geometry
                if geom.box is not None:
                    coll_mesh = trimesh.creation.box(extents=geom.box.size)
                elif geom.cylinder is not None:
                    coll_mesh = trimesh.creation.cylinder(
                        radius=geom.cylinder.radius, height=geom.cylinder.length
                    )
                elif geom.sphere is not None:
                    coll_mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
                elif geom.mesh is not None:
                    coll_mesh = cast(
                        trimesh.Trimesh,
                        trimesh.load(
                            file_obj=filename_handler(geom.mesh.filename),
                            force="mesh"
                        ),
                    )
                    coll_mesh.fix_normals()

                if coll_mesh is None:
                    raise ValueError(f"No collision mesh found for {curr_link}!")
                coll_link_mesh = coll_link_mesh + coll_mesh

            # Create the collision spheres.
            assert isinstance(coll_link_mesh, trimesh.Trimesh), type(coll_link_mesh)
            spheres = SphereColl.from_min_ball(coll_link_mesh)
            n_pts = spheres.centers.shape[0]
            coll_link_to_joint.append([joint_idx] * n_pts)
            coll_link_spheres.append(spheres)
            coll_link_names.append([curr_link] * n_pts)

            # Add the parent/child link to the allowed collision list.
            if ignore_immediate_parent:
                self_coll_ignore.append((joint.parent, curr_link))

        coll_link_to_joint = jnp.array(sum(coll_link_to_joint, []))
        spheres = SphereColl(
            centers=jnp.concatenate([s.centers for s in coll_link_spheres]),
            radii=jnp.concatenate([s.radii for s in coll_link_spheres]),
        )
        n_pts = spheres.centers.shape[0]
        assert coll_link_to_joint.shape[0] == n_pts

        # Calculate the allowed collision matrix.
        coll_link_names = sum(coll_link_names, [])

        def check_coll(i: int, j: int) -> bool:
            if i == j:
                return False

            if (
                urdf.link_map[coll_link_names[i]].name,
                urdf.link_map[coll_link_names[j]].name,
            ) in self_coll_ignore:
                return False
            elif (
                urdf.link_map[coll_link_names[j]].name,
                urdf.link_map[coll_link_names[i]].name,
            ) in self_coll_ignore:
                return False

            return True

        coll_mat = jnp.array(
            [
                [check_coll(i, j) for j in range(n_pts)]
                for i in range(n_pts)
            ]
        )
        assert coll_mat.shape == (n_pts, n_pts)

        print(f"Found {n_pts} collision spheres.")

        jax_urdf = JaxKinematics.from_urdf(urdf)
        return JaxCollKinematics(
            num_joints=len(jax_urdf.parent_indices),
            joint_names=jax_urdf.joint_names,
            num_actuated_joints=jax_urdf.num_actuated_joints,
            idx_actuated_joint=jax_urdf.idx_actuated_joint,
            joint_twists=jax_urdf.joint_twists,
            Ts_parent_joint=jnp.array(jax_urdf.Ts_parent_joint),
            limits_lower=jnp.array(jax_urdf.limits_lower),
            limits_upper=jnp.array(jax_urdf.limits_upper),
            parent_indices=jnp.array(jax_urdf.parent_indices),
            _coll_link_to_joint=coll_link_to_joint,
            _spheres=spheres,
            _coll_mat=coll_mat,
        )

    @jdc.jit
    def d_world(
        self,
        cfg: Float[Array, "num_act_joints"],
        other: Union[SphereColl, PlaneColl]
    ) -> Float[Array, "num_spheres"]:
        """Check if the robot collides with the world, in the provided configuration.
        Get the max signed distance field (sdf) for each joint."""
        self_spheres = self.spheres(cfg)
        n_pts = self_spheres.centers.shape[0]
        if isinstance(other, SphereColl):
            dist = self_spheres.dist(other)
            dist = dist.max(axis=1)
        elif isinstance(other, PlaneColl):
            dist = self_spheres.dist_to_plane(other)
        assert dist.shape == (n_pts,)
        return dist

    @jdc.jit
    def d_self(
        self,
        cfg: Float[Array, "num_act_joints"],
    ) -> Float[Array, "num_spheres"]:
        """Check if the robot collides with itself, in the provided configuration.
        Get the max signed distance field (sdf) for each joint. sdf > 0 means collision.
        Jitted, since the robot embodiment should not change during runtime.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch num_act_joints)`.
        
        Return:
            The signed distance field (sdf) for the robot, in the format `(*batch num_spheres)`.
        """
        self_spheres = self.spheres(cfg)
        n_pts = self_spheres.centers.shape[0]
        dist = SphereColl.dist(self_spheres, self_spheres)
        assert dist.shape == (n_pts, n_pts)
        dist = (dist * self._coll_mat).flatten()
        return dist

    @jdc.jit
    def spheres(
        self,
        cfg: Float[Array, "*batch num_act_joints"]
    ) -> SphereColl:
        """Get the spheres in the world frame, in the provided configuration."""
        batch_size = cfg.shape[:-1]
        num_spheres = self._spheres.centers.shape[0]
        Ts_world_joint = self.forward_kinematics(cfg)
        assert Ts_world_joint.shape == (*batch_size, self.num_joints, 7)

        centers = jaxlie.SE3.from_translation(self._spheres.centers)
        centers_transformed = (
            jaxlie.SE3(Ts_world_joint[..., self._coll_link_to_joint, :])
            @ centers
        ).translation()

        assert centers_transformed.shape == (*batch_size, num_spheres, 3)
        return SphereColl(
            centers=centers_transformed,
            radii=self._spheres.radii,
        )
