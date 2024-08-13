"""Copied from brent's `brent_jax_trajsmooth.py`."""

from __future__ import annotations

from copy import deepcopy

import jax
import jax.experimental
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import viser.transforms as vtf
import yourdfpy
import warnings
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int, Bool
import trimesh
from typing import Dict, List, Tuple, Optional, cast
import tyro

from jaxmp.collision import SphereSDF, MeshSDF
from jaxmp.collbody import Spheres

# define allowed collision matrix

@jdc.pytree_dataclass
class JaxUrdf:
    """A differentiable robot kinematics model."""

    num_joints: jdc.Static[int]
    num_links: jdc.Static[int]
    joint_names: jdc.Static[Tuple[str]]
    """List of joint names, in order."""

    num_actuated_joints: jdc.Static[int]
    idx_actuated_joint: Int[Array, "joints"]
    """Index of actuated joint in `act_joints`, if it is actuated. -1 otherwise."""
    is_actuated: Bool[Array, "joints"]

    joint_twists: Float[Array, "act_joints 6"]
    limits_lower: Float[Array, "act_joints"]
    limits_upper: Float[Array, "act_joints"]

    Ts_parent_joint: Float[Array, "joints 7"]
    parent_indices: Int[Array, "joints"]

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
    ) -> JaxUrdf:
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
                    warnings.warn(f"Parent index {parent_index} >= joint index {joint_idx}! Assuming that parent is root.")
                    if parent_joint.parent != urdf.scene.graph.base_frame:
                        raise ValueError("Parent index >= joint_index, but parent is not root!")
                    T_parent_joint = parent_joint.origin @ T_parent_joint  # T_root_joint.
                    parent_index = -1
                parent_indices.append(parent_index)

            Ts_parent_joint.append(vtf.SE3.from_matrix(T_parent_joint).wxyz_xyz)  # type: ignore

        joint_twists = jnp.array(joint_twists)
        limits_lower = jnp.array(joint_lim_lower)
        limits_upper = jnp.array(joint_lim_upper)
        Ts_parent_joint = jnp.array(Ts_parent_joint)
        parent_indices = jnp.array(parent_indices)
        idx_actuated_joint = jnp.array(idx_actuated_joint)
        is_actuated = jnp.where(
            idx_actuated_joint != -1,
            jnp.ones_like(idx_actuated_joint),
            jnp.zeros_like(idx_actuated_joint),
        )
        joint_names = tuple[str](joint_names)

        num_joints = len(urdf.joint_map)
        num_links = len(urdf.link_map)
        num_actuated_joints = len(urdf.actuated_joints)

        assert idx_actuated_joint.shape == (len(urdf.joint_map),)
        assert joint_twists.shape == (num_actuated_joints, 6)
        assert limits_lower.shape == (num_actuated_joints,)
        assert limits_upper.shape == (num_actuated_joints,)
        assert Ts_parent_joint.shape == (num_joints, 7)
        assert parent_indices.shape == (num_joints,)
        assert idx_actuated_joint.max() == num_actuated_joints - 1

        return JaxUrdf(
            num_joints=num_joints,
            num_links=num_links,
            joint_names=joint_names,
            num_actuated_joints=num_actuated_joints,
            idx_actuated_joint=idx_actuated_joint,
            is_actuated=is_actuated,
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
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.num_actuated_joints), f"""Got {cfg.shape}, expected {(*batch_axes, self.num_actuated_joints)}"""

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

    @jdc.jit
    def forward_kinematics_tangent(
        self,
        cfg: Float[Array, "*batch num_act_joints"],
    ) -> Float[Array, "*batch num_act_joints 7"]:
        batch_axes = cfg.shape[:-1]
        Ts_world_joint_wxyz_xyz = self.forward_kinematics(cfg)

        # Convert to tangent space.
        Ts_world_joint_tangent = jaxlie.SE3(Ts_world_joint_wxyz_xyz).log()
        assert Ts_world_joint_tangent.shape == (*batch_axes, self.num_joints, 6)

        # Only return the actuated joints!
        Ts_world_joint_tangent_act = Ts_world_joint_tangent[..., self.is_actuated, :]

        return Ts_world_joint_tangent_act


@jdc.pytree_dataclass
class JaxUrdfwithSphereCollision(JaxUrdf):
    """A differentiable, collision-aware robot kinematics model."""
    _spheres: Spheres
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
        coll_link_spheres: list[Spheres] = []
        coll_link_names = []
        if self_coll_ignore is None:
            self_coll_ignore = []

        # First, create a urdf with the collision data!
        urdf = yourdfpy.URDF(
            robot=urdf.robot,
            filename_handler=urdf._filename_handler,
            load_collision_meshes=True,
        )

        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            curr_link = joint.child
            coll_mesh_list = urdf.link_map[curr_link].collisions
            if len(coll_mesh_list) == 0:
                continue
            assert len(coll_mesh_list) == 1, coll_mesh_list
            print(f"Found collision mesh for {curr_link}.")

            # Handle different geometry types.
            coll_mesh: Optional[trimesh.Trimesh] = None
            geom = coll_mesh_list[0].geometry
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
                        file_obj=urdf._filename_handler(geom.mesh.filename),
                        force="mesh"
                    ),
                )
                coll_mesh.fix_normals()

            if coll_mesh is None:
                raise ValueError(f"No collision mesh found for {curr_link}!")

            # Create the collision spheres.
            assert isinstance(coll_mesh, trimesh.Trimesh), type(coll_mesh)
            spheres = Spheres.from_voronoi(coll_mesh)
            coll_link_to_joint.append([joint_idx] * spheres.n_pts)
            coll_link_spheres.append(spheres)
            coll_link_names.append([curr_link] * spheres.n_pts)

            # Add the parent/child link to the allowed collision list.
            if ignore_immediate_parent:
                self_coll_ignore.append((joint.parent, curr_link))

        coll_link_to_joint = jnp.array(sum(coll_link_to_joint, []))
        spheres = Spheres(
            n_pts=sum(s.n_pts for s in coll_link_spheres),
            centers=jnp.concatenate([s.centers for s in coll_link_spheres]),
            radii=jnp.concatenate([s.radii for s in coll_link_spheres]),
        )
        assert coll_link_to_joint.shape[0] == spheres.n_pts

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
                [check_coll(i, j) for j in range(spheres.n_pts)]
                for i in range(spheres.n_pts)
            ]
        )
        assert coll_mat.shape == (spheres.n_pts, spheres.n_pts)

        print(f"Found {spheres.n_pts} collision spheres.")

        jax_urdf = JaxUrdf.from_urdf(urdf)
        return JaxUrdfwithSphereCollision(
            num_joints=len(jax_urdf.parent_indices),
            num_links=jax_urdf.num_links,
            joint_names=jax_urdf.joint_names,
            num_actuated_joints=jax_urdf.num_actuated_joints,
            is_actuated=jax_urdf.is_actuated,
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
        other: Spheres
    ) -> Float[Array, "num_spheres"]:
        """Check if the robot collides with the world, in the provided configuration.
        Get the max signed distance field (sdf) for each joint."""
        self_spheres = self.spheres(cfg)
        dist = Spheres.dist(self_spheres, other)
        dist = dist.max(axis=1)
        assert dist.shape == (self_spheres.n_pts,)
        return dist

    @jdc.jit
    def d_self(
        self,
        cfg: Float[Array, "num_act_joints"],
    ) -> Float[Array, "num_links"]:
        """Check if the robot collides with itself, in the provided configuration.
        Get the max signed distance field (sdf) for each joint. sdf > 0 means collision."""
        self_spheres = self.spheres(cfg)
        dist = Spheres.dist(self_spheres, self_spheres)
        assert dist.shape == (self_spheres.n_pts, self_spheres.n_pts)
        dist = (dist * self._coll_mat).flatten()
        return dist

    @jdc.jit
    def spheres(
        self,
        cfg: Float[Array, "*batch num_act_joints"]
    ) -> Spheres:
        """Get the spheres in the world frame, in the provided configuration."""
        batch_size = cfg.shape[:-1]
        num_spheres = self._spheres.n_pts
        Ts_world_joint = self.forward_kinematics(cfg)
        assert Ts_world_joint.shape == (*batch_size, self.num_joints, 7)

        centers = jaxlie.SE3.from_translation(self._spheres.centers)
        centers_transformed = (
            jaxlie.SE3(Ts_world_joint[..., self._coll_link_to_joint, :])
            @ centers
        ).translation()

        assert centers_transformed.shape == (*batch_size, num_spheres, 3)
        return Spheres(
            n_pts=num_spheres,
            centers=centers_transformed,
            radii=self._spheres.radii,
        )


@jdc.pytree_dataclass
class JaxUrdfwithMeshCollision(JaxUrdf):
    """A differentiable, collision-aware robot kinematics model."""
    num_coll_links: jdc.Static[int]
    coll_link_idx: Int[Array, "coll_link"]
    coll_link_meshes: jdc.Static[Tuple[MeshSDF]]  # want to keep them separate for self-coll.

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
    ) -> JaxUrdfwithMeshCollision:
        # URDF must have collision data.
        assert urdf._scene_collision is not None

        jax_urdf = JaxUrdf.from_urdf(urdf)

        # scrape the collision data.
        coll_link_idx = []
        coll_link_meshes = []

        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            curr_link = joint.child
            coll_mesh_list = urdf.link_map[curr_link].collisions
            if len(coll_mesh_list) == 0:
                continue
            assert len(coll_mesh_list) == 1, coll_mesh_list
            print(f"Found collision mesh for {curr_link}.")

            # Handle different geometry types.
            coll_mesh: Optional[trimesh.Trimesh] = None
            geom = coll_mesh_list[0].geometry
            if geom.box is not None:
                coll_mesh = trimesh.creation.box(geom.box.size)
            elif geom.cylinder is not None:
                coll_mesh = trimesh.creation.cylinder(geom.cylinder.radius, geom.cylinder.length)
            elif geom.sphere is not None:
                coll_mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
            elif geom.mesh is not None:
                coll_mesh = cast(
                    trimesh.Trimesh,
                    trimesh.load(urdf._filename_handler(geom.mesh.filename), force="mesh"),
                )
                coll_mesh.fix_normals()

            if coll_mesh is None:
                raise ValueError(f"No collision mesh found for {curr_link}!")

            assert isinstance(coll_mesh, trimesh.Trimesh), type(coll_mesh)
            coll_link_idx.append(joint_idx)
            coll_link_meshes.append(MeshSDF.from_trimesh(coll_mesh))
        coll_link_meshes = tuple(coll_link_meshes)

        return JaxUrdfwithMeshCollision(
            num_joints=len(jax_urdf.parent_indices),
            num_links=jax_urdf.num_links,
            joint_names=jax_urdf.joint_names,
            num_actuated_joints=jax_urdf.num_actuated_joints,
            is_actuated=jax_urdf.is_actuated,
            idx_actuated_joint=jax_urdf.idx_actuated_joint,
            joint_twists=jax_urdf.joint_twists,
            Ts_parent_joint=jnp.array(jax_urdf.Ts_parent_joint),
            limits_lower=jnp.array(jax_urdf.limits_lower),
            limits_upper=jnp.array(jax_urdf.limits_upper),
            parent_indices=jnp.array(jax_urdf.parent_indices),
            num_coll_links=len(coll_link_meshes),
            coll_link_idx=jnp.array(coll_link_idx),
            coll_link_meshes=coll_link_meshes,
        )

    def d_world(
        self,
        cfg: Float[Array, "num_act_joints"],
        points: Float[Array, "num_points 3"]
    ) -> Float[Array, "num_points"]:
        # Point is in world frame.
        n_points = points.shape[0]

        Ts_world_joint = self.forward_kinematics(cfg)
        assert Ts_world_joint.shape == (self.num_joints, 7)

        # Expand to [num_joints, num_points, 3], putting points in joint frame.
        def compute_dist(i: int, dists: Array) -> Array:
            points_in_joint_frame = jaxlie.SE3(Ts_world_joint[self.coll_link_idx[i]]).inverse() @ points
            assert points_in_joint_frame.shape == (n_points, 3)
            dist = self.coll_link_meshes[i].d_points(points_in_joint_frame)
            assert dist.shape == (n_points,)
            return dists.at[i, :].set(dist)

        dists = jnp.zeros((self.num_coll_links, n_points))
        for i in range(self.num_coll_links):
            dists = compute_dist(i, dists)

        # Want to take the maximum SDF over all the links.
        min_dist_idx = jnp.nanargmax(dists, axis=0)
        min_dist = jnp.take_along_axis(dists, min_dist_idx[None], axis=0)[:, 0]

        assert min_dist.shape == (n_points,), min_dist.shape
        return min_dist


def main():
    """Small test script to visualize the Yumi robot, and the collision spheres, in viser."""
    from pathlib import Path
    import viser
    import viser.extras
    import trimesh.creation
    import time

    from robot_descriptions.loaders.yourdfpy import load_robot_description
    yourdf = load_robot_description("yumi_description")
    yumi_rest = onp.array([0.0] * 16)
    yourdf = load_robot_description("ur5_description")
    yumi_rest = onp.array([0.0] * 6)
    jax_urdf = JaxUrdfwithSphereCollision.from_urdf(yourdf)

    server = viser.ViserServer()

    urdf = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf"
    )
    urdf.update_cfg(yumi_rest)

    tf = server.scene.add_transform_controls("point", scale=0.5)
    tf_item = server.gui.add_number("point", initial_value=0.0, step=0.001, disabled=True)
    coll_item = server.gui.add_checkbox("colliding", initial_value=False, disabled=True)

    server.scene.add_mesh_trimesh(
        "spheres",
        jax_urdf.spheres(jnp.array(yumi_rest)).to_trimesh(),
    )

    while True:
        position = jnp.array(tf.position)[None, :]
        dist = jax_urdf.d_world(
            jnp.array(yumi_rest),
            Spheres.from_points(position),
        )
        tf_item.value = dist.item()
        coll_item.value = dist.item() > 0.0
        time.sleep(0.01)


if __name__ == "__main__":
    tyro.cli(main)
