"""Copied from brent's `brent_jax_trajsmooth.py`."""

from __future__ import annotations

import jax
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

# TODO self-collision!


@jdc.pytree_dataclass
class JaxUrdf:
    """A differentiable robot kinematics model."""

    num_joints: jdc.Static[int]
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
    num_spheres: jdc.Static[int]
    coll_link_idx: Int[Array, "spheres"]
    coll_link_centers: Float[Array, "spheres 3"]
    coll_link_radii: Float[Array, "spheres"]

    @staticmethod
    def from_urdf_and_coll(
        urdf: yourdfpy.URDF,
        coll_dict: Dict[str, List[Dict[str, List]]]
    ):
        # scrape the collision data.
        idx, centers, radii = [], [], []

        if len(coll_dict) == 0:
            raise UserWarning("No collision data was loaded! Maybe you forgot to load the collision geometry?")

        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            if joint.child not in coll_dict:
                continue

            list_of_spheres = coll_dict[joint.child]
            for sphere in list_of_spheres:
                idx.append(joint_idx)
                centers.append(sphere['center'])
                radii.append(sphere['radius'])

        if len(idx) == 0:
            raise UserWarning("No collision data was found! Maybe you misnamed the collision data?")

        idx = jnp.array(idx)
        centers = jnp.array(centers)
        radii = jnp.array(radii)
        num_spheres = centers.shape[0]

        assert (
            idx.shape == (num_spheres,) and
            centers.shape == (num_spheres, 3) and
            radii.shape == (num_spheres,)
        )

        jax_urdf = JaxUrdf.from_urdf(urdf)
        return JaxUrdfwithSphereCollision(
            num_joints=len(jax_urdf.parent_indices),
            joint_names=jax_urdf.joint_names,
            num_actuated_joints=jax_urdf.num_actuated_joints,
            is_actuated=jax_urdf.is_actuated,
            idx_actuated_joint=jax_urdf.idx_actuated_joint,
            joint_twists=jax_urdf.joint_twists,
            Ts_parent_joint=jnp.array(jax_urdf.Ts_parent_joint),
            limits_lower=jnp.array(jax_urdf.limits_lower),
            limits_upper=jnp.array(jax_urdf.limits_upper),
            parent_indices=jnp.array(jax_urdf.parent_indices),
            num_spheres=num_spheres,
            coll_link_idx=idx,
            coll_link_centers=centers,
            coll_link_radii=radii,
        )

    @jdc.jit
    def d_world(
        self,
        cfg: Float[Array, "num_act_joints"],
        other: SphereSDF
    ) -> Float[Array, "num_joints"]:
        """Check if the robot collides with the world, in the provided configuration.
        Get the max signed distance field (sdf) for each joint."""
        centers, radii = self.as_spheres(cfg)
        coll = SphereSDF(centers=centers, radii=radii)
        max_sdf = coll.collides_other(other)

        link_idx_as_mask = jnp.arange(self.num_joints)[:, None] == self.coll_link_idx[None]
        assert link_idx_as_mask.shape == (self.num_joints, self.num_spheres)
        assert max_sdf.shape == (self.num_spheres,)
        max_sdf_per_joint = jnp.max(
            jnp.where(
                link_idx_as_mask,
                jnp.broadcast_to(max_sdf, (self.num_joints, self.num_spheres)),
                -jnp.inf,
            ),
            axis=-1,
        )

        assert max_sdf_per_joint.shape == (self.num_joints,)
        return max_sdf_per_joint

    @jdc.jit
    def as_spheres(
        self,
        cfg: Float[Array, "*batch num_act_joints"]
    ) -> Tuple[Float[Array, "*batch num_spheres 3"], Float[Array, "*batch num_spheres"]]:
        """Get the spheres in the world frame, in the provided configuration."""
        batch_size = cfg.shape[:-1]
        num_spheres = self.coll_link_centers.shape[0]
        Ts_world_joint = self.forward_kinematics(cfg)
        assert Ts_world_joint.shape == (*batch_size, self.num_joints, 7)

        centers = jaxlie.SE3.from_translation(self.coll_link_centers)
        centers_transformed = (
            jaxlie.SE3(Ts_world_joint[..., self.coll_link_idx, :])
            @ centers
        ).translation()

        assert centers_transformed.shape == (*batch_size, num_spheres, 3)
        assert self.coll_link_radii.shape == (num_spheres,)
        return centers_transformed, self.coll_link_radii


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
    import yaml
    import viser
    import viser.extras
    import trimesh.creation
    import time

    # jax_urdf = JaxUrdfwithSphereCollision.from_urdf_and_coll(yourdf, yumi_coll)
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    # yourdf = load_robot_description("yumi_description")
    yourdf = load_robot_description("ur5_description")
    # hackily add the collisiondata for yourdf.
    yourdf._scene_collision = yourdf._create_scene(
        use_collision_geometry=True,
        load_geometry=True,
        force_mesh=True,
        force_single_geometry_per_link=True,
    )
    jax_urdf = JaxUrdfwithMeshCollision.from_urdf(yourdf)

    server = viser.ViserServer()

    urdf = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf"
    )
    # yumi_rest = onp.array([0.0] * 16)
    yumi_rest = onp.array([0.0] * 6)
    urdf.update_cfg(yumi_rest)

    tf = server.scene.add_transform_controls("point", scale=0.5)
    tf_item = server.gui.add_number("point", initial_value=0.0, step=0.001, disabled=True)
    coll_item = server.gui.add_checkbox("colliding", initial_value=False, disabled=True)

    while True:
        position = jnp.array(tf.position)[None, :]
        dist = jax_urdf.d_world(jnp.array(yumi_rest), position)
        assert dist.shape == (1,)
        tf_item.value = dist.item()
        coll_item.value = dist.item() > 0.0
        time.sleep(0.01)


if __name__ == "__main__":
    tyro.cli(main)
