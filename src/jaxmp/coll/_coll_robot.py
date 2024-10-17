"""
Differentiable robot collision model, implemented in JAX.
Supports World- and self- collision detection (returns signed distance).
"""

from __future__ import annotations
from typing import Optional, cast

from loguru import logger

import trimesh
import trimesh.bounds
import yourdfpy

import jax
from jax import Array
import jax.numpy as jnp
import numpy as onp

from jaxtyping import Float, Int
import jax_dataclasses as jdc
import jaxlie

from jaxmp.coll._coll_mjx_types import Capsule


@jdc.pytree_dataclass
class RobotColl(Capsule):
    """Collision model for a robot, which can be put into different configurations."""

    coll_link_names: jdc.Static[tuple[str]]
    """Names of the links in the robot, length `links`."""

    link_joint_idx: jdc.Static[Int[Array, "link"]]
    """Index of the parent joint for each link."""

    self_coll_matrix: jdc.Static[Int[Array, "link link"]]
    """Collision matrix, where `coll_matrix[i, j] == 1`
    if we account for the collision between collbodies `i` and `j`.
    Else, `coll_matrix[i, j] == 0`."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        self_coll_ignore: Optional[list[tuple[str, str]]] = None,
        ignore_immediate_parent: bool = True,
    ):
        """
        Build a differentiable robot collision model from a URDF.

        Args:
            urdf: The URDF object.
            self_coll_ignore: List of tuples of link names that are allowed to collide.
            ignore_immediate_parent: If True, ignore collisions between parent and child links.
        """

        # Re-load urdf, but with the collision data.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        urdf = yourdfpy.URDF(
            robot=urdf.robot,
            filename_handler=filename_handler,
            load_collision_meshes=True,
        )

        # Gather all the collision links.
        list_coll_link = list[Capsule]()
        idx_parent_joint = list[int]()
        link_names = list[str]()

        if self_coll_ignore is None:
            self_coll_ignore = []

        # Get all collision links.
        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            curr_link = joint.child
            assert curr_link in urdf.link_map

            coll_link = RobotColl._get_coll_link(urdf, curr_link)
            if coll_link is None:
                continue
            assert coll_link.get_batch_axes() == ()

            list_coll_link.append(coll_link)
            idx_parent_joint.append(joint_idx)
            link_names.append(curr_link)

            if ignore_immediate_parent:
                self_coll_ignore.append((joint.parent, joint.child))

        assert len(list_coll_link) > 0, "No collision links found in URDF."
        logger.info("Found {} collision bodies", len(list_coll_link))

        coll_links = jax.tree.map(lambda *x: jnp.stack(x), *list_coll_link)

        idx_parent_joint = jnp.array(idx_parent_joint)
        link_names = tuple[str](link_names)

        self_coll_matrix = RobotColl.create_self_coll_matrix(
            urdf, link_names, self_coll_ignore
        )
        assert self_coll_matrix.shape == (len(link_names), len(link_names))

        return RobotColl(
            pos=coll_links.pos,
            mat=coll_links.mat,
            size=coll_links.size,
            coll_link_names=link_names,
            link_joint_idx=idx_parent_joint,
            self_coll_matrix=self_coll_matrix,
        )

    @staticmethod
    def _get_coll_link(urdf: yourdfpy.URDF, curr_link: str) -> Optional[Capsule]:
        """
        Get the `CapsuleColl` collision primitives for a given link.
        """
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access

        coll_mesh_list = urdf.link_map[curr_link].collisions
        if len(coll_mesh_list) == 0:
            return None

        coll_link_mesh = trimesh.Trimesh()
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
                        file_obj=filename_handler(geom.mesh.filename), force="mesh"
                    ),
                )
                coll_mesh.fix_normals()

            if coll_mesh is None:
                raise ValueError(f"No collision mesh found for {curr_link}!")
            coll_link_mesh = coll_link_mesh + coll_mesh

        # Create the collision spheres.
        assert isinstance(coll_link_mesh, trimesh.Trimesh), type(coll_link_mesh)

        coll_link = Capsule.from_min_cylinder(coll_link_mesh)
        return coll_link

    def coll_weight(
        self, weights: dict[str, float], default: float = 1.0
    ) -> Float[Array, "links"]:
        """Get the collision weight for each sphere."""
        num_links = len(self.coll_link_names)
        coll_weights = jnp.full((num_links,), default)
        for name, weight in weights.items():
            idx = self.coll_link_names.index(name)
            coll_weights = coll_weights.at[idx].set(weight)
        return jnp.array(coll_weights)

    # def transform(self, tf: jaxlie.SE3) -> RobotColl:
    #     """Re-configure the robot using the robot joint, derived w/ forward kinematics."""
    #     Ts_world_joint = tf.wxyz_xyz  # pylint: disable=invalid-name
    #     _coll = self.transform(
    #         jaxlie.SE3(Ts_world_joint[..., self._idx_parent_joint, :])
    #     )
    #     return RobotColl(
    #         pos=_coll.pos,
    #         mat=_coll.mat,
    #         size=_coll.size,
    #         coll_link_names=self.coll_link_names,
    #         _idx_parent_joint=self._idx_parent_joint,
    #         self_coll_matrix=self.self_coll_matrix,
    #     )

    @staticmethod
    def create_self_coll_matrix(
        urdf: yourdfpy.URDF,
        coll_link_names: tuple[str],
        self_coll_ignore: list[tuple[str, str]],
    ) -> Int[Array, "link link"]:
        """
        Create a collision matrix for the robot, where `coll_matrix[i, j] == 1`.
        """

        def check_coll(i: int, j: int) -> bool:
            """Remove self- and adjacent link collisions."""
            if i == j:
                return False

            if (
                urdf.link_map[coll_link_names[i]].name,
                urdf.link_map[coll_link_names[j]].name,
            ) in self_coll_ignore:
                return False
            if (
                urdf.link_map[coll_link_names[j]].name,
                urdf.link_map[coll_link_names[i]].name,
            ) in self_coll_ignore:
                return False

            return True

        n_links = len(coll_link_names)
        coll_mat = jnp.array(
            [[check_coll(i, j) for j in range(n_links)] for i in range(n_links)]
        )
        assert coll_mat.shape == (n_links, n_links)
        return coll_mat
