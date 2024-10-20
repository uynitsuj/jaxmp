"""
Differentiable robot collision model, implemented in JAX.
Supports World- and self- collision detection (returns signed distance).
"""

from __future__ import annotations
from typing import Optional, cast

from jaxmp.coll._coll_mjx_fnc import collide
from jaxmp.kinematics import JaxKinTree
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

from jaxmp.coll._coll_mjx_types import Capsule, Convex


@jdc.pytree_dataclass
class RobotColl:
    """Collision model for a robot, which can be put into different configurations."""
    num_coll_links: jdc.Static[int]

    coll_links: jdc.Static[tuple[Convex]]

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
        coll_links = list[Convex]()
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

            coll_links.append(coll_link)
            idx_parent_joint.append(joint_idx)
            link_names.append(curr_link)

            if ignore_immediate_parent:
                self_coll_ignore.append((joint.parent, joint.child))

        assert len(coll_links) > 0, "No collision links found in URDF."
        logger.info("Found {} collision bodies", len(coll_links))

        idx_parent_joint = jnp.array(idx_parent_joint)
        link_names = tuple[str](link_names)

        self_coll_matrix = RobotColl.create_self_coll_matrix(
            urdf, link_names, self_coll_ignore
        )
        assert self_coll_matrix.shape == (len(link_names), len(link_names))

        return RobotColl(
            num_coll_links=len(coll_links),
            coll_links=coll_links,
            coll_link_names=link_names,
            link_joint_idx=idx_parent_joint,
            self_coll_matrix=self_coll_matrix,
        )

    @staticmethod
    def _get_coll_link(urdf: yourdfpy.URDF, curr_link: str) -> Optional[Convex]:
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

        if not coll_link_mesh.is_convex:
            coll_link_mesh = coll_link_mesh.convex_hull
        
        import scipy
        hull = scipy.spatial.ConvexHull(coll_link_mesh.vertices, qhull_options="TA18")
        hull_verts = coll_link_mesh.vertices[hull.vertices]
        hull_faces = onp.searchsorted(hull.vertices, hull.simplices.flatten()).reshape(-1, 3)
        _coll_link_mesh = trimesh.Trimesh(vertices=hull_verts, faces=hull_faces)
        _coll_link_mesh.fix_normals()
        coll_link = Convex.from_convex_mesh(_coll_link_mesh)

        # coll_link = Cylinder.from_min_cylinder(coll_link_mesh)
        # coll_link = Capsule.from_min_cylinder(coll_link_mesh)

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

    @jdc.jit
    def collide_self(self, kin: JaxKinTree, cfg: jax.Array) -> jax.Array:
        Ts_joint_world = kin.forward_kinematics(cfg)[..., self.link_joint_idx, :]
        colls = [
            self.coll_links[i].transform(jaxlie.SE3(Ts_joint_world[..., i, :]))
            for i in range(self.num_coll_links)
        ]

        batch_axes = Ts_joint_world.shape[:-2]
        sdf = jnp.zeros((*batch_axes, self.num_coll_links, self.num_coll_links))

        for i in range(self.num_coll_links - 1):
            # for j in range(i + 1, self.num_coll_links):
            j = i + 1
            _sdf = collide(colls[i], colls[j])[0]
            _sdf = _sdf.reshape((*batch_axes, -1)).sum(axis=-1)
            sdf = sdf.at[..., i, j].set(_sdf)

        return sdf

    # def fun(self, coll_i, coll_j):
    #     _sdf = collide(coll_i,coll_j)[0]
    #     return _sdf
        