"""
Collision bodies for differentiable collision detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast, TYPE_CHECKING, TypeVar, Generic, Optional

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


if TYPE_CHECKING:
    import trimesh.nsphere

T = TypeVar("T", bound="CollBody")

class CollBody(ABC, Generic[T]):
    """Abstract base class for collision bodies."""

    @staticmethod
    @abstractmethod
    def from_trimesh(mesh) -> T:
        """Create a collision body from a Trimesh object."""
        raise NotImplementedError

    @abstractmethod
    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the collision body to a Trimesh object."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, tf: jaxlie.SE3) -> T:
        """Transform the collision body by a SE3 transformation."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of collision bodies."""
        raise NotImplementedError

    @abstractmethod
    def __add__(self: T, other: T) -> T:
        """Combine two collision bodies."""
        raise NotImplementedError

    def __radd__(self: T, other: T) -> T:
        """Combine two collision bodies."""
        if other == 0:
            return self
        assert isinstance(other, type(self))
        return type(self).__add__(self, other)  # type: ignore


@jdc.pytree_dataclass
class SphereColl(CollBody):
    """Differentiable collision body, composed of a set of spheres with varying radii."""
    centers: Float[Array, "sphere 3"]
    radii: Float[Array, "sphere"]

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh) -> SphereColl:
        """Create `Spheres` based on the mesh's vertices."""
        return SphereColl.from_min_ball(mesh)

    @staticmethod
    def from_min_ball(mesh: trimesh.Trimesh) -> SphereColl:
        """
        Create `Spheres` based on the mesh's minimum bounding sphere (n_spheres=1).
        Uses trimesh's `minimum_nsphere`.

        Args:
            mesh: A trimesh object.
        Returns:
            Spheres: A collision body, composed of a single sphere.
        """
        centers, radii = trimesh.nsphere.minimum_nsphere(mesh)

        centers = centers[None]
        radii = radii[None]
        assert len(centers.shape) == 2 and len(radii.shape) == 1
        assert centers.shape[0] == radii.shape[0]

        return SphereColl(
            jnp.array(centers),
            jnp.array(radii)
        )

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the spheres to a Trimesh object."""
        spheres = [trimesh.creation.icosphere(radius=radius) for radius in self.radii]
        for sphere_idx, sphere in enumerate(spheres):
            sphere.vertices += onp.array(self.centers[sphere_idx])
        return sum(spheres, trimesh.Trimesh())

    def transform(self, tf: jaxlie.SE3) -> SphereColl:
        """Transform the spheres by a SE3 transformation."""
        centers = tf.apply(self.centers)
        return SphereColl(centers, self.radii)

    def __add__(self, other: SphereColl) -> SphereColl:
        return SphereColl(
            jnp.concatenate([self.centers, other.centers], axis=0),
            jnp.concatenate([self.radii, other.radii], axis=0),
        )

    def __len__(self) -> int:
        return self.centers.shape[0]


@jdc.pytree_dataclass
class CapsuleColl(CollBody):
    """Differentiable collision body, composed of capsules.
    Assume that the capsules are aligned with the z-axis.
    """
    radii: Float[Array, "capsule"]
    heights: Float[Array, "capsule"]  # center-to-center distance between two spheres
    tf: Float[jaxlie.SE3, "capsule"]  # Capsule-to-world transform. Height is along z-axis.

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh) -> CapsuleColl:
        """Create `Capsules` based on the mesh's vertices."""
        return CapsuleColl.from_min_cylinder(mesh)

    @staticmethod
    def from_min_cylinder(mesh: trimesh.Trimesh) -> CapsuleColl:
        """
        Approximate a minimum bounding capsule for a mesh using a minimum cylinder.
        """
        results = trimesh.bounds.minimum_cylinder(mesh)

        assert 'transform' in results
        assert 'radius' in results
        assert 'height' in results

        tf_mat = onp.array(results['transform'])
        radius = results['radius']
        height = results['height']

        # If height is tall enough, we subtract the radius from the height.
        # Otherwise, we cap the two ends.
        if height - 2 * radius > 0:
            height = height - 2 * radius
        else:
            height = 0

        radius = onp.array([radius])
        height = onp.array([height])
        tf = jaxlie.SE3.from_matrix(tf_mat[None, ...])

        cap = CapsuleColl(
            radii=jnp.array(radius),
            heights=jnp.array(height),
            tf=tf,
        )

        return cap

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the capsule to a Trimesh object."""
        capsule_list = []
        for i in range(self.radii.shape[0]):
            capsule = trimesh.creation.capsule(
                self.heights[i].item(),
                self.radii[i].item(),
                transform=jaxlie.SE3(self.tf.wxyz_xyz[i]).as_matrix()
            )
            capsule_list.append(capsule)
        return sum(capsule_list, trimesh.Trimesh())

    def transform(self, tf: jaxlie.SE3) -> CapsuleColl:
        """Transform the capsule by a SE3 transformation."""
        _tf = tf @ self.tf
        return CapsuleColl(self.radii, self.heights, _tf)

    @property
    def centerline(self):
        """Return the centerline of the capsule, as the endpoints."""
        tf_mat = self.tf.as_matrix()
        return (
            tf_mat[..., :3, 3] - 0.5 * self.heights[:, None] * tf_mat[..., :3, 2],
            tf_mat[..., :3, 3] + 0.5 * self.heights[:, None] * tf_mat[..., :3, 2]
        )

    @staticmethod
    def dist_between_seg(
        a0: Float[Array, "capsule_1 3"],
        a1: Float[Array, "capsule_1 3"],
        b0: Float[Array, "capsule_2 3"],
        b1: Float[Array, "capsule_2 3"],
        eta: float = 1e-6,
    ) -> Float[Array, "capsule_1 capsule_2"]:
        """Return the distance between two line segments ((a0, a1), (b0, b1)).
        Taken from https://stackoverflow.com/a/67102941, and ported to JAX.
        """

        # Vector-vector distance.
        def _dist_between_seg(
            _a0: Float[Array, "3"],
            _a1: Float[Array, "3"],
            _b0: Float[Array, "3"],
            _b1: Float[Array, "3"],
        ):
            r = _b0 - _a0
            u = _a1 - _a0
            v = _b1 - _b0

            ru = r @ u.T
            rv = r @ v.T
            uu = u @ u.T
            uv = u @ v.T
            vv = v @ v.T

            det = uu * vv - uv**2
            s = jnp.where(
                det < eta * uu * vv,
                jnp.clip(ru / (uu + eta), 0.0, 1.0),
                jnp.clip((ru * vv - rv * uv) / (det + eta), 0.0, 1.0)
            )
            t = jnp.where(
                det < eta * uu * vv,
                jnp.zeros_like(s),
                jnp.clip((ru * uv - rv * uu) / (det + eta), 0.0, 1.0)
            )

            S = jnp.clip((t * uv + ru) / (uu + eta), 0.0, 1.0)
            T = jnp.clip((s * uv - rv) / (vv + eta), 0.0, 1.0)

            A = _a0 + u * S
            B = _b0 + v * T

            _dist = jnp.linalg.norm(A - B + eta, axis=-1)
            return _dist

        _dist = jax.vmap(
            jax.vmap(_dist_between_seg, (None, None, 0, 0)),
            (0, 0, None, None)
        )(a0, a1, b0, b1)
        assert _dist.shape == (a0.shape[0], b0.shape[0]), _dist.shape
        return _dist

    def __add__(self, other: CapsuleColl) -> CapsuleColl:
        return CapsuleColl(
            jnp.concatenate([self.radii, other.radii], axis=0),
            jnp.concatenate([self.heights, other.heights], axis=0),
            jaxlie.SE3(jnp.concatenate([self.tf.wxyz_xyz, other.tf.wxyz_xyz], axis=0))
        )

    def __len__(self) -> int:
        return self.radii.shape[0]


@jdc.pytree_dataclass
class PlaneColl(CollBody):
    """Differentiable collision body for a plane, defined by a point and a surface normal."""
    point: Float[Array, "3"]
    normal: Float[Array, "3"]

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh) -> PlaneColl:
        """Create a `Plane` from a Trimesh object."""
        raise NotImplementedError("Creating a plane from a Trimesh object is not supported.")

    @staticmethod
    def from_point_normal(point: Float[Array, "3"], normal: Float[Array, "3"]) -> PlaneColl:
        """Create a `Plane` from a point and a normal."""
        assert point.shape == (3,) and normal.shape == (3,)
        assert jnp.isclose(jnp.linalg.norm(normal), 1)
        return PlaneColl(point, normal)

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the plane to a Trimesh object."""
        plane = trimesh.creation.box((3.0, 3.0, 0.001))

        # Find some orthogonal axis to the normal.
        y_axis = jnp.array([
            1.0, 0.0, -self.normal[0] / (self.normal[2] + 1e-7)
        ])
        x_axis = jnp.cross(y_axis, self.normal)
        rotmat = jnp.array([x_axis, y_axis, self.normal]).T

        mat = jnp.eye(4)
        mat = mat.at[:3, :3].set(rotmat)
        mat = mat.at[:3, 3].set(self.point)

        plane.vertices = trimesh.transform_points(
            plane.vertices, mat
        )
        return plane

    def transform(self, tf: jaxlie.SE3) -> PlaneColl:
        """Transform the plane by a SE3 transformation."""
        point = tf.apply(self.point)
        normal = tf.rotation().apply(self.normal)
        return PlaneColl(point, normal)

    def __add__(self, other: PlaneColl) -> PlaneColl:
        raise NotImplementedError("Combining two planes is not supported.")

    def __len__(self) -> int:
        return 1


@jdc.pytree_dataclass
class HalfSpaceColl(CollBody):
    """Differentiable collision body for a halfspace, defined by a point and a surface normal."""
    point: Float[Array, "3"]
    normal: Float[Array, "3"]

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh) -> HalfSpaceColl:
        """Create a `Plane` from a Trimesh object."""
        raise NotImplementedError("Creating a plane from a Trimesh object is not supported.")

    @staticmethod
    def from_point_normal(point: Float[Array, "3"], normal: Float[Array, "3"]) -> HalfSpaceColl:
        """Create a `Plane` from a point and a normal."""
        assert point.shape == (3,) and normal.shape == (3,)
        assert jnp.isclose(jnp.linalg.norm(normal), 1)
        return HalfSpaceColl(point, normal)

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the plane to a Trimesh object."""
        plane = trimesh.creation.box((3.0, 3.0, 0.001))

        # Find some orthogonal axis to the normal.
        y_axis = jnp.array([
            1.0, 0.0, -self.normal[0] / (self.normal[2] + 1e-7)
        ])
        x_axis = jnp.cross(y_axis, self.normal)
        rotmat = jnp.array([x_axis, y_axis, self.normal]).T

        mat = jnp.eye(4)
        mat = mat.at[:3, :3].set(rotmat)
        mat = mat.at[:3, 3].set(self.point)

        plane.vertices = trimesh.transform_points(
            plane.vertices, mat
        )
        return plane

    def transform(self, tf: jaxlie.SE3) -> HalfSpaceColl:
        """Transform the plane by a SE3 transformation."""
        point = tf.apply(self.point)
        normal = tf.rotation().apply(self.normal)
        return HalfSpaceColl(point, normal)

    def __add__(self, other: HalfSpaceColl) -> HalfSpaceColl:
        raise NotImplementedError("Combining two planes is not supported.")

    def __len__(self) -> int:
        return 1


@jdc.pytree_dataclass
class RobotColl(CapsuleColl):
    """Collision model for a robot, which can be put into different configurations."""

    link_names: jdc.Static[tuple[str]]
    """Names of the links in the robot, length `links`."""

    _idx_parent_joint: Int[Array, "link"]
    """Index of the parent joint for each link."""

    self_coll_matrix: Int[Array, "link link"]
    """Collision matrix, where `coll_matrix[i, j] == 1`
    if we account for the collision between collbodies `i` and `j`.
    Else, `coll_matrix[i, j] == 0`."""

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh):
        raise NotImplementedError("RobotColl must be generated from a URDF.")

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
        list_coll_link = list[CapsuleColl]()
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

            assert len(coll_link) == 1, "Only one collision primitive per link supported."
            list_coll_link.append(coll_link)
            idx_parent_joint.append(joint_idx)
            link_names.append(curr_link)

            if ignore_immediate_parent:
                self_coll_ignore.append((joint.parent, joint.child))

        assert len(list_coll_link) > 0, "No collision links found in URDF."
        logger.info("Found {} collision bodies", len(list_coll_link))

        coll_links = cast(CapsuleColl, sum(list_coll_link))
        idx_parent_joint = jnp.array(idx_parent_joint)
        link_names = tuple[str](link_names)

        self_coll_matrix = RobotColl.create_self_coll_matrix(
            urdf, link_names, self_coll_ignore
        )
        assert self_coll_matrix.shape == (len(link_names), len(link_names))

        return RobotColl(
            radii=coll_links.radii,
            heights=coll_links.heights,
            tf=coll_links.tf,
            link_names=link_names,
            _idx_parent_joint=idx_parent_joint,
            self_coll_matrix=self_coll_matrix,
        )

    @staticmethod
    def _get_coll_link(urdf: yourdfpy.URDF, curr_link: str) -> Optional[CapsuleColl]:
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
        coll_link = CapsuleColl.from_trimesh(coll_link_mesh)
        return coll_link

    def coll_weight(
        self,
        weights: dict[str, float],
        default: float = 1.0
    ) -> Float[Array, "links"]:
        """Get the collision weight for each sphere."""
        coll_weights = jnp.full((len(self),), default)
        for name, weight in weights.items():
            idx = self.link_names.index(name)
            coll_weights = coll_weights.at[idx].set(weight)
        return jnp.array(coll_weights)

    def transform(self, tf: jaxlie.SE3) -> RobotColl:
        """Re-configure the robot using the robot joint, derived w/ forward kinematics."""
        Ts_world_joint = tf.wxyz_xyz  # pylint: disable=invalid-name
        transformed_caps = super(RobotColl, self).transform(
            jaxlie.SE3(Ts_world_joint[..., self._idx_parent_joint, :])
        )
        return RobotColl(
            radii=transformed_caps.radii,
            heights=transformed_caps.heights,
            tf=transformed_caps.tf,
            link_names=self.link_names,
            _idx_parent_joint=self._idx_parent_joint,
            self_coll_matrix=self.self_coll_matrix,
        )

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
            [
                [check_coll(i, j) for j in range(n_links)]
                for i in range(n_links)
            ]
        )
        assert coll_mat.shape == (n_links, n_links)
        return coll_mat
