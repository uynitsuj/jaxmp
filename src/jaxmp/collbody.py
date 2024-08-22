"""
Collision bodies for differentiable collision detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Generic

from abc import ABC, abstractmethod
import trimesh
import trimesh.bounds
import numpy as onp

import jax
from jax import Array
import jax.numpy as jnp

from jaxtyping import Float
import jax_dataclasses as jdc
import jaxlie


if TYPE_CHECKING:
    import trimesh.nsphere

T = TypeVar("T")

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
        # TODO(cmk) optimize the capsule height.
        if height - 2 * radius > 0:
            height = height - 2 * radius
        assert height > 0

        radius = onp.array([radius])
        height = onp.array([height])
        tf = jaxlie.SE3.from_matrix(tf_mat[None, ...])

        return CapsuleColl(
            radii=jnp.array(radius),
            heights=jnp.array(height),
            tf=tf,
        )

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


def sdf_to_colldist(
    _dist: Float[Array, "n_dists"],
    eta: float = 0.05,
) -> Float[Array, "n_dists"]:
    """
    Convert a signed distance field to a collision distance field,
    based on https://arxiv.org/pdf/2310.17274#page=7.39.

    Args: 
        dist: Signed distance field values, where sdf > 0 means collision (n_dists,).
        eta: Distance threshold for the SDF.

    Returns:
        Collision distance field values (n_dists,).
    """
    _dist = jnp.maximum(_dist, -eta)
    _dist = jnp.where(
        _dist > 0,
        _dist + 0.5 * eta,
        0.5 / eta * (_dist + eta)**2
    )
    _dist = jnp.maximum(_dist, 0.0)
    return _dist

def sdf_collbody(
    coll_0: CollBody,
    coll_1: CollBody,
) -> Float[Array, "coll_0 coll_1"]:
    """Distance between two collision bodies. sdf > 0 means collision."""
    # Sphere-Sphere collision
    if isinstance(coll_0, SphereColl) and isinstance(coll_1, SphereColl):
        return _dist_sph_sph(coll_0, coll_1)

    # Sphere-HalfSpace collision
    elif isinstance(coll_0, SphereColl) and isinstance(coll_1, HalfSpaceColl):
        return _dist_sph_halfspace(coll_0, coll_1)
    elif isinstance(coll_1, SphereColl) and isinstance(coll_0, HalfSpaceColl):
        return _dist_sph_halfspace(coll_1, coll_0).T

    # Sphere-Plane collision
    elif isinstance(coll_0, SphereColl) and isinstance(coll_1, PlaneColl):
        return _dist_sph_plane(coll_0, coll_1)
    elif isinstance(coll_1, SphereColl) and isinstance(coll_0, PlaneColl):
        return _dist_sph_plane(coll_1, coll_0).T

    # Capsule-Capsule collision
    elif isinstance(coll_0, CapsuleColl) and isinstance(coll_1, CapsuleColl):
        return _dist_cap_cap(coll_0, coll_1)

    # Sphere-Capsule collision
    elif isinstance(coll_0, CapsuleColl) and isinstance(coll_1, SphereColl):
        return _dist_sph_cap(coll_0, coll_1)
    elif isinstance(coll_1, CapsuleColl) and isinstance(coll_0, SphereColl):
        return _dist_sph_cap(coll_1, coll_0).T

    # Capsule-HalfSpace collision
    elif isinstance(coll_0, CapsuleColl) and isinstance(coll_1, HalfSpaceColl):
        return _dist_cap_halfspace(coll_0, coll_1)
    elif isinstance(coll_1, CapsuleColl) and isinstance(coll_0, HalfSpaceColl):
        return _dist_cap_halfspace(coll_1, coll_0).T

    # Capsule-Plane collision
    elif isinstance(coll_0, CapsuleColl) and isinstance(coll_1, PlaneColl):
        return _dist_cap_plane(coll_0, coll_1)
    elif isinstance(coll_1, CapsuleColl) and isinstance(coll_0, PlaneColl):
        return _dist_cap_plane(coll_1, coll_0).T

    else:
        raise NotImplementedError(f"Collision between {type(coll_0)} and {type(coll_1)} not implemented.")

def _dist_sph_sph(
    sphere: SphereColl,
    other: SphereColl,
) -> Float[Array, "sph_0 sph_1"]:
    """Return distance between two `Spheres`. sdf > 0 means collision."""
    n_spheres_sph_0 = sphere.centers.shape[0]
    n_spheres_sph_1 = other.centers.shape[0]

    # Without the `+ 1e-7`, the jacobian becomes unstable / returns NaNs.
    _dist = jnp.linalg.norm(
        sphere.centers[:, None] - other.centers[None] + 1e-7, axis=-1
    )
    sdf = (sphere.radii[:, None] + other.radii[None]) - _dist

    assert sdf.shape == (n_spheres_sph_0, n_spheres_sph_1)
    return sdf

def _dist_sph_plane(
    sphere: SphereColl,
    plane: PlaneColl,
) -> Float[Array, "sph_0"]:
    """Return the distance between a `Plane` and a point. sdf > 0 means collision."""
    n_spheres = sphere.centers.shape[0]
    _dist = jnp.abs(jnp.dot(sphere.centers - plane.point, plane.normal))
    sdf = (sphere.radii - _dist)[:, None]
    assert sdf.shape == (n_spheres, 1)
    return sdf

def _dist_sph_halfspace(
    sphere: SphereColl,
    plane: HalfSpaceColl,
) -> Float[Array, "sph_0"]:
    """Return the distance between a halfspace and a point. sdf > 0 means collision."""
    n_spheres = sphere.centers.shape[0]
    _dist = jnp.dot(sphere.centers - plane.point, plane.normal)
    sdf = (sphere.radii - _dist)[:, None]
    assert sdf.shape == (n_spheres, 1)
    return sdf

def _dist_cap_cap(
    cap_0: CapsuleColl,
    cap_1: CapsuleColl,
) -> Float[Array, "cap_0 cap_1"]:
    a0, a1 = cap_0.centerline
    b0, b1 = cap_1.centerline
    _dist = CapsuleColl.dist_between_seg(a0, a1, b0, b1)
    sdf = (cap_0.radii[:, None] + cap_1.radii[None, :]) - _dist
    assert sdf.shape == (cap_0.radii.shape[0], cap_1.radii.shape[0])
    return sdf

def _dist_sph_cap(
    cap: CapsuleColl,
    sph: SphereColl,
) -> Float[Array, "cap sph"]:
    a0, a1 = cap.centerline
    _dist = CapsuleColl.dist_between_seg(a0, a1, sph.centers, sph.centers)
    sdf = (cap.radii[:, None] + sph.radii[None, :]) - _dist
    assert sdf.shape == (cap.radii.shape[0], sph.radii.shape[0])
    return sdf

def _dist_cap_plane(
    cap: CapsuleColl,
    plane: PlaneColl,
) -> Float[Array, "cap 1"]:
    a0, a1 = cap.centerline
    t = jnp.dot(plane.point - a0, plane.normal) / (jnp.dot(a1 - a0, plane.normal) + 1e-7)
    t = jnp.clip(t, 0.0, 1.0)
    point = a0 + t[:, None] * (a1 - a0)
    _dist = jnp.abs(jnp.dot(point - plane.point, plane.normal))
    sdf = (cap.radii - _dist)[:, None]
    assert sdf.shape == (cap.radii.shape[0], 1)
    return sdf

def _dist_cap_halfspace(
    cap: CapsuleColl,
    plane: HalfSpaceColl,
) -> Float[Array, "cap"]:
    a0, a1 = cap.centerline
    t = jnp.dot(plane.point - a0, plane.normal) / (jnp.dot(a1 - a0, plane.normal) + 1e-7)
    t = jnp.clip(t, 0.0, 1.0)
    point = a0 + t[:, None] * (a1 - a0)
    _dist = jnp.dot(point - plane.point, plane.normal)
    sdf = (cap.radii - _dist)[:, None]
    assert sdf.shape == (cap.radii.shape[0], 1)
    return sdf