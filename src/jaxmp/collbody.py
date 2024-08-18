"""
Collision bodies for differentiable collision detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import trimesh
import numpy as onp

from jaxtyping import Float
import jax_dataclasses as jdc

from jax import Array
import jax.numpy as jnp

if TYPE_CHECKING:
    import trimesh.nsphere


@jdc.pytree_dataclass
class SphereColl:
    """Differentiable collision body, composed of a set of spheres with varying radii."""
    centers: Float[Array, "sphere 3"]
    radii: Float[Array, "sphere"]

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

    def dist(
        self,
        other: SphereColl,
    ) -> Float[Array, "sph_0 sph_1"]:
        """Return distance between two `Spheres`. sdf > 0 means collision."""
        n_spheres_sph_0 = self.centers.shape[0]
        n_spheres_sph_1 = other.centers.shape[0]

        # Without the `+ 1e-7`, the jacobian becomes unstable / returns NaNs.
        dist = jnp.linalg.norm(
            self.centers[:, None] - other.centers[None] + 1e-7, axis=-1
        )
        sdf = (self.radii[:, None] + other.radii[None]) - dist

        assert sdf.shape == (n_spheres_sph_0, n_spheres_sph_1)
        return sdf

    def dist_to_plane(
        self,
        plane: PlaneColl,
    ) -> Float[Array, "sph_0"]:
        """Return the distance between a `Plane` and a point. sdf > 0 means collision."""
        n_spheres = self.centers.shape[0]
        dist = jnp.linalg.norm(jnp.dot(self.centers - plane.point, plane.normal), keepdims=True)
        sdf = self.radii - dist
        assert sdf.shape == (n_spheres,)
        return sdf

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the spheres to a Trimesh object."""
        spheres = [trimesh.creation.icosphere(radius=radius) for radius in self.radii]
        for sphere_idx, sphere in enumerate(spheres):
            sphere.vertices += onp.array(self.centers[sphere_idx])
        return sum(spheres, trimesh.Trimesh())


@jdc.pytree_dataclass
class PlaneColl:
    """Differentiable collision body for a plane, defined by a point and a surface normal."""
    point: Float[Array, "3"]
    normal: Float[Array, "3"]

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


def sdf_to_colldist(
    dist: Float[Array, "n_dists"],
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
    dist = jnp.maximum(dist, -eta)
    dist = jnp.where(
        dist > 0,
        dist + 0.5 * eta,
        0.5 / eta * (dist + eta)**2
    )
    dist = jnp.maximum(dist, 0.0)
    return dist
