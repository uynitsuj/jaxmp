from __future__ import annotations

from typing import Optional, cast
import warnings

import trimesh
import trimesh.bounds
import skeletor as sk

import numpy as onp
import jax_dataclasses as jdc
import jax
from jax import Array
from jaxtyping import Float
import jax.numpy as jnp
import trimesh.nsphere


@jdc.pytree_dataclass
class Spheres:
    n_pts: jdc.Static[int]
    centers: Float[Array, "sphere 3"]
    radii: Float[Array, "sphere"]

    @staticmethod
    def from_voxels(
        mesh: trimesh.Trimesh, n_pts: int = 20, volume: bool = True
    ) -> Spheres:
        """
        Create a sphere-based collision body from a mesh, using voxels.

        Args:
            mesh: The input mesh.
            n_pts: The number of spheres to fit to the mesh.
            volume: If True, the spheres are fitted to the mesh volume. If False, the spheres are fitted to the mesh surface.
        """
        # Heavily based on curobo's mesh-to-spheres file, at:
        # https://github.com/NVlabs/curobo/blob/main/src/curobo/geom/sphere_fit.py.
        # More specifically, we use:
        # 1) VOXEL_VOLUME_SAMPLE_SURFACE (volume=True),
        # 2) SAMPLE_SURFACE (volume=False).

        mesh_extents = mesh.extents
        bb_volume = mesh_extents[0] * mesh_extents[1] * mesh_extents[2]
        if bb_volume <= 0 and volume:
            warnings.warn("Mesh has zero volume. `volume` set to False.")
            volume = False

        # Voxelize the mesh.
        if volume:
            # Calculate the pitch based on the requested number of spheres.
            # From curobo's `get_voxel_pitch`.
            occupancy = 1.0 - ((bb_volume - mesh.volume) / bb_volume)
            pitch = (occupancy * bb_volume / n_pts) ** (1 / 3)

            voxel: trimesh.voxel.VoxelGrid = trimesh.voxel.creation.voxelize(
                mesh, pitch
            )
            voxel.fill()  # fill according to morphology.fill

            voxel_points = voxel.points
            assert isinstance(voxel_points, onp.ndarray)
            voxel_radii = onp.full(voxel.points.shape[0], pitch / 2)

            # Check if inside mesh; if inside mesh, expand them to the surface!
            # checking `dist - voxel_radii > 0` would be more exact.
            pr = trimesh.proximity.ProximityQuery(mesh)
            dist = pr.signed_distance(voxel_points)
            voxel_radii = dist
            idx = dist >= 0.0
            voxel_points = voxel_points[idx]
            voxel_radii = voxel_radii[idx]

            # If more spheres than requested, then remove the spheres
            # furthest away from the mesh surface.
            if voxel_points.shape[0] > n_pts:
                pr = trimesh.proximity.ProximityQuery(mesh)
                dist = pr.signed_distance(voxel_points)
                dist = dist - voxel_radii
                idx = onp.argsort(dist)[:n_pts]
                voxel_points = voxel_points[idx]
                voxel_radii = voxel_radii[idx]

            if voxel_points.shape[0] == 0:
                warnings.warn(
                    "No voxels found inside the mesh. Falling back to surface sampling."
                )
        else:
            voxel_points = onp.zeros((0, 3))
            voxel_radii = onp.zeros((0,))

        # Sample points on the surface.
        surf_samples = trimesh.sample.sample_surface(
            mesh, max(n_pts - voxel_points.shape[0], 0)
        )
        surf_points = surf_samples[0]
        assert isinstance(surf_points, onp.ndarray)
        surf_radii = onp.full((surf_points.shape[0],), 0.005)

        # Create the Spheres object!
        num_points = voxel_points.shape[0] + surf_points.shape[0]
        voxel_points = jnp.array(onp.concatenate((voxel_points, surf_points)))
        voxel_radii = jnp.array(onp.concatenate((voxel_radii, surf_radii)))
        assert voxel_points.shape == (num_points, 3) and voxel_radii.shape == (
            num_points,
        )

        return Spheres(num_points, voxel_points, voxel_radii)

    @staticmethod
    def from_skeleton(
        mesh: trimesh.Trimesh,
        n_pts: int = 20,
    ) -> Spheres:
        """
        (not stable) Create a sphere-based collision body from a mesh, using a skeleton.
        """
        # Create a skeleton from the mesh.
        mesh = cast(trimesh.Trimesh, mesh.subdivide_to_size(0.01))
        mesh = cast(trimesh.Trimesh, sk.pre.fix_mesh(mesh))
        assert isinstance(mesh, trimesh.Trimesh)
        skel = sk.skeletonize.by_wavefront(mesh, waves=1, progress=False)
        vert, edges = jnp.array(skel.vertices), jnp.array(skel.edges)

        # Sample points on the skeleton.
        rand_edge_idx = jax.random.uniform(
            jax.random.PRNGKey(0),
            (10 * n_pts,),
        )
        rand_idx = jax.random.uniform(
            jax.random.PRNGKey(0),
            (10 * n_pts,),
        )
        edge_lengths = jax.vmap(jnp.linalg.norm)(vert[edges[:, 0]] - vert[edges[:, 1]])
        edge_cumsum = jnp.cumsum(edge_lengths)
        edge_cumsum = edge_cumsum / edge_cumsum[-1]

        edge_idx = jnp.searchsorted(edge_cumsum, rand_edge_idx)
        centers = vert[edges[edge_idx, 0]] + rand_idx[:, None] * (
            vert[edges[edge_idx, 1]] - vert[edges[edge_idx, 0]]
        )

        # radii = jnp.full(centers.shape[0], 0.005)
        # ... and create spheres from them!
        pr = trimesh.proximity.ProximityQuery(mesh)
        dist = pr.signed_distance(centers)
        idx = dist > 1e-5  # don't want the spheres to be too small.
        centers = centers[idx]
        radii = dist[idx]

        # Sort the spheres by radius, return the biggest ones.
        idx = jnp.argsort(radii, descending=True)[:n_pts]
        centers = centers[idx]
        radii = radii[idx]

        n_pts = centers.shape[0]
        assert n_pts > 0

        return Spheres(n_pts, jnp.array(centers), jnp.array(radii))

    @staticmethod
    def from_points(
        points: Float[Array, "point 3"],
        radius: Optional[float] = None,
    ) -> Spheres:
        """Create a sphere-based collision body from points."""
        if radius is None:
            radius = 0.005
        radii = jnp.full(points.shape[0], radius)
        return Spheres(points.shape[0], points, radii)

    @staticmethod
    def from_obb(
        mesh: trimesh.Trimesh,
    ) -> Spheres:
        """
        Generate spheres from an oriented bounding box.
        """
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        d3_idx = jnp.argmax(extents)
        d3 = extents[d3_idx]
        d1, d2 = extents[[i for i in range(3) if i != d3_idx]]
        dia = jnp.sqrt(d1**2 + d2**2)
        n_sph = int(jnp.ceil(1 + (d3 / dia)))

        # Create the spheres.
        centers = onp.zeros((n_sph, 3))
        centers[:, d3_idx] = onp.linspace(-d3 / 2, d3 / 2, n_sph)
        centers = trimesh.transform_points(centers, onp.linalg.inv(to_origin))
        centers = jnp.array(centers)

        radii = jnp.full((n_sph,), dia / 2)
        return Spheres(n_sph, centers, radii)

    @staticmethod
    def from_voronoi(
        mesh: trimesh.Trimesh,
    ) -> Spheres:
        """Use trimesh's `minimum_nsphere` to generate spheres."""
        centers, radii = trimesh.nsphere.minimum_nsphere(mesh)

        # If only detected one sphere, reshape to (1, 3) and (1,). 
        if len(centers.shape) == 1:
            centers = centers[None]
            radii = radii[None]

        assert len(centers.shape) == 2 and len(radii.shape) == 1
        assert centers.shape[0] == radii.shape[0]
        return Spheres(
            centers.shape[0],
            jnp.array(centers),
            jnp.array(radii)
        )

    @staticmethod
    def dist(
        sph_0: Spheres,
        sph_1: Spheres,
    ) -> Float[Array, "sph_0 sph_1"]:
        """Return distance to another set of spheres. sdf > 0 means collision."""
        n_spheres_sph_0 = sph_0.centers.shape[0]
        n_spheres_sph_1 = sph_1.centers.shape[0]

        dist = jnp.linalg.norm(
            sph_0.centers[:, None] - sph_1.centers[None] + 1e-7, axis=-1
        )
        sdf = (sph_0.radii[:, None] + sph_1.radii[None]) - dist
        assert sdf.shape == (n_spheres_sph_0, n_spheres_sph_1)
        return sdf

    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert the spheres to a Trimesh object."""
        spheres = [trimesh.creation.icosphere(radius=radius) for radius in self.radii]
        for sphere_idx, sphere in enumerate(spheres):
            sphere.vertices += onp.array(self.centers[sphere_idx])
        return sum(spheres, trimesh.Trimesh())


if __name__ == "__main__":
    mesh = trimesh.creation.box((1, 1, 1))
    # mesh = trimesh.creation.icosphere(radius=1)
    # spheres = Spheres.from_trimesh(mesh, 100)
    # spheres = Spheres.from_skeleton(mesh, 100)
    spheres = Spheres.from_obb(mesh)

    import viser

    server = viser.ViserServer()
    server.scene.add_mesh_trimesh("mesh", mesh)
    server.scene.add_mesh_trimesh("spheres", spheres.to_trimesh())

    breakpoint()
