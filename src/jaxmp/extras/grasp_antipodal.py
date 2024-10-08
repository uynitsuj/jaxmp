"""
Antipodal grasp sampling.
"""

from __future__ import annotations

from typing import cast
import numpy as onp
from jax import Array
import jax.numpy as jnp
from jaxtyping import Float
import jax_dataclasses as jdc
import jaxlie

import trimesh
import trimesh.sample


@jdc.pytree_dataclass
class Grasp:
    centers: Float[Array, "*batch 3"]
    axes: Float[Array, "*batch 3"]

    @staticmethod
    def sample_antipodal(
        mesh: trimesh.Trimesh,
        num_samples=1000,
        max_width=float("inf"),
        max_angle_deviation=onp.pi / 4,
    ) -> Grasp:
        """
        Sample antipodal grasps from a given mesh, considering a maximum grasp width.
        """
        grasp_centers, grasp_axes = [], []

        sampled_points, sampled_face_indices = cast(
            tuple[onp.ndarray, onp.ndarray],
            trimesh.sample.sample_surface(mesh, num_samples),
        )
        min_dot_product = onp.cos(max_angle_deviation)

        for sample_idx in range(num_samples):
            p1 = sampled_points[sample_idx]
            n1 = mesh.face_normals[sampled_face_indices[sample_idx]]

            # Raycast!
            locations, _, index_tri = mesh.ray.intersects_location(
                p1.reshape(1, 3), -n1.reshape(1, 3), multiple_hits=False
            )

            if len(locations) == 0:
                continue

            p2 = locations[0]
            n2 = mesh.face_normals[index_tri[0]]

            # Check grasp width.
            grasp_width = onp.linalg.norm(p2 - p1)
            if grasp_width > max_width:
                continue

            # Check for antipodal condition.
            grasp_center = (p1 + p2) / 2
            grasp_direction = p1 - p2
            grasp_direction /= onp.linalg.norm(grasp_direction)

            if (
                onp.dot(n1, grasp_direction) > min_dot_product
                and onp.dot(n2, -grasp_direction) > min_dot_product
            ):
                grasp_centers.append(grasp_center)
                grasp_axes.append(grasp_direction)

        return Grasp(
            centers=jnp.array(grasp_centers),
            axes=jnp.array(grasp_axes),
        )

    def to_trimesh(
        self, axes_radius: float = 0.0002, axes_height: float = 0.08
    ) -> trimesh.Trimesh:
        """
        Convert the grasp to a trimesh object.
        """
        # Create "base" grasp visualization, centered at origin + lined up with x-axis.
        transform = onp.eye(4)
        rotation = trimesh.transformations.rotation_matrix(onp.pi / 2, [0, 1, 0])
        transform[:3, :3] = rotation[:3, :3]
        mesh = trimesh.creation.cylinder(
            radius=axes_radius, height=axes_height, transform=transform
        )
        mesh.visual.vertex_colors = [150, 150, 255, 255]  # type: ignore[attr-defined]

        meshes = []
        grasp_transforms = self.to_se3().as_matrix()
        for idx in range(self.centers.shape[0]):
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(grasp_transforms[idx])
            meshes.append(mesh_copy)

        return sum(meshes, trimesh.Trimesh())

    def to_se3(self) -> jaxlie.SE3:
        x_axes = self.axes

        delta = x_axes + (
            jnp.sign(x_axes[..., 0] + 1e-6)[..., None]
            * jnp.roll(x_axes, shift=1, axis=-1)
        )
        y_axes = jnp.cross(x_axes, delta)
        y_axes = y_axes / (jnp.linalg.norm(y_axes, axis=-1, keepdims=True) + 1e-6)
        assert jnp.isclose(x_axes, y_axes).all(axis=-1).sum() == 0

        z_axes = jnp.cross(x_axes, y_axes)

        rotmat = jnp.stack([x_axes, y_axes, z_axes], axis=-1)
        assert jnp.isnan(rotmat).sum() == 0

        # Use the axis-angle representation to create the rotation matrix.
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(rotmat), self.centers
        )
