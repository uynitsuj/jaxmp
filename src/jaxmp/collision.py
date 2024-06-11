from __future__ import annotations

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

@jdc.pytree_dataclass
class SphereCollision:
    """A differentiable collision model, parameterized as spheres. For point collisions, radii can be zero."""
    centers: Float[Array, "spheres 3"]
    radii: Float[Array, "spheres"]

    @jax.jit
    def collides_points(
        self,
        points: Float[Array, "points 3"]
    ) -> Float[Array, "spheres"]:
        """Check if points are inside the spheres. sdf > 0 means collision."""
        num_points = points.shape[0]
        num_spheres = self.centers.shape[0]

        assert self.centers.shape == (num_points, 3) and self.radii.shape == (num_points,)
        assert points.shape == (num_points, 3)

        dist = jnp.sum((points[None] - self.centers[:, None]) ** 2, axis=-1)
        assert dist.shape == (num_spheres, num_points)
        sdf = self.radii - jnp.sqrt(dist)

        max_sdf = jnp.max(sdf, axis=-1)
        assert max_sdf.shape == (num_spheres,)

        return max_sdf

    @jax.jit
    def collides_other(
        self,
        other: SphereCollision
    ) -> Float[Array, "spheres"]:
        """Check if collides with another SphereCollision. sdf > 0 means collision."""
        num_spheres_self = self.centers.shape[0]
        num_spheres_other = other.centers.shape[0]

        assert self.centers.shape == (num_spheres_self, 3) and self.radii.shape == (num_spheres_self,)
        assert other.centers.shape == (num_spheres_other, 3) and other.radii.shape == (num_spheres_other,)

        dist = jnp.sum((self.centers[:, None] - other.centers[None]) ** 2, axis=-1)
        assert dist.shape == (num_spheres_self, num_spheres_other)
        sdf = (self.radii[:, None] + other.radii[None]) - jnp.sqrt(dist)

        max_sdf = jnp.max(sdf, axis=-1)
        assert max_sdf.shape == (num_spheres_self,), f"{max_sdf.shape}, expected ({num_spheres_self},)"
        return max_sdf

    @jax.jit
    def intersection_volume(
        self,
        other: SphereCollision
    ) -> Float[Array, "spheres"]:
        """Compute the volume of the intersection of two SphereCollisions."""
        # TODO this doesn't consider the density of the spheres.
        num_spheres_self = self.centers.shape[0]
        num_spheres_other = other.centers.shape[0]

        assert self.centers.shape == (num_spheres_self, 3) and self.radii.shape == (num_spheres_self,)
        assert other.centers.shape == (num_spheres_other, 3) and other.radii.shape == (num_spheres_other,)

        dist = jnp.sqrt(jnp.sum((self.centers[:, None] - other.centers[None]) ** 2, axis=-1))
        assert dist.shape == (num_spheres_self, num_spheres_other)

        # Calculate the volume of intersection of two spheres.
        # -> https://mathworld.wolfram.com/Sphere-SphereIntersection.html
        _r_self = jnp.broadcast_to(self.radii[:, None], (num_spheres_self, num_spheres_other))
        _r_other = jnp.broadcast_to(other.radii, (num_spheres_self, num_spheres_other))
        volume = (
            (jnp.pi / 12 * (dist + 1e-5))
            * jnp.square(_r_self + _r_other - dist)
            * (
                jnp.square(dist)
                + 2 * _r_self * dist
                + 2 * _r_other * dist
                - 3 * jnp.square(_r_self)
                - 3 * jnp.square(_r_other)
                + 6 * _r_self * _r_other
            )
        )
        assert volume.shape == (num_spheres_self, num_spheres_other)
        
        # Volume for non-intersecting spheres is zero.
        volume_capped = jnp.where(dist > _r_self + _r_other, 0.0, volume)
        # Equation doesn't hold if dist < min(r_self, r_other), so cap the volume.
        # -> https://gamedev.stackexchange.com/questions/75756/sphere-sphere-intersection-and-circle-sphere-intersection
        volume_capped_for_min = jnp.where(
            dist < jnp.minimum(_r_self, _r_other),
            4/3 * jnp.pi * jnp.minimum(_r_self, _r_other)**3,
            volume_capped
        )

        # Approximate volume overlap, via summing the volume of intersection with all other spheres.
        intersection_vol_per_self_sphere = jnp.sum(volume_capped_for_min, axis=-1)
        assert intersection_vol_per_self_sphere.shape == (num_spheres_self,)

        return intersection_vol_per_self_sphere


def test_JaxSphereCollision():
    centers_1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii_1 = jnp.array([0.5, 0.5])
    sphere_collision_1 = SphereCollision(centers_1, radii_1)

    centers_2 = jnp.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    radii = jnp.array([0.5, 0.5])
    sphere_collision_2 = SphereCollision(centers_2, radii)

    assert jnp.all(sphere_collision_1.collides_other(sphere_collision_2) <= 0)
    assert jnp.all(sphere_collision_1.collides_other(sphere_collision_1) > 0)

    assert jnp.all(sphere_collision_1.intersection_volume(sphere_collision_2) == 0.0)
    assert jnp.all(
        sphere_collision_1.intersection_volume(sphere_collision_1)
        == 4.0 / 3.0 * jnp.pi * 0.5**3
    )


if __name__ == "__main__":
    centers = jax.random.uniform(jax.random.PRNGKey(0), (10, 3), minval=-2, maxval=2)
    radii = jax.random.uniform(jax.random.PRNGKey(1), (10,), minval=0.1, maxval=0.5)

    sphere_collision = SphereCollision(centers, radii)

    import viser
    server = viser.ViserServer()
    tf_handle = server.scene.add_transform_controls(
        "tf", scale=0.5, disable_rotations=True, disable_sliders=True
    )
    contains_gui = server.gui.add_checkbox("contains", initial_value=False, disabled=True)

    import trimesh.creation
    spheres_list = []
    for i, (center, radius) in enumerate(zip(centers, radii)):
        sphere_mesh = trimesh.creation.icosphere(radius=radius)
        spheres_list.append(server.scene.add_mesh_trimesh(f"sphere_{i}", sphere_mesh, position=center))

    @tf_handle.on_update
    def _(_):
        position = jnp.array(tf_handle.position.reshape(-1, 3))
        contains_gui.value = onp.array(
            jnp.any(sphere_collision.collides_points(position) > 0)
        ).item()

    test_JaxSphereCollision()
    breakpoint()
