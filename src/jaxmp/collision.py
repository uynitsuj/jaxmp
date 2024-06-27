from __future__ import annotations

import abc
import trimesh

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

class DiffSDF(abc.ABC):
    """
    sdf > 0: inside mesh
    sdf < 0: means outside.
    """
    @abc.abstractmethod
    def d_points(self, points: Float[Array, "points 3"]) -> Float[Array, "points"]:
        raise NotImplementedError

    @abc.abstractmethod
    def d_other(self, other: DiffSDF) -> Float[Array, "1"]:
        raise NotImplementedError


@jdc.pytree_dataclass
class SphereSDF:
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

        assert self.centers.shape == (num_spheres, 3) and self.radii.shape == (num_spheres,)
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
        other: SphereSDF
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
        other: SphereSDF
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


@jdc.pytree_dataclass
class MeshSDF(DiffSDF):
    vertices: Float[Array, "vert 3"]
    faces: Float[Array, "face 3 3"]  # face, num_points (trimesh), 3 (3D)
    face_normals: Float[Array, "face 3"]
    face_areas: Float[Array, "face 1"]

    @staticmethod
    def from_trimesh(mesh: trimesh.Trimesh) -> MeshSDF:
        assert mesh.is_watertight

        # Track faces with the vertex locations.
        # Combine the face + vertices into a single tensor.
        vertices = jnp.array(mesh.vertices)
        faces = jnp.array(mesh.faces)
        num_faces = faces.shape[0]
        face_with_points = jax.vmap(lambda face: vertices[face], in_axes=-2)(faces)
        assert face_with_points.shape == (num_faces, 3, 3), face_with_points.shape  # 3 points per face, 3D points.

        face_areas = mesh.area_faces
        face_areas = face_areas[:, None]
        assert face_areas.shape == (num_faces, 1)

        return MeshSDF(
            vertices=vertices,
            faces=face_with_points,
            face_normals=jnp.array(mesh.face_normals),
            face_areas=jnp.array(face_areas)
        )

    def d_points(self, points: Float[Array, "points 3"]) -> Float[Array, "points"]:
        """Calculate SDF of point(s) with respect to mesh."""
        n_points = points.shape[0]
        n_faces = self.faces.shape[0]

        def d_points_fun(i: int, d_faces_points: Array) -> Array:
            d_face_points = self._d_point_tris(
                points[i],
            )
            min_dist_idx = jnp.argmin(jnp.abs(d_face_points))
            min_dist = d_face_points[min_dist_idx]
            return d_faces_points.at[i].set(min_dist)

        d_points = jax.lax.fori_loop(
            lower=0, 
            upper=n_points,
            body_fun=d_points_fun,
            init_val=jnp.zeros((n_points,))
        )
        assert d_points.shape == (n_points,)
        return d_points

    @jax.jit
    def _d_point_tris(
        self,
        point: Float[Array, "3"]
    ) -> Float[Array, "face"]:
        n_faces = self.faces.shape[0]

        def d_face_points_fun(i: int, d_faces_points: Array) -> Array:
            face_0 = self.faces[i, 0, :]
            face_1 = self.faces[i, 1, :]
            face_2 = self.faces[i, 2, :]

            d_face_points = self.d_point_tri(
                point,
                face_0,
                face_1,
                face_2,
                normal=self.face_normals[i],
                area=self.face_areas[i]
            )

            return d_faces_points.at[i].set(d_face_points.squeeze())

        d_face_points = jax.lax.fori_loop(
            lower=0, 
            upper=n_faces,
            body_fun=d_face_points_fun,
            init_val = jnp.zeros((n_faces,))
        )
        assert d_face_points.shape == (n_faces,)
        return d_face_points

    @staticmethod
    @jax.jit
    def d_point_tri(
        point: Float[Array, "3"],
        face_0: Float[Array, "3"],
        face_1: Float[Array, "3"],
        face_2: Float[Array, "3"],
        normal: Float[Array, "3"],
        area: Float[Array, "1"]
    ) -> Float[Array, "1"]:
        # Reimplementation of pysdf.
        uwv = MeshSDF.bary(point, face_0, face_1, face_2, normal, area)

        dist = jnp.linalg.norm(
            (
                uwv[0] * face_0 + 
                uwv[1] * face_1 + 
                uwv[2] * face_2 - 
                point
            ),
            axis=-1
        )
        assert isinstance(dist, Array)

        sign = MeshSDF.sign_point_tri(point, face_0, normal)
        assert sign.shape == (1,)

        dist_w_0 = jnp.where(uwv[0] < 0, MeshSDF.d_unsigned_point_lineseg(point, face_1, face_2), dist)
        dist_w_1 = jnp.where(uwv[1] < 0, MeshSDF.d_unsigned_point_lineseg(point, face_0, face_2), dist_w_0)  # type: ignore (complains when jit added)
        dist_w_2 = jnp.where(uwv[2] < 0, MeshSDF.d_unsigned_point_lineseg(point, face_0, face_1), dist_w_1)
        assert dist_w_2.shape == (1,)

        return dist_w_2 * sign

    @staticmethod
    @jax.jit
    def sign_point_tri(
        points: Float[Array, "3"],
        face_0: Float[Array, "3"],
        normal: Float[Array, "3"],
        eps: jdc.Static[float] = 1e-6
    ) -> Float[Array, "1"]:
        # Check if point is on the `normals` direction of the face.
        direction = points - face_0
        is_forward = jnp.dot(direction, normal)[None]
        assert is_forward.shape == (1,)
        sign = jnp.where(
            is_forward >= eps,  # i.e., if outside mesh,
            jnp.full((1,), -1),  # then sdf < 0.
            jnp.full((1,), 1),
        )
        assert sign.shape == (1,)
        return sign

    @staticmethod
    @jax.jit
    def bary(
        points: Float[Array, "3"],
        face_0: Float[Array, "3"],
        face_1: Float[Array, "3"],
        face_2: Float[Array, "3"],
        normal: Float[Array, "3"],
        area: Float[Array, "1"]
    ) -> Float[Array, "3"]:
        # Reimplementation of pysdf.
        area_pbc = jnp.dot(normal, (jnp.cross((face_1 - points), (face_2 - points)))) # / area
        area_pca = jnp.dot(normal, (jnp.cross((face_2 - points), (face_0 - points)))) # / area
        uwv = jnp.stack([area_pbc, area_pca, 1 - area_pbc - area_pca], axis=-1)
        assert uwv.shape == (3,)
        return uwv

    @staticmethod
    @jax.jit
    def d_unsigned_point_lineseg(
        point: Float[Array, "3"],
        a: Float[Array, "3"],
        b: Float[Array, "3"],
    ) -> Float[Array, "1"]:
        # Reimplementation from pysdf.
        ap, ab = point - a, b - a
        t = jnp.multiply(ap, ab) / jnp.linalg.norm(ab)**2
        dist_sq = jnp.linalg.norm(ap - t * ab, axis=-1)[None]
        assert dist_sq.shape == (1,)
        return dist_sq

    def d_other(self, other: DiffSDF) -> Float[Array, "1"]:
        assert isinstance(other, MeshSDF)
        d_points = self.d_points(other.vertices)
        assert len(d_points.shape) == 1
        d_other = jnp.min(d_points, keepdims=True)
        assert d_other.shape == (1,)
        return d_other


def test_JaxSphereSDF():
    centers_1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii_1 = jnp.array([0.5, 0.5])
    sphere_collision_1 = SphereSDF(centers_1, radii_1)

    centers_2 = jnp.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    radii = jnp.array([0.5, 0.5])
    sphere_collision_2 = SphereSDF(centers_2, radii)

    assert jnp.all(sphere_collision_1.collides_other(sphere_collision_2) <= 0)
    assert jnp.all(sphere_collision_1.collides_other(sphere_collision_1) > 0)

    assert jnp.all(sphere_collision_1.intersection_volume(sphere_collision_2) == 0.0)
    assert jnp.all(
        sphere_collision_1.intersection_volume(sphere_collision_1)
        == 4.0 / 3.0 * jnp.pi * 0.5**3
    )


if __name__ == "__main__":
    centers = jax.random.uniform(jax.random.PRNGKey(0), (10, 3), minval=-0.5, maxval=0.5)
    radii = jax.random.uniform(jax.random.PRNGKey(1), (10,), minval=0.1, maxval=0.5)

    sphere_collision = SphereSDF(centers, radii)
    test_JaxSphereSDF()

    import viser
    server = viser.ViserServer()
    tf_handle = server.scene.add_transform_controls("tf")
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

    sph = MeshSDF.from_trimesh(trimesh.creation.icosphere(radius=1, subdivisions=3))
    
    print("sph init")
    key = jax.random.PRNGKey(0)
    points = jax.random.uniform(key=key, shape=(10000, 3))
    points = jnp.array([[0.0, 0.0, 0.0]])

    import time
    start = time.time()
    dist = sph.d_points(points)
    print("Time taken:", time.time() - start, "seconds.")
    print(dist)

    start = time.time()
    dist = sph.d_points(points)
    print("Time taken:", time.time() - start, "seconds.")

    breakpoint()
