"""
Small wrapper around mjx collision functions to handle batched geometries,
This avoids dealing with the MJX model and data structures directly...
"""

from __future__ import annotations

import abc
from typing import cast

import jax
import numpy as onp
import jax.numpy as jnp
import jaxlie
from jaxtyping import Float
import jax_dataclasses as jdc

from mujoco.mjx._src.types import ConvexMesh
from mujoco.mjx._src.mesh import _get_face_norm, _get_edge_normals
import trimesh


@jdc.pytree_dataclass
class CollGeom(abc.ABC):
    pos: Float[jax.Array, "*batch 3"]  # Translation.
    mat: Float[jax.Array, "*batch 3 3"]  # SO3.
    size: Float[jax.Array, "*batch 3"]  # Object shape (e.g., radii, height).

    def get_batch_axes(self):
        return self.pos.shape[:-1]

    def broadcast_to(self, *shape):
        with jdc.copy_and_mutate(self, validate=False) as _self:
            _self.pos = jnp.broadcast_to(self.pos, shape + (3,))
            _self.mat = jnp.broadcast_to(self.mat, shape + (3, 3))
            _self.size = jnp.broadcast_to(self.size, shape + (3,))
        return _self

    def reshape(self, *shape):
        with jdc.copy_and_mutate(self, validate=False) as _self:
            _self.pos = self.pos.reshape(shape + (3,))
            _self.mat = self.mat.reshape(shape + (3, 3))
            _self.size = self.size.reshape(shape + (3,))
        return _self
    
    def transform(self, tf: jaxlie.SE3):
        with jdc.copy_and_mutate(self, validate=False) as _self:
            _self.mat = (tf.rotation() @ jaxlie.SO3.from_matrix(self.mat)).as_matrix()
            _self.pos = tf.apply(self.pos)
        return _self

    def to_trimesh(self) -> trimesh.Trimesh:
        _self = self.reshape(-1,)

        meshes = [trimesh.Trimesh()]
        for i in range(_self.get_batch_axes()[0]):
            meshes.append(
                self._create_one_mesh(_self.pos[i], _self.mat[i], _self.size[i])
            )

        return cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))

    @abc.abstractmethod
    def _create_one_mesh(
        self,
        pos: Float[jax.Array, "3"],
        mat: Float[jax.Array, "3 3"],
        size: Float[jax.Array, "3"],
    ):
        raise NotImplementedError


@jdc.pytree_dataclass
class Plane(CollGeom):
    @staticmethod
    def from_point_and_normal(point: jax.Array, normal: jax.Array) -> Plane:
        batch_axes = point.shape[:-1]
        assert point.shape[-1] == 3

        mat = Plane._normal_to_SO3(normal)
        assert mat.shape[:-2] == batch_axes

        size = jnp.zeros(batch_axes + (3,))
        return Plane(pos=point, mat=mat, size=size)

    @staticmethod
    def _normal_to_SO3(normal: jax.Array) -> jax.Array:
        # Align z-axis with normal.
        delta = normal + (
            jnp.sign(normal[..., 0] + 1e-6)[..., None]
            * jnp.roll(normal, shift=1, axis=-1)
        )
        x_axes = jnp.cross(normal, delta)
        x_axes = x_axes / (jnp.linalg.norm(x_axes, axis=-1, keepdims=True) + 1e-6)
        assert jnp.isclose(normal, x_axes).all(axis=-1).sum() == 0
        y_axes = jnp.cross(normal, x_axes)
        return jnp.stack([x_axes, y_axes, normal], axis=-1)

    @staticmethod
    def _create_one_mesh(pos: jax.Array, mat: jax.Array, size: jax.Array):
        plane = trimesh.creation.box(extents=[5, 5, 0.01])
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        plane.vertices = trimesh.transform_points(plane.vertices, tf)
        return plane


@jdc.pytree_dataclass
class Sphere(CollGeom):
    @staticmethod
    def from_center_and_radius(center: jax.Array, radius: jax.Array) -> Sphere:
        batch_axes = center.shape[:-1]
        assert center.shape[-1] == 3

        mat = jaxlie.SO3.identity(batch_axes).as_matrix()

        # Uses sphere.size[0] as the radius.
        assert radius.shape == batch_axes + (1,)

        size = jnp.zeros(batch_axes + (2,))
        size = jnp.concatenate([radius, size], axis=-1)
        return Sphere(pos=center, mat=mat, size=size)

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        sphere = trimesh.creation.icosphere(radius=size[0].item())
        tf = onp.eye(4)
        tf[:3, 3] = pos
        sphere.vertices = trimesh.transform_points(sphere.vertices, tf)
        return sphere


@jdc.pytree_dataclass
class Capsule(CollGeom):
    @staticmethod
    def from_radius_and_height(
        radius: jax.Array, height: jax.Array, transform: jaxlie.SE3
    ) -> Capsule:
        batch_axes = transform.get_batch_axes()
        center = transform.translation()
        mat = transform.rotation().as_matrix()

        mat = jaxlie.SO3.identity(batch_axes).as_matrix()

        # Uses capsule.size[0] as the radius and capsule.size[1] as the height.
        assert radius.shape == batch_axes + (1,)
        assert height.shape == batch_axes + (1,)

        # `plane_capsule` uses offsets (in [segment, -segment]).
        segment = height / 2

        shape = jnp.concatenate([radius, segment, jnp.zeros_like(radius)], axis=-1)
        return Capsule(pos=center, mat=mat, size=shape)

    @staticmethod
    def from_min_cylinder(mesh: trimesh.Trimesh) -> Capsule:
        """
        Approximate a minimum bounding capsule for a mesh using a minimum cylinder.
        """
        import trimesh.bounds

        results = trimesh.bounds.minimum_cylinder(mesh)

        assert "transform" in results
        assert "radius" in results
        assert "height" in results

        tf_mat = results["transform"]
        radius = results["radius"]
        height = results["height"]
        tf = jaxlie.SE3.from_matrix(tf_mat)

        cap = Capsule.from_radius_and_height(
            radius=jnp.array([radius]),
            height=jnp.array([height]),
            transform=tf,
        )

        return cap

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        capsule = trimesh.creation.capsule(
            radius=size[0].item(), height=size[1].item()
        )
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        capsule.vertices = trimesh.transform_points(capsule.vertices, tf)
        return capsule


@jdc.pytree_dataclass
class Ellipsoid(CollGeom):
    @staticmethod
    def from_center_and_abc(center: jax.Array, abc: jax.Array) -> Ellipsoid:
        batch_axes = center.shape[:-1]
        assert center.shape[-1] == 3

        mat = jaxlie.SO3.identity(batch_axes).as_matrix()

        # Uses ellipsoid.size as the radii.
        assert abc.shape == batch_axes + (3,)
        return Ellipsoid(pos=center, mat=mat, size=abc)
    
    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        ellipsoid = trimesh.creation.icosphere(radius=size[0].item())
        ellipsoid.apply_scale(size)
        tf = onp.eye(4)
        tf[:3, 3] = pos
        ellipsoid.vertices = trimesh.transform_points(ellipsoid.vertices, tf)
        return ellipsoid
    

@jdc.pytree_dataclass
class Convex(CollGeom):
    mesh: jdc.Static[trimesh.Trimesh]
    # MJX batching works across pose, but one unique mesh at a time.

    @staticmethod
    def from_convex_mesh(mesh: trimesh.Trimesh, batch_axes: tuple[int, ...] = ()) -> Convex:
        """
        Create geometry from convex mesh.
        """
        # Check if the mesh is convex.
        assert mesh.is_convex

        tf = jaxlie.SE3.identity(batch_axes)

        return Convex(
            pos=tf.translation(),
            mat=tf.rotation().as_matrix(),
            size=jnp.full((*batch_axes, 3), 0),  # unused.
            mesh=mesh,
        )
    
    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        mesh = self.mesh.copy()
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        mesh.vertices = trimesh.transform_points(mesh.vertices, tf)
        return mesh
    
    def get_mesh_mjx(self) -> ConvexMesh:
        # Based on https://github.com/google-deepmind/mujoco/blob/43a7493d1739d5cb1618de6863f929d99c7a8822/mjx/mujoco/mjx/_src/mesh.py#L280.
        vert = onp.array(self.mesh.vertices)
        face = onp.array(self.mesh.faces)
        face_normal = _get_face_norm(vert, face)
        edge, edge_face_normal = _get_edge_normals(face, face_normal)
        face = vert[face]  # materialize full nface x nvert matrix

        return ConvexMesh(
            vert=jnp.array(vert),
            face=jnp.array(face),
            face_normal=jnp.array(face_normal),
            edge=jnp.array(edge),
            edge_face_normal=jnp.array(edge_face_normal),
        )
    
@jdc.pytree_dataclass
class Cylinder(CollGeom):
    @staticmethod
    def from_radius_and_height(
        radius: jax.Array, height: jax.Array, transform: jaxlie.SE3
    ) -> Cylinder:
        batch_axes = transform.get_batch_axes()
        center = transform.translation()
        mat = transform.rotation().as_matrix()

        mat = jaxlie.SO3.identity(batch_axes).as_matrix()

        # Uses cylinder.size[0] as the radius and cylinder.size[1] as the height.
        assert radius.shape == batch_axes + (1,)
        assert height.shape == batch_axes + (1,)

        shape = jnp.concatenate([radius, height, jnp.zeros_like(radius)], axis=-1)
        return Cylinder(pos=center, mat=mat, size=shape)
    
    @staticmethod
    def from_min_cylinder(
        mesh: trimesh.Trimesh, batch_axes: tuple[int, ...] = ()
    ) -> Cylinder:
        """
        Approximate a minimum bounding cylinder for a mesh.
        """
        import trimesh.bounds

        results = trimesh.bounds.minimum_cylinder(mesh)

        assert "transform" in results
        assert "radius" in results
        assert "height" in results

        tf_mat = results["transform"]
        radius = results["radius"]
        height = results["height"]
        tf = jaxlie.SE3.from_matrix(tf_mat)

        cap = Cylinder.from_radius_and_height(
            radius=jnp.array([radius]),
            height=jnp.array([height]),
            transform=tf,
        )
        return cap

    def _create_one_mesh(self, pos: jax.Array, mat: jax.Array, size: jax.Array):
        cylinder = trimesh.creation.cylinder(
            radius=size[0].item(), height=size[1].item()
        )
        tf = onp.eye(4)
        tf[:3, :3] = mat
        tf[:3, 3] = pos
        cylinder.vertices = trimesh.transform_points(cylinder.vertices, tf)
        return cylinder