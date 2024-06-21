"""Generate antipodal grasps, given a mesh."""

from typing import Optional
import tyro
from pathlib import Path

import numpy as onp
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float, Int
import jaxlie


# Jit... probably won't work here because points/faces are different for each object.
def create_antipodal_grasps(
    vertices: Float[Array, "*batch points 3"],
    faces: Int[Array, "*batch faces 3"]
) -> Float[Array, "*batch faces 7"]:
    batch_size = vertices.shape[:-2]
    num_points = vertices.shape[-2]
    num_faces = faces.shape[-2]
    assert vertices.shape == (*batch_size, num_points, 3) and faces.shape == (*batch_size, num_faces, 3)

    # Combine the face + vertices into a single tensor.
    face_with_points = jax.vmap(lambda face: vertices[*batch_size, face], in_axes=-2)(faces)
    assert face_with_points.shape == (*batch_size, num_faces, 3, 3), face_with_points.shape  # 3 points per face, 3D points.

    # Calculate normals at each face + normalize it. Axes are based on the 0-th point of the face.
    _axis_x = (face_with_points[..., 1, :] - face_with_points[..., 0, :])
    _axis_y = -(face_with_points[..., 2, :] - face_with_points[..., 0, :])
    _axis_z = jnp.cross(_axis_x, _axis_y)
    _axis_x_normalized = _axis_x / jnp.linalg.norm(_axis_x, axis=-1, keepdims=True)
    _axis_z_normalized = _axis_z / jnp.linalg.norm(_axis_z, axis=-1, keepdims=True)

    face_normals = _axis_z_normalized
    assert face_normals.shape == (*batch_size, num_faces, 3) and jnp.isclose(jnp.linalg.norm(face_normals, axis=-1), 1.0).all()
    _axis_y_normalized = jnp.cross(face_normals, _axis_x_normalized)
    assert _axis_x_normalized.shape == (*batch_size, num_faces, 3)
    assert _axis_y_normalized.shape == (*batch_size, num_faces, 3)

    # Also create a SO3 representation of the normals, where normal is along the x-axis.
    face_normals_wxyz = jaxlie.SO3.from_matrix(
        jnp.stack((
            face_normals,
            _axis_x_normalized,
            _axis_y_normalized,  # these two should be orthogonal + normalized.
        ), axis=-1)
    ).wxyz

    # Sample points on the mesh -- for simplicity, let's do one point per face.
    random_face_idx = jax.nn.softmax(
        jax.random.uniform(
            jax.random.PRNGKey(0), (*batch_size, num_faces, 3), minval=0.01, maxval=1.0
        ),
        axis=-1,
    )
    assert random_face_idx.shape == (*batch_size, num_faces, 3) and jnp.isclose(random_face_idx.sum(axis=-1), 1.0).all()
    random_face_point = jnp.sum(face_with_points * random_face_idx[..., None], axis=-2)

    # Get the antipodal point, via batched ray-plane intersections. Then, check for bounds.
    # This section may include NaNs.
    assert face_normals.shape == (*batch_size, num_faces, 3)
    assert random_face_point.shape == (*batch_size, num_faces, 3)
    print(face_with_points[..., None, :, 0, :])
    t_intersect = (
        (
            jnp.multiply(
                jnp.broadcast_to(random_face_point[..., :, None, :], (*batch_size, num_faces, num_faces, 3)), 
                jnp.broadcast_to(face_normals[..., None, :, :], (*batch_size, num_faces, num_faces, 3))
            )
            - face_with_points[..., None, :, 0, :]
        ).sum(axis=-1)
    ) / (
        jnp.multiply(
            jnp.broadcast_to(face_normals[..., :, None, :], (*batch_size, num_faces, num_faces, 3)), 
            jnp.broadcast_to(face_normals[..., None, :, :], (*batch_size, num_faces, num_faces, 3))
        ).sum(axis=-1)
    )
    assert t_intersect.shape == (*batch_size, num_faces, num_faces)  # (grasps, num_faces).
    # assert jnp.isclose(t_intersect[*batch_size].diagonal(axis1=-2, axis2=-1), 0.0).all(), t_intersect
    assert jnp.isclose(t_intersect[*batch_size].diagonal(axis1=-2, axis2=-1), 0.0).all(), t_intersect[*batch_size].diagonal(axis1=-2, axis2=-1)

    # Then, check if the intersection point is within the face.
    p_intersect = random_face_point[..., :, None, :] + t_intersect[..., None] * face_normals[..., :, None, :]
    assert p_intersect.shape == (*batch_size, num_faces, num_faces, 3)
    bary_alpha = jnp.multiply(p_intersect - face_with_points[..., None, :, 0, :], _axis_x_normalized[:, None, :]).sum(axis=-1)
    bary_beta = jnp.multiply(-(p_intersect - face_with_points[..., None, :, 0, :]), _axis_y_normalized[:, None, :]).sum(axis=-1)
    inside_face = (
        (0 <= bary_alpha)
        & (bary_alpha <= jnp.linalg.norm(_axis_x, axis=-1))
        & (0 <= bary_beta)
        & (bary_beta <= jnp.linalg.norm(_axis_y, axis=-1))
        & (~jnp.isnan(t_intersect))
        & (~jnp.isinf(t_intersect))
        & (t_intersect > 0)
    )
    assert inside_face.shape == (*batch_size, num_faces, num_faces), inside_face.shape

    print(random_face_point[0])
    print(t_intersect[0])
    print(p_intersect[0])
    print(bary_alpha[0])
    print(bary_beta[0])
    print(inside_face[0])
    print(inside_face)

    # Get the normal at the antipodal point

    # Check for normal alignment

    grasps = jnp.concatenate([face_normals_wxyz, random_face_point], axis=-1)
    assert grasps.shape == (*batch_size, num_faces, 7)

    return grasps


def main(
    mesh_path: Optional[Path] = None,
):
    import trimesh
    import trimesh.creation

    if mesh_path is not None:
        mesh = trimesh.load(mesh_path)
    else:
        mesh = trimesh.creation.box([0.05]*3)
        # mesh = trimesh.creation.icosphere(radius=0.1)
    assert isinstance(mesh, trimesh.Trimesh)
    vertices, faces = mesh.vertices, mesh.faces

    # Create antipodal grasps
    grasps = create_antipodal_grasps(
        jnp.array(vertices),
        jnp.array(faces)
    )

    # Visualize the mesh, and the grasps!
    import viser
    server = viser.ViserServer()
    grasp_mesh = trimesh.creation.cylinder(radius=0.005, height=0.05)  # along z-axis.
    grasp_mesh.vertices = trimesh.transform_points(
        grasp_mesh.vertices,
        trimesh.transformations.rotation_matrix(jnp.pi / 2, [0, 1, 0]),
    )  # rotate to x-axis.

    server.scene.add_mesh_trimesh("mesh", mesh)

    # print(grasps)
    # server.scene.add_point_cloud("grasps", onp.array(grasps), onp.zeros_like(grasps), point_size=0.001)

    grasp_frame_list = []
    for i, grasp in enumerate(grasps):
        grasp_frame = server.scene.add_frame(
            f"grasps/grasp_{i}",
            show_axes=False,
            position=grasp[4:],
            wxyz=grasp[:4],
        )
        server.scene.add_mesh_trimesh(f"grasps/grasp_{i}/grasp", grasp_mesh)
        grasp_frame_list.append(grasp_frame)

    breakpoint()


if __name__ == "__main__":
    tyro.cli(main)
