""" 02_collbody.py
Tests the different collision bodies, as defined in `collbody.py`.
"""

import time
import trimesh

import jax.numpy as jnp
import numpy as onp
import jaxlie

import viser
import viser.extras

from jaxmp.coll import collide, Capsule, Sphere, Plane, Convex
# from jaxmp.coll._coll_mjx_types import Convex

def main():
    server = viser.ViserServer()
    timing_handle = server.gui.add_number("Timing (ms)", 0.001, disabled=True)

    meshes = [trimesh.creation.uv_sphere(radius=0.2) for _ in range(2)]
    convex_0 = Convex.from_meshes(meshes)
    convex_0 = convex_0.transform(
        jaxlie.SE3.from_translation(jnp.array([[0.0, 0, 0], [1.0, 0, 0]]))
    )

    meshes = [trimesh.creation.uv_sphere(radius=0.2), trimesh.creation.uv_sphere(radius=0.4)]
    convex_1 = Convex.from_meshes(meshes)
    convex_1 = convex_1.transform(
        jaxlie.SE3.from_translation(jnp.array([[0.0, 0, 0], [1.0, 0, 0]]))
    )

    coll_list = [
        Plane.from_point_and_normal(jnp.zeros((3,)), jnp.array([0.0, 0.0, 1.0])),
        Sphere.from_center_and_radius(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.1])),
        Capsule.from_radius_and_height(jnp.array([0.1]), jnp.array([0.2]), jaxlie.SE3.identity()),
        Convex.from_meshes([trimesh.creation.uv_sphere(radius=0.2)]),
        Convex.from_meshes([trimesh.creation.uv_sphere(radius=0.2)]),
        convex_0.reshape(-1, 1, mesh_axis=0),
        convex_0.reshape(1, -1, mesh_axis=1),
        convex_1.reshape(-1, 1, mesh_axis=0),
        convex_1.reshape(1, -1, mesh_axis=1),
    ]
    handle_list = [server.scene.add_transform_controls(f"coll_{i}") for i in range(len(coll_list))]

    def update_collisions():
        start_time = time.time()
        _coll_list = [
            coll.transform(jaxlie.SE3(jnp.array([*handle.wxyz, *handle.position])))
            for (coll, handle) in zip(coll_list, handle_list)
        ]

        in_collision_list = [False] * len(coll_list)

        for i in range(len(coll_list)):
            for j in range(i + 1, len(coll_list)):
                collision = collide(_coll_list[i], _coll_list[j])
                if jnp.any(collision.dist < 0.0):
                    in_collision_list[i], in_collision_list[j] = True, True

        timing_handle.value = (time.time() - start_time) * 1000

        # Visualize.
        mesh_list = [coll.to_trimesh() for coll in _coll_list]

        for i, (mesh, in_collision) in enumerate(zip(mesh_list, in_collision_list)):
            assert mesh.visual is not None
            if in_collision:
                mesh.visual.vertex_colors = onp.array([1.0, 0.5, 0.5, 1.0])
            else:
                mesh.visual.vertex_colors = onp.array([0.5, 1.0, 0.5, 1.0])
            server.scene.add_mesh_trimesh(f"coll_mesh_{i}", mesh)


    while True:
        update_collisions()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
