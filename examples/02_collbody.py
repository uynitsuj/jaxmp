""" 02_collbody.py
Tests the different collision bodies, as defined in `collbody.py`.
"""

import time

import jax.numpy as jnp
import numpy as onp
import jaxlie

import viser

from jaxmp.coll.collision_types import CollBody, PlaneColl, SphereColl, CapsuleColl, HalfSpaceColl
from jaxmp.coll.collision_sdf import dist_signed

def main():
    server = viser.ViserServer()

    coll_list: list[CollBody] = [
        PlaneColl(point=jnp.array([0.0, 0.0, 0.0]), normal=jnp.array([0.0, 0.0, 1.0])),
        HalfSpaceColl(
            point=jnp.array([0.0, 0.0, 0.0]), normal=jnp.array([0.0, 0.0, 1.0])
        ),
        SphereColl(centers=jnp.array([[0.0, 0.0, 0.0]]), radii=jnp.array([0.2])),
        SphereColl(centers=jnp.array([[0.0, 0.0, 0.0]]), radii=jnp.array([0.2])),
        CapsuleColl(
            radii=jnp.array([0.2]),
            heights=jnp.array([0.5]),
            tf=jaxlie.SE3.identity(batch_axes=(1,)),
        ),
        CapsuleColl(
            radii=jnp.array([0.2]),
            heights=jnp.array([0.5]),
            tf=jaxlie.SE3.identity(batch_axes=(1,)),
        ),
    ]
    handle_list = [server.scene.add_transform_controls(f"coll_{i}") for i in range(len(coll_list))]

    def update_collisions():
        _coll_list = [
            coll.transform(jaxlie.SE3(jnp.array([*handle.wxyz, *handle.position])))
            for (coll, handle) in zip(coll_list, handle_list)
        ]

        in_collision_list = [False] * len(coll_list)

        for i in range(len(coll_list)):
            for j in range(i + 1, len(coll_list)):
                try:
                    if jnp.any(dist_signed(_coll_list[i], _coll_list[j]) > 0.0):
                        in_collision_list[i], in_collision_list[j] = True, True
                except NotImplementedError:
                    pass

        # Visualize.
        mesh_list = [coll.to_trimesh() for coll in _coll_list]

        for i, (mesh, in_collision) in enumerate(zip(mesh_list, in_collision_list)):
            assert mesh.visual is not None
            if in_collision:
                mesh.visual.vertex_colors = onp.array([1.0, 0.5, 0.5, 1.0])
            else:
                mesh.visual.vertex_colors = onp.array([0.5, 1.0, 0.5, 1.0])
            server.scene.add_mesh_trimesh(f"coll_mesh_{i}", mesh)

    for handle in handle_list:
        handle.on_update(lambda _: update_collisions())

    update_collisions()

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
