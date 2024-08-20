""" 02_collbody.py
Tests collision between `SphereColl` and `PlaneColl`.
"""

import time

import jax.numpy as jnp
import numpy as onp
import jaxlie

import viser

from jaxmp.collbody import PlaneColl, SphereColl

def main():
    """Create 2 balls and a plane, and check for collisions."""
    server = viser.ViserServer()
    plane_handle = server.scene.add_transform_controls("plane")
    sphere_1_handle = server.scene.add_transform_controls("sphere_1")
    sphere_2_handle = server.scene.add_transform_controls("sphere_2")

    plane_coll = PlaneColl(point=jnp.array([0.0, 0.0, 0.0]), normal=jnp.array([0.0, 0.0, 1.0]))
    sphere_1_coll = SphereColl(centers=jnp.array([[0.0, 0.0, 0.0]]), radii=jnp.array([0.2]))
    sphere_2_coll = SphereColl(centers=jnp.array([[0.0, 0.0, 0.0]]), radii=jnp.array([0.2]))

    def update_collisions():
        _plane_coll = plane_coll.transform(
            jaxlie.SE3(jnp.array([*plane_handle.wxyz, *plane_handle.position]))
        )
        _sphere_1_coll = sphere_1_coll.transform(
            jaxlie.SE3(jnp.array([*sphere_1_handle.wxyz, *sphere_1_handle.position]))
        )
        _sphere_2_coll = sphere_2_coll.transform(
            jaxlie.SE3(jnp.array([*sphere_2_handle.wxyz, *sphere_2_handle.position]))
        )

        plane_in_collision = False
        sphere_1_in_collision = False
        sphere_2_in_collision = False

        if jnp.any(_sphere_1_coll.dist(_sphere_2_coll) > 0.0):
            sphere_1_in_collision, sphere_2_in_collision = True, True
        if jnp.any(_sphere_1_coll.dist_to_plane(_plane_coll) > 0.0):
            plane_in_collision, sphere_1_in_collision = True, True
        if jnp.any(_sphere_2_coll.dist_to_plane(_plane_coll) > 0.0):
            plane_in_collision, sphere_2_in_collision = True, True

        # Visualize.
        plane_mesh = _plane_coll.to_trimesh()
        sphere_1_mesh = _sphere_1_coll.to_trimesh()
        sphere_2_mesh = _sphere_2_coll.to_trimesh()

        assert plane_mesh.visual is not None
        assert sphere_1_mesh.visual is not None
        assert sphere_2_mesh.visual is not None

        if plane_in_collision:
            plane_mesh.visual.vertex_colors = onp.array([1.0, 0.5, 0.5, 1.0])
        else:
            plane_mesh.visual.vertex_colors = onp.array([0.5, 1.0, 0.5, 1.0])

        if sphere_1_in_collision:
            sphere_1_mesh.visual.vertex_colors = onp.array([1.0, 0.5, 0.5, 1.0])
        else:
            sphere_1_mesh.visual.vertex_colors = onp.array([0.5, 1.0, 0.5, 1.0])

        if sphere_2_in_collision:
            sphere_2_mesh.visual.vertex_colors = onp.array([1.0, 0.5, 0.5, 1.0])
        else:
            sphere_2_mesh.visual.vertex_colors = onp.array([0.5, 1.0, 0.5, 1.0])

        server.scene.add_mesh_trimesh("plane_mesh", plane_mesh)
        server.scene.add_mesh_trimesh("sphere_1_mesh", sphere_1_mesh)
        server.scene.add_mesh_trimesh("sphere_2_mesh", sphere_2_mesh)

    
    plane_handle.on_update(lambda _: update_collisions())
    sphere_1_handle.on_update(lambda _: update_collisions())
    sphere_2_handle.on_update(lambda _: update_collisions())

    update_collisions()

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
