""" 02_collbody.py
Tests collision between `SphereColl` and `PlaneColl`.
"""

import jax.numpy as jnp
import numpy as onp
import jaxlie

import viser

from jaxmp.collbody import PlaneColl, SphereColl

def main():
    """Create 2 balls and a plane, and check for collisions."""
    server = viser.ViserServer()
    plane_handle = server.scene.add_transform_controls("plane", scale=0.5)
    sphere_1_handle = server.scene.add_transform_controls("sphere_1", scale=0.5)
    sphere_2_handle = server.scene.add_transform_controls("sphere_2", scale=0.5)

    while True:
        plane_coll = PlaneColl(
            jnp.array([*plane_handle.position]),
            jaxlie.SO3(jnp.array(plane_handle.wxyz)).as_matrix()[:, 2]
        )
        sphere_1_coll = SphereColl(jnp.array([[*sphere_1_handle.position]]), jnp.array([0.2]))
        sphere_2_coll = SphereColl(jnp.array([[*sphere_2_handle.position]]), jnp.array([0.2]))

        plane_in_collision = False
        sphere_1_in_collision = False
        sphere_2_in_collision = False

        if jnp.any(sphere_1_coll.dist(sphere_2_coll) > 0.0):
            sphere_1_in_collision, sphere_2_in_collision = True, True
        if jnp.any(sphere_1_coll.dist_to_plane(plane_coll) > 0.0):
            plane_in_collision, sphere_1_in_collision = True, True
        if jnp.any(sphere_2_coll.dist_to_plane(plane_coll) > 0.0):
            plane_in_collision, sphere_2_in_collision = True, True

        # Visualize.
        plane_mesh = plane_coll.to_trimesh()
        sphere_1_mesh = sphere_1_coll.to_trimesh()
        sphere_2_mesh = sphere_2_coll.to_trimesh()

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

if __name__ == "__main__":
    main()
