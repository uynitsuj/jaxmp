""" 02_collbody.py
Tests the different collision bodies, as defined in `collbody.py`.
"""

import time
import trimesh

import jax.numpy as jnp
from jaxmp.extras.urdf_loader import load_urdf
from jaxmp.kinematics import JaxKinTree
import numpy as onp
import jaxlie

import viser
import viser.extras

from jaxmp.coll import collide, Capsule, Sphere, Plane, RobotColl, Convex

def main():
    server = viser.ViserServer()
    timing_handle = server.gui.add_number("Timing (ms)", 0.001, disabled=True)

    # Load robot description.
    urdf = load_urdf("yumi")

    kin = JaxKinTree.from_urdf(urdf)
    robot_coll = RobotColl.from_urdf(urdf)
    rest_pose = (kin.limits_upper + kin.limits_lower) / 2

    breakpoint()

    # Visualize robot.
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")
    server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    # Put robot to rest pose.
    urdf_vis.update_cfg(onp.array(rest_pose))

    Ts_joint_world = kin.forward_kinematics(rest_pose)

    coll_list = [
        robot_coll.transform(jaxlie.SE3(Ts_joint_world[..., robot_coll.link_joint_idx, :])),
        Plane.from_point_and_normal(jnp.zeros((3,)), jnp.array([0.0, 0.0, 1.0])).broadcast_to(10,1),
        Sphere.from_center_and_radius(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.1])),
        Capsule.from_radius_and_height(jnp.array([0.1]), jnp.array([0.2]), jaxlie.SE3.identity()),
        Convex.from_convex_mesh(trimesh.creation.box(extents=[0.1, 0.1, 0.1])),
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
                dist = collision[0]
                if jnp.any(dist < 0.0):
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
