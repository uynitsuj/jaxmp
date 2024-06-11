"""Copied from brent's `brent_jax_trajsmooth.py`."""

from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import viser.transforms as vtf
import yourdfpy
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int
from typing import Dict, List, Tuple
import tyro

from jaxmp.collision import SphereCollision

# Maybe this can be basically a reimplementaiton of `yourdfpy`, but in jax.
# And we can take inspiration from keeping actuated_joints and joints separate.
# Then we can optimize for the actuated joints, and use the other joints as constraints.

@jdc.pytree_dataclass
class JaxUrdf:
    """A differentiable robot kinematics model."""

    num_joints: jdc.Static[int]
    joint_twists: Float[Array, "joints 6"]
    Ts_parent_joint: Float[Array, "joints 7"]
    limits_lower: Float[Array, "joints"]
    limits_upper: Float[Array, "joints"]
    parent_indices: Int[Array, "joints"]

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
    ) -> JaxUrdf:
        """Build a differentiable robot model from a URDF."""

        joint_from_child = {joint.child: joint for joint in urdf.joint_map.values()}

        # Get the parent indices + joint twist parameters.
        joint_twists = list[onp.ndarray]()
        Ts_parent_joint = list[onp.ndarray]()
        joint_lim_lower = list[float]()
        joint_lim_upper = list[float]()
        parent_indices = list[int]()
        for joint in urdf.actuated_joints:
            assert joint.origin.shape == (4, 4)
            assert joint.axis.shape == (3,)
            assert (
                joint.limit is not None
                and joint.limit.lower is not None
                and joint.limit.upper is not None
            ), "We currently assume there are joint limits!"
            joint_lim_lower.append(joint.limit.lower)
            joint_lim_upper.append(joint.limit.upper)

            # We use twists in the (v, omega) convention.
            if joint.type == "revolute":
                joint_twists.append(onp.concatenate([onp.zeros(3), joint.axis]))
            elif joint.type == "prismatic":
                joint_twists.append(onp.concatenate([joint.axis, onp.zeros(3)]))
            else:
                assert False

            # Get the transform from the parent joint to the current joint.
            # The loop is required to take unactuated joints into account.
            T_parent_joint = joint.origin
            parent_joint = joint_from_child[joint.parent]
            root = False
            while parent_joint not in urdf.actuated_joints:
                T_parent_joint = parent_joint.origin @ T_parent_joint
                if parent_joint.parent not in joint_from_child:
                    root = True
                    break
                parent_joint = joint_from_child[parent_joint.parent]
            Ts_parent_joint.append(vtf.SE3.from_matrix(T_parent_joint).wxyz_xyz)  # type: ignore
            parent_indices.append(
                -1 if root else urdf.actuated_joints.index(parent_joint)
            )

        joint_twists = jnp.array(joint_twists)
        return JaxUrdf(
            num_joints=len(parent_indices),
            joint_twists=joint_twists,
            Ts_parent_joint=jnp.array(Ts_parent_joint),
            limits_lower=jnp.array(joint_lim_lower),
            limits_upper=jnp.array(joint_lim_upper),
            parent_indices=jnp.array(parent_indices),
        )

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch num_joints"],
    ) -> Float[Array, "*batch num_joints 7"]:
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.num_joints)

        Ts_joint_child = jaxlie.SE3.exp(self.joint_twists * cfg[..., None]).wxyz_xyz
        assert Ts_joint_child.shape == (*batch_axes, self.num_joints, 7)

        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                self.parent_indices[i] == -1,
                jnp.broadcast_to(jaxlie.SE3.identity().wxyz_xyz, (*batch_axes, 7)),
                Ts_world_joint[..., self.parent_indices[i], :],
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent)
                    @ jaxlie.SE3(self.Ts_parent_joint[i])
                    @ jaxlie.SE3(Ts_joint_child[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=self.num_joints,
            body_fun=compute_joint,
            init_val=jnp.zeros((*batch_axes, self.num_joints, 7)),
        )
        assert Ts_world_joint.shape == (*batch_axes, self.num_joints, 7)
        return Ts_world_joint


@jdc.pytree_dataclass
class JaxUrdfwithCollision(JaxUrdf):
    """A differentiable, collision-aware robot kinematics model."""
    num_spheres: jdc.Static[int]
    Ts_coll_joint: Float[Array, "spheres 7"]
    coll_link_idx: Int[Array, "spheres"]
    coll_link_centers: Float[Array, "spheres 3"]
    coll_link_radii: Float[Array, "spheres"]
    # TODO self-collision + self-collision ignore -- put this into CollisionBody?
    # self_ignore: Float[Array, "spheres spheres"]  # ...make this sparse?
    # This section ignores mimic joints!

    @staticmethod
    def from_urdf_and_coll(
        urdf: yourdfpy.URDF,
        coll_dict: Dict[str, List[Dict[str, List]]]
    ):
        # scrape the collision data.
        idx, centers, radii = [], [], []
        Ts_coll_joint = []

        joint_from_child = {joint.child: joint for joint in urdf.joint_map.values()}

        for joint_name, joint in urdf.joint_map.items():
            if joint.child not in coll_dict:
                continue

            # Find the actuated joint that is the parent of this link.
            parent_joint = joint
            T_coll_joint = onp.eye(4)
            while parent_joint not in urdf.actuated_joints:
                T_coll_joint = parent_joint.origin @ T_coll_joint
                parent_joint = joint_from_child[parent_joint.parent]

            joint_idx = urdf.actuated_joints.index(parent_joint)
            list_of_spheres = coll_dict[joint.child]
            for sphere in list_of_spheres:
                idx.append(joint_idx)
                centers.append(sphere['center'])
                radii.append(sphere['radius'])
                Ts_coll_joint.append(vtf.SE3.from_matrix(T_coll_joint).wxyz_xyz)


        idx = jnp.array(idx)
        centers = jnp.array(centers)
        radii = jnp.array(radii)
        Ts_coll_joint = jnp.array(Ts_coll_joint)
        num_spheres = centers.shape[0]

        assert (
            idx.shape == (num_spheres,) and
            centers.shape == (num_spheres, 3) and
            radii.shape == (num_spheres,) and
            Ts_coll_joint.shape == (num_spheres, 7)
        )

        jax_urdf = JaxUrdf.from_urdf(urdf)
        return JaxUrdfwithCollision(
            num_joints=jax_urdf.num_joints,
            joint_twists=jax_urdf.joint_twists,
            Ts_parent_joint=jax_urdf.Ts_parent_joint,
            limits_lower=jax_urdf.limits_lower,
            limits_upper=jax_urdf.limits_upper,
            parent_indices=jax_urdf.parent_indices,
            num_spheres=num_spheres,
            Ts_coll_joint=Ts_coll_joint,
            coll_link_idx=idx,
            coll_link_centers=centers,
            coll_link_radii=radii,
        )

    @jdc.jit
    def d_world(
        self,
        cfg: Float[Array, "num_joints"],
        other: SphereCollision
    ) -> Float[Array, "num_joints"]:
        """Check if the robot collides with the world, in the provided configuration.
        Get the max signed distance field (sdf) for each joint."""
        centers, radii = self.as_spheres(cfg)
        coll = SphereCollision(centers=centers, radii=radii)
        max_sdf = coll.collides_other(other)

        link_idx_as_mask = jnp.arange(self.num_joints)[:, None] == self.coll_link_idx[None]
        assert link_idx_as_mask.shape == (self.num_joints, self.num_spheres)
        assert max_sdf.shape == (self.num_spheres,)
        max_sdf_per_joint = jnp.max(
            jnp.where(
                link_idx_as_mask,
                jnp.broadcast_to(max_sdf, (self.num_joints, self.num_spheres)),
                -jnp.inf,
            ),
            axis=-1,
        )

        assert max_sdf_per_joint.shape == (self.num_joints,)
        return max_sdf_per_joint

    @jdc.jit
    def as_spheres(
        self,
        cfg: Float[Array, "*batch num_joints"]
    ) -> Tuple[Float[Array, "*batch num_spheres 3"], Float[Array, "*batch num_spheres"]]:
        """Get the spheres in the world frame, in the provided configuration."""
        batch_size = cfg.shape[:-1]
        num_spheres = self.coll_link_centers.shape[0]
        Ts_world_joint = self.forward_kinematics(cfg)

        centers = jaxlie.SE3.from_translation(self.coll_link_centers)
        centers_transformed = (
            jaxlie.SE3(Ts_world_joint[..., self.coll_link_idx, :])
            @ jaxlie.SE3(self.Ts_coll_joint)
            @ centers
        ).translation()

        assert centers_transformed.shape == (*batch_size, num_spheres, 3)
        assert self.coll_link_radii.shape == (num_spheres,)
        return centers_transformed, self.coll_link_radii

def force_yumi_gripper_frozen() -> yourdfpy.URDF:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    yourdf = load_robot_description("yumi_description")
    yourdf.joint_map['gripper_l_joint'].type = 'fixed'
    yourdf.joint_map['gripper_l_joint'].origin[0, 3] -= 0.025
    yourdf.joint_map['gripper_l_joint_m'].type = 'fixed'
    yourdf.joint_map['gripper_l_joint_m'].mimic = None
    yourdf.joint_map['gripper_l_joint_m'].origin[0, 3] += 0.025
    yourdf.joint_map['gripper_r_joint'].type = 'fixed'
    yourdf.joint_map['gripper_r_joint'].origin[0, 3] -= 0.025
    yourdf.joint_map['gripper_r_joint_m'].type = 'fixed'
    yourdf.joint_map['gripper_r_joint_m'].mimic = None
    yourdf.joint_map['gripper_r_joint_m'].origin[0, 3] += 0.025
    yourdf._update_actuated_joints()
    yourdf._cfg = yourdf.zero_cfg
    yourdf._scene = yourdf._create_scene(
        use_collision_geometry=False,
        load_geometry=True,
        force_mesh=True,
        force_single_geometry_per_link=True,
    )
    return yourdf

def main():
    from pathlib import Path
    import yaml
    import viser
    import viser.extras
    import trimesh.creation
    import time

    yumi_cfg_path = Path(__file__).parent.parent.parent / "data/yumi.yml"
    yumi_cfg = yaml.load(yumi_cfg_path.read_text(), Loader=yaml.Loader)
    yumi_coll = yumi_cfg['robot_cfg']['collision_spheres']

    # load the rest pose.
    yumi_rest = onp.array(yumi_cfg['robot_cfg']['rest_pose'])

    # Temporary hack...
    yourdf = force_yumi_gripper_frozen()

    jax_urdf = JaxUrdfwithCollision.from_urdf_and_coll(yourdf, yumi_coll)

    server = viser.ViserServer()

    rob_spheres_list = []
    vis_spheres_gui = server.gui.add_checkbox("vis_spheres", initial_value=False)
    @vis_spheres_gui.on_update
    def _(_):
        nonlocal rob_spheres_list
        if not vis_spheres_gui.value:
            for handle in rob_spheres_list:
                handle.remove()
            rob_spheres_list = []
        else:
            centers, radii = jax_urdf.as_spheres(jnp.array(yumi_rest))
            for i, (center, radius) in enumerate(zip(centers, radii)):
                sphere_mesh = trimesh.creation.icosphere(radius=radius)
                rob_spheres_list.append(
                    server.scene.add_mesh_trimesh(
                        f"sphere/{i}", sphere_mesh, position=center
                    )
                )

    urdf = viser.extras.ViserUrdf(
        server, yourdf, root_node_name="/urdf"
    )
    urdf.update_cfg(yumi_rest)


    while True:
        time.sleep(10)


if __name__ == "__main__":
    tyro.cli(main)
