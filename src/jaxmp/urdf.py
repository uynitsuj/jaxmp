"""Copied from brent's `brent_jax_trajsmooth.py`."""

from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import viser.transforms as vtf
import yourdfpy
import warnings
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int, Bool
from typing import Dict, List, Tuple
import tyro

from jaxmp.collision import SphereCollision


@jdc.pytree_dataclass
class JaxUrdf:
    """A differentiable robot kinematics model."""

    num_joints: jdc.Static[int]
    joint_names: jdc.Static[Tuple[str]]
    """List of joint names, in order."""

    num_actuated_joints: jdc.Static[int]
    idx_actuated_joint: Int[Array, "joints"]
    """Index of actuated joint in `act_joints`, if it is actuated. -1 otherwise."""

    joint_twists: Float[Array, "act_joints 6"]
    limits_lower: Float[Array, "act_joints"]
    limits_upper: Float[Array, "act_joints"]

    Ts_parent_joint: Float[Array, "joints 7"]
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
        joint_names = list[str]()
        idx_actuated_joint = list[int]()

        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            assert joint.origin.shape == (4, 4)
            joint_names.append(joint.name)

            # Check if this joint is a mimic joint.
            if joint.mimic is not None:
                mimicked_joint = urdf.joint_map[joint.mimic.joint]
                mimicked_joint_idx = urdf.actuated_joints.index(mimicked_joint)
                assert mimicked_joint_idx < joint_idx, "Code + fk `fori_loop` assumes this!"
                warnings.warn("Mimic joint detected, assuming multiplier=1.0, offset=0.0.")
                act_joint_idx = urdf.actuated_joints.index(mimicked_joint)
                idx_actuated_joint.append(act_joint_idx)

                # ... skip the twist info, since mimic joints are not actuated.
            elif joint in urdf.actuated_joints:
                assert joint.axis.shape == (3,)
                assert (
                    joint.limit is not None
                    and joint.limit.lower is not None
                    and joint.limit.upper is not None
                ), "We currently assume there are joint limits!"
                joint_lim_lower.append(joint.limit.lower)
                joint_lim_upper.append(joint.limit.upper)

                act_joint_idx = urdf.actuated_joints.index(joint)
                idx_actuated_joint.append(act_joint_idx)

                # We use twists in the (v, omega) convention.
                if joint.type == "revolute":
                    joint_twists.append(onp.concatenate([onp.zeros(3), joint.axis]))
                elif joint.type == "prismatic":
                    joint_twists.append(onp.concatenate([joint.axis, onp.zeros(3)]))
            else:
                idx_actuated_joint.append(-1)


            # Get the transform from the parent joint to the current joint.
            # Do this for all the joints.
            T_parent_joint = joint.origin
            if joint.parent not in joint_from_child:
                # Must be root node.
                parent_indices.append(-1)
            else:
                parent_joint = joint_from_child[joint.parent]
                parent_index = urdf.joint_names.index(parent_joint.name)
                if parent_index >= joint_idx:
                    warnings.warn(f"Parent index {parent_index} >= joint index {joint_idx}! Assuming that parent is root.")
                    if parent_joint.parent != urdf.scene.graph.base_frame:
                        raise ValueError("Parent index >= joint_index, but parent is not root!")
                    T_parent_joint = parent_joint.origin @ T_parent_joint  # T_root_joint.
                    parent_index = -1
                parent_indices.append(parent_index)

            Ts_parent_joint.append(vtf.SE3.from_matrix(T_parent_joint).wxyz_xyz)  # type: ignore

        joint_twists = jnp.array(joint_twists)
        limits_lower = jnp.array(joint_lim_lower)
        limits_upper = jnp.array(joint_lim_upper)
        Ts_parent_joint = jnp.array(Ts_parent_joint)
        parent_indices = jnp.array(parent_indices)
        idx_actuated_joint = jnp.array(idx_actuated_joint)
        joint_names = tuple[str](joint_names)

        num_joints = len(urdf.joint_map)
        num_actuated_joints = len(urdf.actuated_joints)

        assert idx_actuated_joint.shape == (len(urdf.joint_map),)
        assert joint_twists.shape == (num_actuated_joints, 6)
        assert limits_lower.shape == (num_actuated_joints,)
        assert limits_upper.shape == (num_actuated_joints,)
        # assert mimic_joint_inds.shape == (num_joints,)
        assert Ts_parent_joint.shape == (num_joints, 7)
        assert parent_indices.shape == (num_joints,)
        assert idx_actuated_joint.max() == num_actuated_joints - 1

        return JaxUrdf(
            num_joints=num_joints,
            joint_names=joint_names,
            num_actuated_joints=num_actuated_joints,
            idx_actuated_joint=idx_actuated_joint,
            joint_twists=joint_twists,
            Ts_parent_joint=Ts_parent_joint,
            limits_lower=limits_lower,
            limits_upper=limits_upper,
            parent_indices=parent_indices,
        )

    # @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch num_act_joints"],
    ) -> Float[Array, "*batch num_joints 7"]:
        batch_axes = cfg.shape[:-1]
        assert cfg.shape == (*batch_axes, self.num_actuated_joints)

        Ts_joint_child = jaxlie.SE3.exp(self.joint_twists * cfg[..., None]).wxyz_xyz
        assert Ts_joint_child.shape == (*batch_axes, self.num_actuated_joints, 7)

        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                self.parent_indices[i] == -1,
                jnp.broadcast_to(jaxlie.SE3.identity().wxyz_xyz, (*batch_axes, 7)),
                Ts_world_joint[..., self.parent_indices[i], :],
            )

            # If this joint is a mimic joint, then we need to use the mimic joint's transform.
            # if self.mimic_joint_inds[i] != -1:
            #     actuated_joint_idx = self.mimic_joint_inds[i]
            # else:
            #     actuated_joint_idx = self.idx_actuated_joint[i]
            # actuated_joint_idx = jnp.where(
            #     self.mimic_joint_inds[i] != -1,
            #     self.mimic_joint_inds[i],
            #     self.idx_actuated_joint[i]
            # )
            actuated_joint_idx = self.idx_actuated_joint[i]

            T_joint_child = jnp.where(
                actuated_joint_idx != -1,
                Ts_joint_child[..., actuated_joint_idx, :],
                jnp.broadcast_to(jaxlie.SE3.identity().wxyz_xyz, (*batch_axes, 7)),
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent)
                    @ jaxlie.SE3(self.Ts_parent_joint[i])
                    @ jaxlie.SE3(T_joint_child)
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

        for joint_idx, joint in enumerate(urdf.joint_map.values()):
            if joint.child not in coll_dict:
                continue

            list_of_spheres = coll_dict[joint.child]
            for sphere in list_of_spheres:
                idx.append(joint_idx)
                centers.append(sphere['center'])
                radii.append(sphere['radius'])


        idx = jnp.array(idx)
        centers = jnp.array(centers)
        radii = jnp.array(radii)
        num_spheres = centers.shape[0]

        assert (
            idx.shape == (num_spheres,) and
            centers.shape == (num_spheres, 3) and
            radii.shape == (num_spheres,)
        )

        jax_urdf = JaxUrdf.from_urdf(urdf)
        return JaxUrdfwithCollision(
            num_joints=len(jax_urdf.parent_indices),
            joint_names=jax_urdf.joint_names,
            num_actuated_joints=jax_urdf.num_actuated_joints,
            idx_actuated_joint=jax_urdf.idx_actuated_joint,
            joint_twists=jax_urdf.joint_twists,
            Ts_parent_joint=jnp.array(jax_urdf.Ts_parent_joint),
            limits_lower=jnp.array(jax_urdf.limits_lower),
            limits_upper=jnp.array(jax_urdf.limits_upper),
            parent_indices=jnp.array(jax_urdf.parent_indices),
            num_spheres=num_spheres,
            coll_link_idx=idx,
            coll_link_centers=centers,
            coll_link_radii=radii,
        )

    @jdc.jit
    def d_world(
        self,
        cfg: Float[Array, "num_act_joints"],
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
        cfg: Float[Array, "*batch num_act_joints"]
    ) -> Tuple[Float[Array, "*batch num_spheres 3"], Float[Array, "*batch num_spheres"]]:
        """Get the spheres in the world frame, in the provided configuration."""
        batch_size = cfg.shape[:-1]
        num_spheres = self.coll_link_centers.shape[0]
        Ts_world_joint = self.forward_kinematics(cfg)
        assert Ts_world_joint.shape == (*batch_size, self.num_joints, 7)

        centers = jaxlie.SE3.from_translation(self.coll_link_centers)
        centers_transformed = (
            jaxlie.SE3(Ts_world_joint[..., self.coll_link_idx, :])
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

    from robot_descriptions.loaders.yourdfpy import load_robot_description
    yourdf = load_robot_description("yumi_description")

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

    jax_urdf.forward_kinematics(jnp.array(yumi_rest))
    breakpoint()

    while True:
        time.sleep(10)


if __name__ == "__main__":
    tyro.cli(main)
