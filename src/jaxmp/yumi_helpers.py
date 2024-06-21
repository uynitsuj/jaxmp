import viser.transforms as vtf
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as onp

import yourdfpy


def get_yumi_urdf(
    include_dummy: bool = True,
    freeze_gripper: bool = True,
) -> yourdfpy.URDF:
    """Create yumi urdf with optional dummy points for grippers centerpoints.
    Offset from gripper is hardcoded to 13cm."""
    yourdf = load_robot_description("yumi_description")

    if freeze_gripper:
        yourdf.joint_map['gripper_r_joint'].type = 'fixed'
        yourdf.joint_map['gripper_r_joint_m'].type = 'fixed'
        yourdf.joint_map['gripper_r_joint_m'].mimic = None
        yourdf.joint_map['gripper_r_joint'].origin[0, 3] -= 0.025
        yourdf.joint_map['gripper_r_joint_m'].origin[0, 3] += 0.025
        yourdf.joint_map['gripper_l_joint'].type = 'fixed'
        yourdf.joint_map['gripper_l_joint_m'].type = 'fixed'
        yourdf.joint_map['gripper_l_joint_m'].mimic = None
        yourdf.joint_map['gripper_l_joint'].origin[0, 3] -= 0.025
        yourdf.joint_map['gripper_l_joint_m'].origin[0, 3] += 0.025

        yourdf._create_maps()
        yourdf._update_actuated_joints()
        yourdf._scene = yourdf._create_scene(load_geometry=True)

    if not include_dummy:
        return yourdf

    yourdf.robot.joints.extend(
        [
            yourdfpy.Joint(
                name="gripper_r_dummy",
                type="fixed",
                parent="gripper_r_base",
                child="gripper_r_dummy_point",
                origin=vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_x_radians(onp.pi/2),
                    translation=onp.array([0.0, 0.0, 0.13])
                ).as_matrix(),
            ),
            yourdfpy.Joint(
                name="gripper_l_dummy",
                type="fixed",
                parent="gripper_l_base",
                child="gripper_l_dummy_point",
                origin=vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_x_radians(onp.pi/2),
                    translation=onp.array([0.0, 0.0, 0.13])
                ).as_matrix(),
            ),
        ]
    )
    yourdf.robot.links.extend(
        [
            yourdfpy.Link(name="gripper_r_dummy_point"),
            yourdfpy.Link(name="gripper_l_dummy_point"),
        ]
    )
    yourdf._create_maps()
    yourdf._scene = yourdf._create_scene(load_geometry=True)
    return yourdf
