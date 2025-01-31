# JAX by default pre-allocates 75%, which will cause OOM with nerfstudio model loading.
# This line needs to go above any jax import.
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from typing import Literal, Optional
import time
from dataclasses import dataclass
from loguru import logger

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import viser
import viser.extras
import os

from jaxmp import JaxKinTree, RobotFactors, BatchedRobotFactors
from jaxmp.extras.urdf_loader import load_urdf
from jaxmp.extras.solve_ik import solve_ik, solve_ik_batched

from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../../data")


@dataclass
class TransformHandle:
    """Data class to store transform handles."""
    frame: viser.FrameHandle
    control: Optional[viser.TransformControlsHandle] = None

class YuMiBaseInterface:
    """
    Base interface for YuMi robot visualization.
    - This class does not require ROS or real robot as this is a VIRTUAL representation, but serves as base class for the ROS interface
    - Running this file allows you to control a virtual YuMi robot in viser with transform handle gizmos.
    """
    
    YUMI_REST_POSE = {
        "yumi_joint_1_r": 1.21442839,
        "yumi_joint_2_r": -1.03205606,
        "yumi_joint_7_r": -1.10072738,
        "yumi_joint_3_r": 0.2987352 - 0.2,
        "yumi_joint_4_r": -1.85257716,
        "yumi_joint_5_r": 1.25363652,
        "yumi_joint_6_r": -2.42181893,
        "yumi_joint_1_l": -1.24839656,
        "yumi_joint_2_l": -1.09802876,
        "yumi_joint_7_l": 1.06634394,
        "yumi_joint_3_l": 0.31386161 - 0.2,
        "yumi_joint_4_l": 1.90125141,
        "yumi_joint_5_l": 1.3205139,
        "yumi_joint_6_l": 2.43563939,
        "gripper_r_joint": 0,
        "gripper_l_joint": 0,
    }

    def __init__(
        self,
        minimal: bool = False,
        pos_weight: float = 5.0,
        rot_weight: float = 1.0,
        rest_weight: float = 0.01,
        limit_weight: float = 100.0,
        device: Literal["cpu", "gpu"] = "gpu",
    ):
        self.minimal = minimal
        # Set device
        jax.config.update("jax_platform_name", device)

        # Initialize viser server
        self.server = viser.ViserServer()
        
        # Load robot description
        # self.urdf = load_urdf("yumi", None)
        self.num_batches = 50
        self.urdf = load_urdf(None, Path(data_dir + "/yumi_description/urdf/yumi.urdf"))
        self.kin = JaxKinTree.from_urdf(self.urdf)
        self.rest_pose = jnp.array(list(self.YUMI_REST_POSE.values()))
        self.JointVar = BatchedRobotFactors.get_var_class(self.kin, self.rest_pose, self.num_batches)
        
        self.env = 0
        
        # Env selector
        self.env_selector = self.server.gui.add_dropdown(
            "Environment Selector",
            [str(i) for i in range(self.num_batches)],
            initial_value='0'
        )
        
        # Store weights for IK
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.rest_weight = rest_weight
        self.limit_weight = limit_weight
        
        # Target transform handle names HARDCODED BAD
        # self.target_names = ["yumi_joint_6_l", "yumi_joint_6_r"]
        self.target_names = ["left_dummy_joint", "right_dummy_joint"]
        
        self.joints = self.rest_pose
        
        # Setup visualization
        self._setup_visualization()
        
        if not minimal:
            self._setup_gui()
            self._setup_transform_handles()
        
        # Initialize state
        self.base_pose = jaxlie.SE3.identity()
        self.base_frame.position = onp.array(self.base_pose.translation())
        self.base_frame.wxyz = onp.array(self.base_pose.rotation().wxyz)
        
        if not minimal:
            # Initialize solver parameters
            self.solver_type = "conjugate_gradient"
            self.smooth = False
            self.manipulability_weight = 0.0
            self.has_jitted = False
            
            self.base_mask, self.target_mask = self.get_freeze_masks()
            self.ConstrainedSE3Var = BatchedRobotFactors.get_constrained_se3(self.base_mask)
        
    def _setup_visualization(self):
        """Setup basic visualization elements."""
        # Add base frame and robot URDF
        self.base_frame = self.server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis = viser.extras.ViserUrdf(
            self.server, 
            self.urdf, 
            root_node_name="/base"
        )
        self.urdf_vis.update_cfg(self.YUMI_REST_POSE)
        
        # Add ground grid
        self.server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)
        
    def _setup_gui(self):
        """Setup GUI elements."""
        # Add timing display
        self.timing_handle = self.server.gui.add_number("Time (ms)", 0.01, disabled=True)
        
        # Add gizmo size control
        self.tf_size_handle = self.server.gui.add_slider(
            "Gizmo size", min=0.01, max=0.4, step=0.01, initial_value=0.2
        )
        
        # Add solver controls
        self.solver_type_handle = self.server.gui.add_dropdown(
            "Solver type",
            ("conjugate_gradient", "dense_cholesky", "cholmod"),
            initial_value="conjugate_gradient",
        )
        self.smooth_handle = self.server.gui.add_checkbox("Smooth", initial_value=True)
        
        # Add manipulability controls
        with self.server.gui.add_folder("Manipulability") as manip_folder:
            manip_folder.expand_by_default = False
            self.manipulability_weight_handle = self.server.gui.add_slider(
                "weight", 0.0, 0.01, 0.001, 0.00
            )
            self.manipulability_cost_handle = self.server.gui.add_number(
                "Yoshikawa index", 0.001, disabled=True
            )
        
        # Add DoF freezing controls
        self._setup_dof_controls()
        
    def _setup_dof_controls(self):
        """Setup controls for freezing degrees of freedom."""
        self.base_dof_handles = []
        self.target_dof_handles = []
        
        with self.server.gui.add_folder("T_base_world") as T_base_world_folder:
            T_base_world_folder.expand_by_default = False
            for dof in ["x", "y", "z", "rx", "ry", "rz"]:
                self.base_dof_handles.append(
                    self.server.gui.add_checkbox(f"Freeze {dof}", initial_value=True)
                )
                
        with self.server.gui.add_folder("Target pose DoF") as target_pose_dof_folder:
            target_pose_dof_folder.expand_by_default = False
            for dof in ["x", "y", "z", "rx", "ry", "rz"]:
                self.target_dof_handles.append(
                    self.server.gui.add_checkbox(f"Freeze {dof}", initial_value=True)
                )
                
    def _setup_transform_handles(self):
        """Setup transform handles for end effectors for each environment."""
        self.transform_handles = {}
        
        for env in range(self.num_batches):
            self.transform_handles[env] = {
                'left': TransformHandle(
                    frame=self.server.scene.add_frame(
                        f"tf_left_env{env}",
                        axes_length=0.5 * self.tf_size_handle.value,
                        axes_radius=0.01 * self.tf_size_handle.value,
                        origin_radius=0.1 * self.tf_size_handle.value,
                    ),
                    control=self.server.scene.add_transform_controls(
                        f"target_left_env{env}",
                        scale=self.tf_size_handle.value
                    )
                ),
                'right': TransformHandle(
                    frame=self.server.scene.add_frame(
                        f"tf_right_env{env}",
                        axes_length=0.5 * self.tf_size_handle.value,
                        axes_radius=0.01 * self.tf_size_handle.value,
                        origin_radius=0.1 * self.tf_size_handle.value,
                    ),
                    control=self.server.scene.add_transform_controls(
                        f"target_right_env{env}",
                        scale=self.tf_size_handle.value
                    )
                )
            }

        
       # Initialize positions
        base_pose = jnp.array(
            self.base_frame.wxyz.tolist() + self.base_frame.position.tolist()
        )
        
        for env in range(self.num_batches):
            for target_frame_handle, target_name in zip(
                list(self.transform_handles[env].values()), self.target_names
            ):
                target_joint_idx = self.kin.joint_names.index(target_name)
                T_target_world = jaxlie.SE3(base_pose) @ jaxlie.SE3(
                    self.kin.forward_kinematics(self.joints)[target_joint_idx]
                )

                target_frame_handle.control.position = onp.array(T_target_world.translation())
                target_frame_handle.control.wxyz = onp.array(T_target_world.rotation().wxyz)
                
                # Offset each environment's handles slightly to make them visible
                curr_pos = onp.array(target_frame_handle.control.position)
                target_frame_handle.control.position = curr_pos + onp.array([0.0005 * env, 0, 0])

        # Update transform handles when size changes
        @self.tf_size_handle.on_update
        def update_tf_size(_):
            for env in range(self.num_batches):
                for handle in self.transform_handles[env].values():
                    if handle.control:
                        handle.control.scale = self.tf_size_handle.value
                    handle.frame.axes_length = 0.5 * self.tf_size_handle.value
                    handle.frame.axes_radius = 0.01 * self.tf_size_handle.value
                    handle.frame.origin_radius = 0.1 * self.tf_size_handle.value
                
    def get_freeze_masks(self):
        """Get DoF freeze masks for base and targets."""
        base_mask = jnp.array([h.value for h in self.base_dof_handles]).astype(jnp.float32)
        target_mask = jnp.array([h.value for h in self.target_dof_handles]).astype(jnp.float32)
        return base_mask, target_mask
    
    def update_visualization(self):
        """Update visualization with current state."""
        # Update base frame
        self.base_frame.position = onp.array(self.base_pose.translation())
        self.base_frame.wxyz = onp.array(self.base_pose.rotation().wxyz)
        
        # Update robot configuration
        self.urdf_vis.update_cfg(onp.array(self.joints))
        
        # Update end-effector frames
        target_joint_indices = {
            'left': self.kin.joint_names.index(self.target_names[0]),
            'right': self.kin.joint_names.index(self.target_names[1])
        }
        
        # Only update the currently selected environment
        current_env = int(self.env_selector.value)
        
        for side, idx in target_joint_indices.items():
            T_target_world = self.base_pose @ jaxlie.SE3(
                self.kin.forward_kinematics(self.joints)[idx]
            )
            self.transform_handles[current_env][side].frame.position = onp.array(T_target_world.translation())
            self.transform_handles[current_env][side].frame.wxyz = onp.array(T_target_world.rotation().wxyz)
            
    def solve_ik(self):
        """Solve inverse kinematics for all environments."""
        # Initialize list to store target poses for all environments
        target_poses_list = []
        
        # For each environment, get both left and right target poses
        for env in range(self.num_batches):
            env_poses = [
                jaxlie.SE3(jnp.array([*tf_handle.control.wxyz, *tf_handle.control.position]))
                for tf_handle in self.transform_handles[env].values()
            ]
            target_poses_list.append([pose.wxyz_xyz for pose in env_poses])
        
        # Stack into shape (num_batches, num_targets=2, 7)
        target_poses = jaxlie.SE3(jnp.array(target_poses_list))
        
        # Create batched initial poses (num_batches, num_joints)
        if self.smooth:
            if hasattr(self, "joints_all"):
                initial_poses = self.joints_all
            else:
                initial_poses = jnp.tile(self.rest_pose[None, :], (self.num_batches, 1))
            joint_vel_weight = self.limit_weight
        else:
            if hasattr(self, "joints_all"):
                initial_poses = self.joints_all
            else:
                initial_poses = jnp.tile(self.rest_pose[None, :], (self.num_batches, 1))
            joint_vel_weight = 0.0
        
        target_indices = jnp.array([
            self.kin.joint_names.index(self.target_names[0]),
            self.kin.joint_names.index(self.target_names[1]),
        ])
        
        ik_weight = jnp.array([self.pos_weight] * 3 + [self.rot_weight] * 3)
        ik_weight = ik_weight * self.target_mask
        
        # Solve IK for all environments
        # if not self.has_jitted:
        start_time = time.time()
            
        self.base_pose, self.joints_all = solve_ik_batched(
            self.kin,
            target_poses,
            target_indices,
            initial_poses,  # Now batched
            self.JointVar,
            ik_weight,
            ConstrainedSE3Var=self.ConstrainedSE3Var,
            rest_weight=self.rest_weight,
            limit_weight=self.limit_weight,
            joint_vel_weight=joint_vel_weight,
            use_manipulability=(self.manipulability_weight > 0),
            manipulability_weight=self.manipulability_weight,
            solver_type=self.solver_type_handle.value,
            num_batches=self.num_batches,
        )

        # Update timing
        jax.block_until_ready((self.base_pose, self.joints_all))
        if not self.has_jitted:
            self.timing_handle.value = (time.time() - start_time) * 1000
            logger.info("JIT compile + running took {} ms.", self.timing_handle.value)
            self.has_jitted = True
        else:
            self.timing_handle.value = (time.time() - start_time) * 1000
        
        # Get current environment's solution
        current_env = int(self.env_selector.value)
        self.joints = self.joints_all[current_env]
        
    def run(self):
        """Main run loop."""
        while True:
            self.solve_ik()
            self.update_visualization()
            
if __name__ == "__main__":
    yumi_interface = YuMiBaseInterface()
    yumi_interface.run()