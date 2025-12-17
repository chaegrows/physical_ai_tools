#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# GR00T Multi-Camera Inference ROS2 Node
#
# This node subscribes to 3 camera images and joint states, runs GR00T inference,
# and publishes predicted actions. Designed to work with ros2 bag playback.
#
# Camera Inputs:
# - video.ego_view: Head camera
# - video.cam_wrist_left: Left wrist camera
# - video.cam_wrist_right: Right wrist camera
#
# Features:
# - Supports /use_sim_time for ros2 bag playback
# - Waits for ALL required messages before inference
# - Runs inference at configurable rate (default: 10 Hz)
# - Prints predicted actions to console
# - Dry-run mode (no publishing)
# - Clear status logging for debugging

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2
from dataclasses import dataclass
from collections import deque
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration

# Add Isaac-GR00T to path
ISAAC_GROOT_PATH = Path("/workspace/Isaac-GR00T")
if ISAAC_GROOT_PATH.exists() and str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))


@dataclass
class JointConfig:
    """Configuration for joint ordering and mapping."""
    # Based on ffw_sg2_rev1_config.yaml
    LEFT_ARM_JOINTS = [
        "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
        "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
    ]
    LEFT_GRIPPER_JOINTS = ["gripper_l_joint1"]
    RIGHT_ARM_JOINTS = [
        "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
        "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
    ]
    RIGHT_GRIPPER_JOINTS = ["gripper_r_joint1"]


@dataclass
class CameraConfig:
    """Configuration for camera topics."""
    # Based on ffw_sg2_rev1_config.yaml
    EGO_VIEW_TOPIC = "/zed/zed_node/left/image_rect_color/compressed"
    WRIST_LEFT_TOPIC = "/camera_left/camera_left/color/image_rect_raw/compressed"
    WRIST_RIGHT_TOPIC = "/camera_right/camera_right/color/image_rect_raw/compressed"


class Gr00tMultiCamInferenceNode(Node):
    """ROS2 node for running GR00T multi-camera inference.
    
    This node:
    1. Subscribes to 3 camera images and joint states
    2. Buffers the latest valid messages
    3. Waits for ALL required data before running inference
    4. Runs inference at a fixed rate
    5. Prints and optionally publishes predicted actions
    
    Topics Subscribed:
    - /zed/zed_node/left/image_rect_color/compressed (ego_view)
    - /camera_left/camera_left/color/image_rect_raw/compressed (cam_wrist_left)
    - /camera_right/camera_right/color/image_rect_raw/compressed (cam_wrist_right)
    - /joint_states (JointState)
    
    Topics Published (when not in dry-run mode):
    - /gr00t/predicted_action (JointTrajectory)
    - /leader/joint_trajectory_command_broadcaster_left/joint_trajectory
    - /leader/joint_trajectory_command_broadcaster_right/joint_trajectory
    """
    
    def __init__(self, parameter_overrides: Optional[list] = None):
        super().__init__(
            "gr00t_multicam_inference_node",
            parameter_overrides=parameter_overrides
        )
        
        # Declare parameters
        self._declare_parameters()
        
        # Get parameters
        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.embodiment_tag = self.get_parameter("embodiment_tag").value
        self.device = self.get_parameter("device").value
        self.inference_rate = self.get_parameter("inference_rate").value
        self.dry_run = self.get_parameter("dry_run").value
        self.task_instruction = self.get_parameter("task_instruction").value
        
        # Camera topics
        self.ego_view_topic = self.get_parameter("ego_view_topic").value
        self.wrist_left_topic = self.get_parameter("wrist_left_topic").value
        self.wrist_right_topic = self.get_parameter("wrist_right_topic").value
        
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.action_topic = self.get_parameter("action_topic").value
        self.print_rate = self.get_parameter("print_rate").value
        
        # Initialize state - image buffers
        self.latest_images: Dict[str, Optional[np.ndarray]] = {
            "ego_view": None,
            "cam_wrist_left": None,
            "cam_wrist_right": None,
        }
        self.latest_image_stamps: Dict[str, Optional[Time]] = {
            "ego_view": None,
            "cam_wrist_left": None,
            "cam_wrist_right": None,
        }
        
        self.latest_joint_state: Optional[JointState] = None
        self.latest_joint_state_stamp: Optional[Time] = None
        
        self.inference_count = 0
        self.last_print_time = time.time()
        self.inference_times: deque = deque(maxlen=100)
        
        # Status tracking for debugging
        self.last_status_print = time.time()
        self.status_print_interval = 5.0  # Print status every 5 seconds
        
        # Joint configuration
        self.joint_config = JointConfig()
        self.camera_config = CameraConfig()
        
        # QoS for subscriptions (compatible with ros2 bag)
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        # Initialize subscribers
        self._init_subscribers()
        
        # Initialize publisher (if not dry-run)
        if not self.dry_run:
            self._init_publishers()
        else:
            self.get_logger().info("Dry-run mode: Actions will NOT be published")
        
        # Load GR00T model
        self._load_model()
        
        # Create inference timer
        self.inference_timer = self.create_timer(
            1.0 / self.inference_rate,
            self._inference_callback
        )
        
        self._print_startup_info()
    
    def _declare_parameters(self):
        """Declare ROS2 parameters with defaults."""
        # Model parameters
        self.declare_parameter(
            "checkpoint_path",
            "/workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000"
        )
        self.declare_parameter("embodiment_tag", "new_embodiment")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("inference_rate", 10.0)  # Hz
        self.declare_parameter("dry_run", True)  # Default to dry-run for safety
        self.declare_parameter("task_instruction", "Execute the task")
        
        # Camera topics (based on ffw_sg2_rev1_config.yaml)
        self.declare_parameter(
            "ego_view_topic",
            "/zed/zed_node/left/image_rect_color/compressed"
        )
        self.declare_parameter(
            "wrist_left_topic",
            "/camera_left/camera_left/color/image_rect_raw/compressed"
        )
        self.declare_parameter(
            "wrist_right_topic",
            "/camera_right/camera_right/color/image_rect_raw/compressed"
        )
        
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("action_topic", "/gr00t/predicted_action")
        self.declare_parameter("print_rate", 1.0)  # Print at 1 Hz
    
    def _print_startup_info(self):
        """Print startup information."""
        self.get_logger().info("=" * 70)
        self.get_logger().info("GR00T Multi-Camera Inference Node Started")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"  Checkpoint: {self.checkpoint_path}")
        self.get_logger().info(f"  Embodiment: {self.embodiment_tag}")
        self.get_logger().info(f"  Device: {self.device}")
        self.get_logger().info(f"  Inference rate: {self.inference_rate} Hz")
        self.get_logger().info(f"  Dry run: {self.dry_run}")
        self.get_logger().info("-" * 70)
        self.get_logger().info("Camera Topics:")
        self.get_logger().info(f"  ego_view:       {self.ego_view_topic}")
        self.get_logger().info(f"  cam_wrist_left: {self.wrist_left_topic}")
        self.get_logger().info(f"  cam_wrist_right: {self.wrist_right_topic}")
        self.get_logger().info(f"  joint_states:   {self.joint_state_topic}")
        self.get_logger().info("=" * 70)
        self.get_logger().info("Waiting for ALL 3 camera images + joint states...")
    
    def _init_subscribers(self):
        """Initialize ROS2 subscribers for 3 cameras + joint states."""
        # Ego view (head camera)
        self.ego_view_sub = self.create_subscription(
            CompressedImage,
            self.ego_view_topic,
            lambda msg: self._image_callback(msg, "ego_view"),
            self.sensor_qos
        )
        
        # Left wrist camera
        self.wrist_left_sub = self.create_subscription(
            CompressedImage,
            self.wrist_left_topic,
            lambda msg: self._image_callback(msg, "cam_wrist_left"),
            self.sensor_qos
        )
        
        # Right wrist camera
        self.wrist_right_sub = self.create_subscription(
            CompressedImage,
            self.wrist_right_topic,
            lambda msg: self._image_callback(msg, "cam_wrist_right"),
            self.sensor_qos
        )
        
        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            self.sensor_qos
        )
    
    def _init_publishers(self):
        """Initialize ROS2 publishers."""
        # Debug/monitoring topic
        self.action_pub = self.create_publisher(
            JointTrajectory,
            self.action_topic,
            10
        )
        
        # Robot control topics
        self.left_arm_pub = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            10
        )
        self.right_arm_pub = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            10
        )
        
        self.get_logger().info("Publishers initialized:")
        self.get_logger().info(f"  - Debug: {self.action_topic}")
        self.get_logger().info("  - Left arm: /leader/joint_trajectory_command_broadcaster_left/joint_trajectory")
        self.get_logger().info("  - Right arm: /leader/joint_trajectory_command_broadcaster_right/joint_trajectory")
    
    def _load_model(self):
        """Load the GR00T inference model."""
        try:
            from gr00t_bridge_multi.gr00t_policy import Gr00tMultiCamInferenceWrapper
            
            self.model = Gr00tMultiCamInferenceWrapper(
                checkpoint_path=self.checkpoint_path,
                embodiment_tag=self.embodiment_tag,
                device=self.device,
            )
            self.get_logger().info("GR00T multi-cam model loaded successfully!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load GR00T model: {e}")
            raise
    
    def _image_callback(self, msg: CompressedImage, camera_key: str):
        """Callback for camera image messages."""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Convert BGR to RGB (GR00T expects RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                self.latest_images[camera_key] = image
                self.latest_image_stamps[camera_key] = Time.from_msg(msg.header.stamp)
            else:
                self.get_logger().warn(f"Failed to decode image from {camera_key}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing {camera_key} image: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Callback for joint state messages."""
        self.latest_joint_state = msg
        self.latest_joint_state_stamp = Time.from_msg(msg.header.stamp)
    
    def _extract_joint_positions(
        self,
        joint_state: JointState
    ) -> Dict[str, np.ndarray]:
        """Extract joint positions from JointState message."""
        joint_dict = {
            name: pos
            for name, pos in zip(joint_state.name, joint_state.position)
        }
        
        left_arm = np.array([
            joint_dict.get(name, 0.0)
            for name in self.joint_config.LEFT_ARM_JOINTS
        ])
        left_hand = np.array([
            joint_dict.get(name, 0.0)
            for name in self.joint_config.LEFT_GRIPPER_JOINTS
        ])
        right_arm = np.array([
            joint_dict.get(name, 0.0)
            for name in self.joint_config.RIGHT_ARM_JOINTS
        ])
        right_hand = np.array([
            joint_dict.get(name, 0.0)
            for name in self.joint_config.RIGHT_GRIPPER_JOINTS
        ])
        
        return {
            "left_arm": left_arm,
            "left_hand": left_hand,
            "right_arm": right_arm,
            "right_hand": right_hand,
        }
    
    def _check_data_ready(self) -> bool:
        """Check if all required data is available.
        
        Returns True if all 3 camera images + joint state are available.
        """
        # Check all cameras
        missing_cameras = [
            k for k, v in self.latest_images.items() if v is None
        ]
        
        # Check joint state
        has_joint_state = self.latest_joint_state is not None
        
        # Print status periodically
        current_time = time.time()
        if current_time - self.last_status_print >= self.status_print_interval:
            self._print_data_status(missing_cameras, has_joint_state)
            self.last_status_print = current_time
        
        return len(missing_cameras) == 0 and has_joint_state
    
    def _print_data_status(self, missing_cameras: List[str], has_joint_state: bool):
        """Print current data status for debugging."""
        self.get_logger().info("-" * 50)
        self.get_logger().info("Data Status:")
        
        for camera_key in self.latest_images.keys():
            status = "✓" if self.latest_images[camera_key] is not None else "✗"
            shape = ""
            if self.latest_images[camera_key] is not None:
                shape = f" {self.latest_images[camera_key].shape}"
            self.get_logger().info(f"  {camera_key}: {status}{shape}")
        
        joint_status = "✓" if has_joint_state else "✗"
        self.get_logger().info(f"  joint_states: {joint_status}")
        
        if missing_cameras:
            self.get_logger().warn(f"Missing cameras: {missing_cameras}")
        if not has_joint_state:
            self.get_logger().warn("Missing joint states!")
        
        if not missing_cameras and has_joint_state:
            self.get_logger().info("All data ready! Running inference.")
    
    def _inference_callback(self):
        """Timer callback for running inference."""
        # Check if all data is ready
        if not self._check_data_ready():
            return
        
        # Run inference
        try:
            start_time = time.time()
            
            # Extract state from joint state message
            state = self._extract_joint_positions(self.latest_joint_state)
            
            # Run inference with all 3 cameras
            action = self.model.predict(
                images=self.latest_images.copy(),
                state=state,
                task_instruction=self.task_instruction,
            )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.inference_count += 1
            
            # Print action at configured rate
            current_time = time.time()
            if current_time - self.last_print_time >= 1.0 / self.print_rate:
                self._print_action(action, inference_time)
                self.last_print_time = current_time
            
            # Publish action (if not dry-run)
            if not self.dry_run:
                self._publish_action(action)
                
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _print_action(self, action, inference_time: float):
        """Print predicted action to console."""
        first_action = action.get_first_timestep()
        concat_action = action.get_concatenated_action()
        
        avg_time = np.mean(self.inference_times) if self.inference_times else 0.0
        
        self.get_logger().info("-" * 60)
        self.get_logger().info(f"[Inference #{self.inference_count}] Time: {inference_time*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")
        self.get_logger().info(f"  Left Arm:  [{', '.join([f'{v:.3f}' for v in first_action['left_arm'].flatten()])}]")
        self.get_logger().info(f"  Left Hand: [{first_action['left_hand'].flatten()[0]:.3f}]")
        self.get_logger().info(f"  Right Arm: [{', '.join([f'{v:.3f}' for v in first_action['right_arm'].flatten()])}]")
        self.get_logger().info(f"  Right Hand: [{first_action['right_hand'].flatten()[0]:.3f}]")
        self.get_logger().info(f"  Concatenated: {concat_action[:4]}... (len={len(concat_action)})")
    
    def _publish_action(self, action):
        """Publish predicted action as JointTrajectory."""
        first_action = action.get_first_timestep()
        current_stamp = self.get_clock().now().to_msg()
        
        # ========== LEFT ARM ==========
        left_msg = JointTrajectory()
        left_msg.header = Header()
        left_msg.header.stamp = current_stamp
        left_msg.header.frame_id = ""
        left_msg.joint_names = (
            self.joint_config.LEFT_ARM_JOINTS +
            self.joint_config.LEFT_GRIPPER_JOINTS
        )
        
        left_point = JointTrajectoryPoint()
        left_point.positions = (
            first_action["left_arm"].flatten().tolist() +
            first_action["left_hand"].flatten().tolist()
        )
        left_point.time_from_start = Duration(sec=0, nanosec=100000000)  # 100ms
        left_msg.points = [left_point]
        
        # ========== RIGHT ARM ==========
        right_msg = JointTrajectory()
        right_msg.header = Header()
        right_msg.header.stamp = current_stamp
        right_msg.header.frame_id = ""
        right_msg.joint_names = (
            self.joint_config.RIGHT_ARM_JOINTS +
            self.joint_config.RIGHT_GRIPPER_JOINTS
        )
        
        right_point = JointTrajectoryPoint()
        right_point.positions = (
            first_action["right_arm"].flatten().tolist() +
            first_action["right_hand"].flatten().tolist()
        )
        right_point.time_from_start = Duration(sec=0, nanosec=100000000)
        right_msg.points = [right_point]
        
        # ========== COMBINED (debug) ==========
        combined_msg = JointTrajectory()
        combined_msg.header = Header()
        combined_msg.header.stamp = current_stamp
        combined_msg.header.frame_id = ""
        combined_msg.joint_names = (
            self.joint_config.LEFT_ARM_JOINTS +
            self.joint_config.LEFT_GRIPPER_JOINTS +
            self.joint_config.RIGHT_ARM_JOINTS +
            self.joint_config.RIGHT_GRIPPER_JOINTS
        )
        
        combined_point = JointTrajectoryPoint()
        combined_point.positions = (
            first_action["left_arm"].flatten().tolist() +
            first_action["left_hand"].flatten().tolist() +
            first_action["right_arm"].flatten().tolist() +
            first_action["right_hand"].flatten().tolist()
        )
        combined_point.time_from_start = Duration(sec=0, nanosec=100000000)
        combined_msg.points = [combined_point]
        
        # Publish to all topics
        self.left_arm_pub.publish(left_msg)
        self.right_arm_pub.publish(right_msg)
        self.action_pub.publish(combined_msg)


def main(args=None):
    """Main entry point for the GR00T multi-cam inference node."""
    rclpy.init(args=args)
    
    try:
        node = Gr00tMultiCamInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
