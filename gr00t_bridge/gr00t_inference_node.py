#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# GR00T Inference ROS2 Node
#
# This node subscribes to camera images and joint states, runs GR00T inference,
# and publishes predicted actions. Designed to work with ros2 bag playback.
#
# Features:
# - Supports /use_sim_time for ros2 bag playback
# - Runs inference at configurable rate (default: 10 Hz)
# - Prints predicted actions to console
# - Dry-run mode (no publishing)
# - Time synchronization handling

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
    # Joint names in the order they appear in JointState messages
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
    
    # Head and lift joints (not used for action output in this config)
    HEAD_JOINTS = ["head_joint1", "head_joint2"]
    LIFT_JOINTS = ["lift_joint"]


class Gr00tInferenceNode(Node):
    """ROS2 node for running GR00T inference.
    
    This node:
    1. Subscribes to camera images and joint states
    2. Buffers the latest valid messages
    3. Runs inference at a fixed rate
    4. Prints and optionally publishes predicted actions
    
    Topics Subscribed:
    - /zed/zed_node/left/image_rect_color/compressed (CompressedImage)
    - /joint_states (JointState)
    
    Topics Published (when not in dry-run mode):
    - /gr00t/predicted_action (JointTrajectory)
    """
    
    def __init__(self, parameter_overrides: Optional[list] = None):
        super().__init__(
            "gr00t_inference_node",
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
        self.image_topic = self.get_parameter("image_topic").value
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.action_topic = self.get_parameter("action_topic").value
        self.print_rate = self.get_parameter("print_rate").value
        
        # Initialize state
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_stamp: Optional[Time] = None
        self.latest_joint_state: Optional[JointState] = None
        self.latest_joint_state_stamp: Optional[Time] = None
        
        self.inference_count = 0
        self.last_print_time = time.time()
        self.inference_times: deque = deque(maxlen=100)
        
        # Joint configuration
        self.joint_config = JointConfig()
        
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
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("GR00T Inference Node Started")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Checkpoint: {self.checkpoint_path}")
        self.get_logger().info(f"  Embodiment: {self.embodiment_tag}")
        self.get_logger().info(f"  Device: {self.device}")
        self.get_logger().info(f"  Inference rate: {self.inference_rate} Hz")
        self.get_logger().info(f"  Dry run: {self.dry_run}")
        self.get_logger().info(f"  Image topic: {self.image_topic}")
        self.get_logger().info(f"  Joint state topic: {self.joint_state_topic}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Waiting for image and joint state messages...")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters with defaults."""
        self.declare_parameter(
            "checkpoint_path",
            "/workspace/Isaac-GR00T/results/gr00t/checkpoint-100000"
        )
        self.declare_parameter("embodiment_tag", "new_embodiment")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("inference_rate", 10.0)  # Hz
        self.declare_parameter("dry_run", True)  # Default to dry-run for safety
        self.declare_parameter("task_instruction", "Execute the task")
        self.declare_parameter(
            "image_topic",
            "/zed/zed_node/left/image_rect_color/compressed"
        )
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("action_topic", "/gr00t/predicted_action")
        self.declare_parameter("print_rate", 1.0)  # Print at 1 Hz
    
    def _init_subscribers(self):
        """Initialize ROS2 subscribers."""
        # Image subscriber
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self._image_callback,
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
        """Initialize ROS2 publishers.
        
        For FFW SG2 robot, we need separate publishers for left and right arms.
        The robot expects JointTrajectory messages on specific topics.
        """
        # Debug/monitoring topic (always published)
        self.action_pub = self.create_publisher(
            JointTrajectory,
            self.action_topic,
            10
        )
        
        # Robot control topics (for actual robot movement)
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
            from gr00t_bridge.gr00t_policy import Gr00tInferenceWrapper
            
            self.model = Gr00tInferenceWrapper(
                checkpoint_path=self.checkpoint_path,
                embodiment_tag=self.embodiment_tag,
                device=self.device,
            )
            self.get_logger().info("GR00T model loaded successfully!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load GR00T model: {e}")
            raise
    
    def _image_callback(self, msg: CompressedImage):
        """Callback for camera image messages."""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Convert BGR to RGB (GR00T expects RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                self.latest_image = image
                self.latest_image_stamp = Time.from_msg(msg.header.stamp)
            else:
                self.get_logger().warn("Failed to decode image")
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Callback for joint state messages."""
        self.latest_joint_state = msg
        self.latest_joint_state_stamp = Time.from_msg(msg.header.stamp)
    
    def _extract_joint_positions(
        self,
        joint_state: JointState
    ) -> Dict[str, np.ndarray]:
        """Extract joint positions from JointState message.
        
        Args:
            joint_state: ROS2 JointState message
            
        Returns:
            Dict with keys: left_arm, left_hand, right_arm, right_hand
        """
        # Create joint name to position mapping
        joint_dict = {
            name: pos
            for name, pos in zip(joint_state.name, joint_state.position)
        }
        
        # Extract left arm joints
        left_arm = np.array([
            joint_dict.get(name, 0.0)
            for name in self.joint_config.LEFT_ARM_JOINTS
        ])
        
        # Extract left gripper (hand)
        left_hand = np.array([
            joint_dict.get(name, 0.0)
            for name in self.joint_config.LEFT_GRIPPER_JOINTS
        ])
        
        # Extract right arm joints
        right_arm = np.array([
            joint_dict.get(name, 0.0)
            for name in self.joint_config.RIGHT_ARM_JOINTS
        ])
        
        # Extract right gripper (hand)
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
    
    def _inference_callback(self):
        """Timer callback for running inference."""
        # Check if we have valid data
        if self.latest_image is None:
            if self.inference_count == 0:
                self.get_logger().warn("No image received yet", throttle_duration_sec=5.0)
            return
        
        if self.latest_joint_state is None:
            if self.inference_count == 0:
                self.get_logger().warn("No joint state received yet", throttle_duration_sec=5.0)
            return
        
        # Run inference
        try:
            start_time = time.time()
            
            # Extract state from joint state message
            state = self._extract_joint_positions(self.latest_joint_state)
            
            # Run inference
            action = self.model.predict(
                images={"ego_view": self.latest_image},
                state=state,
                task_instruction=self.task_instruction,
            )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.inference_count += 1
            
            # Get first timestep action
            first_action = action.get_concatenated_action(timestep=0)
            
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
        """Publish predicted action as JointTrajectory to robot control topics."""
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
        right_point.time_from_start = Duration(sec=0, nanosec=100000000)  # 100ms
        right_msg.points = [right_point]
        
        # ========== COMBINED (for debug/monitoring) ==========
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
        self.action_pub.publish(combined_msg)  # Debug topic


def main(args=None):
    """Main entry point for the GR00T inference node."""
    rclpy.init(args=args)
    
    try:
        node = Gr00tInferenceNode()
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
