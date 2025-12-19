#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Offline inference script for GR00T MULTI-CAMERA model testing with ros2 bag playback
#
# Usage:
#   Terminal 1 (inside Docker container):
#     ros2 bag play /path/to/rosbag --clock --exclude-topics /leader/joint_trajectory_command_broadcaster_left/joint_trajectory /leader/joint_trajectory_command_broadcaster_right/joint_trajectory
#
#   Terminal 2 (inside Docker container):
#     python3 scripts/run_gr00t_inference_offline.py \
#         --checkpoint /workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000 \
#         --dry-run
#
# This script runs inference in dry-run mode by default (no publishing).
# Use --publish to enable action publishing.

import argparse
import sys
from pathlib import Path

# Add Isaac-GR00T and gr00t_bridge_multi to path
ISAAC_GROOT_PATH = Path("/workspace/Isaac-GR00T")
if ISAAC_GROOT_PATH.exists() and str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))

# Add gr00t_bridge_multi parent to path
GROOT_BRIDGE_PATH = Path(__file__).parent.parent.parent
if str(GROOT_BRIDGE_PATH) not in sys.path:
    sys.path.insert(0, str(GROOT_BRIDGE_PATH))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GR00T multi-camera inference with ros2 bag playback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run mode (no publishing, just print actions)
  python3 run_gr00t_inference_offline.py --dry-run

  # With action publishing
  python3 run_gr00t_inference_offline.py --publish

  # Custom checkpoint path
  python3 run_gr00t_inference_offline.py \\
      --checkpoint /workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000 \\
      --dry-run

  # With custom camera topics
  python3 run_gr00t_inference_offline.py \\
      --ego-view-topic /zed/zed_node/left/image_rect_color/compressed \\
      --wrist-left-topic /camera_left/camera_left/color/image_rect_raw/compressed \\
      --wrist-right-topic /camera_right/camera_right/color/image_rect_raw/compressed \\
      --dry-run

  # With use_sim_time for ros2 bag sync
  python3 run_gr00t_inference_offline.py \\
      --use-sim-time \\
      --dry-run

ros2 bag play command:
  ros2 bag play /path/to/bag --clock --exclude-topics \\
      /leader/joint_trajectory_command_broadcaster_left/joint_trajectory \\
      /leader/joint_trajectory_command_broadcaster_right/joint_trajectory
"""
    )
    
    # Model parameters
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000",
        help="Path to GR00T multi-cam checkpoint directory"
    )
    parser.add_argument(
        "--embodiment",
        type=str,
        default="new_embodiment",
        help="Embodiment tag (must match training)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Inference rate in Hz"
    )
    
    # Mode parameters
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry-run mode (no publishing, just print)"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        default=False,
        help="Enable action publishing (default if --dry-run not specified)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Execute the task",
        help="Task instruction for the model"
    )
    
    # Camera topic parameters
    parser.add_argument(
        "--ego-view-topic",
        type=str,
        default="/zed/zed_node/left/image_rect_color/compressed",
        help="Ego view (head camera) topic"
    )
    parser.add_argument(
        "--wrist-left-topic",
        type=str,
        default="/camera_left/camera_left/color/image_rect_raw/compressed",
        help="Left wrist camera topic"
    )
    parser.add_argument(
        "--wrist-right-topic",
        type=str,
        default="/camera_right/camera_right/color/image_rect_raw/compressed",
        help="Right wrist camera topic"
    )
    parser.add_argument(
        "--joint-topic",
        type=str,
        default="/joint_states",
        help="Joint state topic to subscribe"
    )
    parser.add_argument(
        "--action-topic",
        type=str,
        default="/gr00t/predicted_action",
        help="Action topic to publish"
    )
    
    # Other parameters
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        help="Use simulation time (for ros2 bag playback with --clock)"
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=1.0,
        help="Rate at which to print actions (Hz)"
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="Velocity scale factor (1.0=normal, 0.5=half speed, 2.0=double speed)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at: {checkpoint_path}")
        print("Please verify the path is correct.")
        print("")
        print("Expected path inside Docker:")
        print("  /workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000")
        sys.exit(1)
    
    # Determine dry-run mode
    if args.publish:
        dry_run = False
    elif args.dry_run:
        dry_run = True
    else:
        dry_run = True
        print("[INFO] Neither --publish nor --dry-run specified. Using dry-run mode for safety.")
    
    print("=" * 70)
    print("GR00T Multi-Camera Offline Inference")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Embodiment: {args.embodiment}")
    print(f"  Device: {args.device}")
    print(f"  Inference rate: {args.rate} Hz")
    print(f"  Dry-run mode: {dry_run}")
    print(f"  Use sim time: {args.use_sim_time}")
    print("-" * 70)
    print("Camera Topics:")
    print(f"  ego_view:       {args.ego_view_topic}")
    print(f"  cam_wrist_left: {args.wrist_left_topic}")
    print(f"  cam_wrist_right: {args.wrist_right_topic}")
    print(f"  joint_states:   {args.joint_topic}")
    if not dry_run:
        print(f"  action output:  {args.action_topic}")
        print(f"  velocity scale: {args.velocity_scale}x {'(normal)' if args.velocity_scale == 1.0 else '(slower)' if args.velocity_scale < 1.0 else '(faster)'}")
    print("=" * 70)
    
    # Import ROS2
    import rclpy
    from rclpy.parameter import Parameter
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Import and create the node with parameter overrides
        from gr00t_bridge_multi.gr00t_inference_node import Gr00tMultiCamInferenceNode
        
        # Pass parameters directly via parameter_overrides
        param_overrides = [
            Parameter("checkpoint_path", Parameter.Type.STRING, args.checkpoint),
            Parameter("embodiment_tag", Parameter.Type.STRING, args.embodiment),
            Parameter("device", Parameter.Type.STRING, args.device),
            Parameter("inference_rate", Parameter.Type.DOUBLE, float(args.rate)),
            Parameter("dry_run", Parameter.Type.BOOL, dry_run),
            Parameter("task_instruction", Parameter.Type.STRING, args.task),
            Parameter("ego_view_topic", Parameter.Type.STRING, args.ego_view_topic),
            Parameter("wrist_left_topic", Parameter.Type.STRING, args.wrist_left_topic),
            Parameter("wrist_right_topic", Parameter.Type.STRING, args.wrist_right_topic),
            Parameter("joint_state_topic", Parameter.Type.STRING, args.joint_topic),
            Parameter("action_topic", Parameter.Type.STRING, args.action_topic),
            Parameter("print_rate", Parameter.Type.DOUBLE, float(args.print_rate)),
            Parameter("velocity_scale", Parameter.Type.DOUBLE, float(args.velocity_scale)),
        ]
        
        if args.use_sim_time:
            param_overrides.append(
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            )
        
        # Create the inference node with parameter overrides
        inference_node = Gr00tMultiCamInferenceNode(parameter_overrides=param_overrides)
        
        print("\n[INFO] Node started. Waiting for messages from ALL 3 cameras...")
        print("[INFO] Run 'ros2 bag play /path/to/bag --clock' in another terminal")
        print("[INFO] Press Ctrl+C to stop\n")
        
        # Spin
        rclpy.spin(inference_node)
        
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
