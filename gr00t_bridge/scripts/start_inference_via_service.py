#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Start GR00T inference using the existing physical_ai_server system
#
# This script calls ROS2 services to:
# 1. Set the robot type
# 2. Start inference with the GR00T model
#
# Usage:
#   python3 start_inference_via_service.py --checkpoint /path/to/checkpoint

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from physical_ai_interfaces.srv import SetRobotType, SendCommand
from physical_ai_interfaces.msg import TaskInfo


class InferenceStarter(Node):
    """Node to start inference via ROS2 services."""
    
    def __init__(self):
        super().__init__('inference_starter')
        
        # Create service clients
        self.set_robot_type_client = self.create_client(
            SetRobotType, '/set_robot_type'
        )
        self.send_command_client = self.create_client(
            SendCommand, '/task/command'
        )
        
        self.get_logger().info('Waiting for services...')
        
    def wait_for_services(self, timeout_sec: float = 10.0) -> bool:
        """Wait for required services to be available."""
        services = [
            (self.set_robot_type_client, '/set_robot_type'),
            (self.send_command_client, '/task/command'),
        ]
        
        for client, name in services:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(f'Service {name} not available')
                return False
            self.get_logger().info(f'Service {name} is available')
        
        return True
    
    def set_robot_type(self, robot_type: str) -> bool:
        """Set the robot type."""
        request = SetRobotType.Request()
        request.robot_type = robot_type
        
        self.get_logger().info(f'Setting robot type to: {robot_type}')
        
        future = self.set_robot_type_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Robot type set successfully: {response.message}')
                return True
            else:
                self.get_logger().error(f'Failed to set robot type: {response.message}')
                return False
        else:
            self.get_logger().error('Service call timed out')
            return False
    
    def start_inference(
        self,
        policy_path: str,
        fps: int = 10,
        task_instruction: str = 'Execute the task',
        record_mode: bool = False
    ) -> bool:
        """Start inference with the specified policy."""
        request = SendCommand.Request()
        request.command = SendCommand.Request.START_INFERENCE  # 3
        
        task_info = TaskInfo()
        task_info.policy_path = policy_path
        task_info.fps = fps
        task_info.task_instruction = [task_instruction]
        task_info.record_inference_mode = record_mode
        
        request.task_info = task_info
        
        self.get_logger().info(f'Starting inference with policy: {policy_path}')
        
        future = self.send_command_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Inference started: {response.message}')
                return True
            else:
                self.get_logger().error(f'Failed to start inference: {response.message}')
                return False
        else:
            self.get_logger().error('Service call timed out')
            return False
    
    def stop_inference(self) -> bool:
        """Stop inference."""
        request = SendCommand.Request()
        request.command = SendCommand.Request.FINISH  # 5
        
        self.get_logger().info('Stopping inference...')
        
        future = self.send_command_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Inference stopped: {response.message}')
                return True
            else:
                self.get_logger().error(f'Failed to stop: {response.message}')
                return False
        else:
            self.get_logger().error('Service call timed out')
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Start GR00T inference via physical_ai_server'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/workspace/Isaac-GR00T/results/gr00t/checkpoint-100000',
        help='Path to GR00T checkpoint'
    )
    parser.add_argument(
        '--robot-type',
        type=str,
        default='ffw_sg2_rev1',
        help='Robot type'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Inference rate in Hz'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='Execute the task',
        help='Task instruction'
    )
    parser.add_argument(
        '--record',
        action='store_true',
        help='Enable recording during inference'
    )
    parser.add_argument(
        '--stop',
        action='store_true',
        help='Stop inference instead of starting'
    )
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('GR00T Inference Starter (via physical_ai_server)')
    print('=' * 60)
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Robot type: {args.robot_type}')
    print(f'  FPS: {args.fps}')
    print(f'  Task: {args.task}')
    print('=' * 60)
    
    rclpy.init()
    
    try:
        node = InferenceStarter()
        
        if not node.wait_for_services():
            print('\n[ERROR] Required services not available.')
            print('Make sure physical_ai_server is running:')
            print('  ros2 launch physical_ai_server physical_ai_server.launch.py')
            return 1
        
        if args.stop:
            # Stop inference
            if node.stop_inference():
                print('\n[SUCCESS] Inference stopped.')
                return 0
            else:
                print('\n[ERROR] Failed to stop inference.')
                return 1
        
        # Set robot type
        if not node.set_robot_type(args.robot_type):
            print('\n[ERROR] Failed to set robot type.')
            return 1
        
        # Wait a bit for the robot type to be applied
        time.sleep(1.0)
        
        # Start inference
        if node.start_inference(
            policy_path=args.checkpoint,
            fps=args.fps,
            task_instruction=args.task,
            record_mode=args.record
        ):
            print('\n[SUCCESS] Inference started!')
            print('The robot should now be moving based on GR00T predictions.')
            print('Press Ctrl+C or run with --stop to stop inference.')
            return 0
        else:
            print('\n[ERROR] Failed to start inference.')
            return 1
            
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted.')
    finally:
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
