#!/usr/bin/env python3
"""
FFW SG2 로봇의 GR00T inference 결과를 Isaac Sim에서 재생하는 스크립트.

사용법:
1. Isaac Sim 실행
2. 이 스크립트 실행
3. GR00T inference 노드 실행 (ros2 bag play와 함께)

터미널 1: ros2 bag play 123/ --clock -l
터미널 2: python3 scripts/run_gr00t_inference_offline.py --checkpoint ... --publish --use-sim-time
터미널 3: (Isaac Sim) python ffw_sg2_inference_replay.py
"""

import os
import sys

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import rclpy
from isaac_utils import ReplayWorld

USD_PATH = os.path.join(
    os.path.dirname(__file__), 
    "assets/ai_worker/FFW_SG2.usd"  
)

# 시뮬레이션 설정
SIM_CONFIG = {
    'isaac_fps': 30,
    
    'trajectory_control': {
        'articulation_prim_path': '/Root/ffw_sg2_follower',  # from warning: /Root/ffw_sg2_follower/...
        
        # GR00T inference가 publish하는 토픽
        'left_topic': '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory',
        'right_topic': '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory',
        
        # Isaac Sim 
        # 기본값은 FFW SG2 표준 이름 사용
        'isaac_left_joints': [
            'arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3', 'arm_l_joint4',
            'arm_l_joint5', 'arm_l_joint6', 'arm_l_joint7', 'gripper_l_joint1'
        ],
        'isaac_right_joints': [
            'arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3', 'arm_r_joint4',
            'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7', 'gripper_r_joint1'
        ],
    },
}


def main():
    # ROS2 초기화
    rclpy.init()
    
    print("=" * 60)
    print("FFW SG2 GR00T Inference Replay")
    print("=" * 60)
    print(f"USD Path: {USD_PATH}")
    print(f"Articulation: {SIM_CONFIG['trajectory_control']['articulation_prim_path']}")
    print("=" * 60)
    print("\nSubscribing to:")
    print(f"  - {SIM_CONFIG['trajectory_control']['left_topic']}")
    print(f"  - {SIM_CONFIG['trajectory_control']['right_topic']}")
    print("\nRun GR00T inference in another terminal:")
    print("  python3 scripts/run_gr00t_inference_offline.py --checkpoint ... --publish --use-sim-time")
    print("=" * 60)
    
    # 시뮬레이션 시작
    replay = ReplayWorld(
        simulation_app=simulation_app,
        env_path=USD_PATH,
        sim_ros_comm_params=SIM_CONFIG
    )
    
    # 메인 루프 실행
    replay.play()


if __name__ == "__main__":
    main()
