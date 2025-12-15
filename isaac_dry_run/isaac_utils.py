import os
import time
import threading
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Isaac Sim imports
from isaacsim.core.api import SimulationContext, World
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils import extensions, prims, rotations, stage, viewports
from isaacsim.core.api.robots import Robot
from isaacsim.sensors.camera import Camera as IsaacCamera
from isaacsim.core.prims import SingleXFormPrim, XFormPrim

# USD imports
import omni.usd
from pxr import PhysxSchema, UsdGeom, UsdShade, Tf, Usd, UsdPhysics, Gf, Sdf

# ROS2 message imports
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_STATES_TOPIC_NAME = "/joint_states"

class JointStatesToIssacArticulation:
    def __init__(self, node, articulation_prim_name, isaac_joint_names, mf_joint_names):
        self.node = node
        self.isaac_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.moveit_joint_states_sub = node.create_subscription(
            JointState, JOINT_STATES_TOPIC_NAME, self._cb, self.isaac_qos
        )
        self.isaac_joint_names = isaac_joint_names
        self.mf_joint_names = mf_joint_names
        self.articulation_prim_name = articulation_prim_name
        self.last_msg = None
        self.isaac_articulation = None
        self.joint_name_to_index = {}

    def initialize(self):
        try:
            self.isaac_articulation = Robot(
                prim_path=self.articulation_prim_name, 
                name="mf_sim_joint_states_driver"
            )
            self.isaac_articulation.initialize()
            self.joint_name_to_index = {n: i for i, n in enumerate(self.isaac_joint_names)}
        except Exception as e:
            self.node.get_logger().error(f"Error initializing Isaac articulation: {e}")
            self.node.get_logger().error(f'do you have prim following prim? {self.articulation_prim_name}')
            raise

    def _cb(self, msg: JointState):
        self.last_msg = msg

    def step(self):
        if self.last_msg is None or self.isaac_articulation is None:
            return

        try:
            indices, values = self._jointstate_to_indices_and_positions(self.last_msg)
            self.last_msg = None
            
            if indices and len(values) > 0:
                self.isaac_articulation.set_joint_positions(values, joint_indices=indices)
        except Exception as e:
            self.node.get_logger().error(f"Error setting joint positions: {e}")

    def _jointstate_to_indices_and_positions(self, msg: JointState):
        """
        isaac_joint_names: Isaac Sim에서 사용하는 조인트 이름 (예: ['base_0', 'shoulder_0', ...])
        mf_joint_names: ROS JointState 메시지에서 받는 조인트 이름 (예: ['base', 'shoulder', ...])
        
        - mf_joint_names가 있으면: isaac_joint_names[i] <-> mf_joint_names[i]로 순서 매핑
        - mf_joint_names가 없으면: isaac_joint_names를 직접 사용해서 msg에서 찾음
        
        Note: 메시지의 조인트 이름이 'namespace/joint_name' 형식일 경우 suffix만 비교
        """
        indices = []
        values = []
        
        if msg.name and msg.position:
            # Create a dictionary for fast lookup: joint_name -> position
            # Also create a suffix-based lookup for namespaced joint names
            msg_joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
            
            # suffix 기반 lookup 딕셔너리 생성 (예: 'robot1/base' -> 'base')
            msg_joint_suffix_dict = {}
            for name, pos in zip(msg.name, msg.position):
                suffix = name.split('/')[-1]   # 'robot1/base' -> 'base' or robot1_base => base
                suffix = name.split('_')[-1]
                msg_joint_suffix_dict[suffix] = pos
            
            if self.mf_joint_names:
                # mf_joint_names가 있으면 순서대로 매핑
                for i, isaac_joint_name in enumerate(self.isaac_joint_names):
                    if i < len(self.mf_joint_names):
                        mf_joint_name = self.mf_joint_names[i]
                        
                        # 1. 먼저 정확한 이름으로 찾기
                        if mf_joint_name in msg_joint_dict:
                            indices.append(self.isaac_articulation.get_dof_index(isaac_joint_name))
                            values.append(float(msg_joint_dict[mf_joint_name]))
                        # 2. suffix 기반으로 찾기 (namespace 포함된 경우)
                        elif mf_joint_name in msg_joint_suffix_dict:
                            indices.append(self.isaac_articulation.get_dof_index(isaac_joint_name))
                            values.append(float(msg_joint_suffix_dict[mf_joint_name]))
                        else:
                            self.node.get_logger().warn(
                                f"Joint '{mf_joint_name}' (maps to Isaac '{isaac_joint_name}') not found in JointState message. "
                                f"Available joints: {list(msg.name)}", 
                                throttle_duration_sec=5.0
                            )
            else:
                # mf_joint_names가 없으면 isaac_joint_names를 직접 사용
                for isaac_joint_name in self.isaac_joint_names:
                    if isaac_joint_name in msg_joint_dict:
                        indices.append(self.isaac_articulation.get_dof_index(isaac_joint_name))
                        values.append(float(msg_joint_dict[isaac_joint_name]))
                    elif isaac_joint_name in msg_joint_suffix_dict:
                        indices.append(self.isaac_articulation.get_dof_index(isaac_joint_name))
                        values.append(float(msg_joint_suffix_dict[isaac_joint_name]))
                    else:
                        self.node.get_logger().warn(
                            f"Joint '{isaac_joint_name}' not found in JointState message. "
                            f"Available joints: {list(msg.name)}", 
                            throttle_duration_sec=5.0
                        )
            
            if indices:
                return indices, np.array(values, dtype=np.float32)

        return [], np.array([], dtype=np.float32)


class JointTrajectoryToIsaacArticulation:
    """
    GR00T inference에서 나온 JointTrajectory 메시지를 받아 Isaac Sim articulation을 제어하는 클래스.
    
    FFW SG2 로봇 구조:
    - Left arm: arm_l_joint1~7, gripper_l_joint1
    - Right arm: arm_r_joint1~7, gripper_r_joint1
    
    Topics:
    - /leader/joint_trajectory_command_broadcaster_left/joint_trajectory (왼팔)
    - /leader/joint_trajectory_command_broadcaster_right/joint_trajectory (오른팔)
    """
    
    # FFW SG2 기본 조인트 매핑 (ROS topic joint name -> Isaac Sim joint name)
    DEFAULT_LEFT_ARM_JOINTS = [
        "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
        "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1"
    ]
    
    DEFAULT_RIGHT_ARM_JOINTS = [
        "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
        "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1"
    ]
    
    def __init__(
        self, 
        node, 
        articulation_prim_path: str,
        left_topic: str = "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
        right_topic: str = "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
        left_joint_mapping: Optional[Dict[str, str]] = None,
        right_joint_mapping: Optional[Dict[str, str]] = None,
        isaac_left_joints: Optional[List[str]] = None,
        isaac_right_joints: Optional[List[str]] = None,
    ):
        """
        Args:
            node: ROS2 노드
            articulation_prim_path: Isaac Sim articulation prim 경로 (예: "/World/robot")
            left_topic: 왼팔 JointTrajectory 토픽
            right_topic: 오른팔 JointTrajectory 토픽
            left_joint_mapping: {ros_joint_name: isaac_joint_name} 왼팔 매핑 (None이면 기본값 사용)
            right_joint_mapping: {ros_joint_name: isaac_joint_name} 오른팔 매핑 (None이면 기본값 사용)
            isaac_left_joints: Isaac Sim 왼팔 조인트 이름 리스트 (매핑 없이 순서대로 사용할 때)
            isaac_right_joints: Isaac Sim 오른팔 조인트 이름 리스트 (매핑 없이 순서대로 사용할 때)
        """
        self.node = node
        self.articulation_prim_path = articulation_prim_path
        
        # QoS 설정
        self.isaac_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        # 조인트 매핑 설정
        self.isaac_left_joints = isaac_left_joints or self.DEFAULT_LEFT_ARM_JOINTS
        self.isaac_right_joints = isaac_right_joints or self.DEFAULT_RIGHT_ARM_JOINTS
        self.left_joint_mapping = left_joint_mapping  # {ros_name: isaac_name}
        self.right_joint_mapping = right_joint_mapping
        
        # 메시지 버퍼
        self.last_left_msg: Optional[JointTrajectory] = None
        self.last_right_msg: Optional[JointTrajectory] = None
        
        # Isaac articulation
        self.isaac_articulation = None
        
        # 구독자 생성
        self.left_sub = node.create_subscription(
            JointTrajectory, left_topic, self._left_cb, self.isaac_qos
        )
        self.right_sub = node.create_subscription(
            JointTrajectory, right_topic, self._right_cb, self.isaac_qos
        )
        
        self.node.get_logger().info(f"[JointTrajectoryToIsaac] Subscribing to:")
        self.node.get_logger().info(f"  - Left arm: {left_topic}")
        self.node.get_logger().info(f"  - Right arm: {right_topic}")
    
    def initialize(self):
        """Isaac articulation 초기화"""
        try:
            self.isaac_articulation = Robot(
                prim_path=self.articulation_prim_path,
                name="ffw_sg2_trajectory_driver"
            )
            self.isaac_articulation.initialize()
            
            # 사용 가능한 조인트 이름 로깅
            dof_names = self.isaac_articulation.dof_names
            self.node.get_logger().info(f"[JointTrajectoryToIsaac] Articulation initialized: {self.articulation_prim_path}")
            self.node.get_logger().info(f"[JointTrajectoryToIsaac] Available DOFs ({len(dof_names)}): {dof_names}")
            
        except Exception as e:
            self.node.get_logger().error(f"[JointTrajectoryToIsaac] Error initializing articulation: {e}")
            self.node.get_logger().error(f"  Prim path: {self.articulation_prim_path}")
            raise
    
    def _left_cb(self, msg: JointTrajectory):
        """왼팔 JointTrajectory 콜백"""
        self.last_left_msg = msg
    
    def _right_cb(self, msg: JointTrajectory):
        """오른팔 JointTrajectory 콜백"""
        self.last_right_msg = msg
    
    def step(self):
        """매 프레임 호출되어 조인트 위치 업데이트"""
        if self.isaac_articulation is None:
            return
        
        # 왼팔 처리
        if self.last_left_msg is not None:
            self._apply_trajectory(self.last_left_msg, "left")
            self.last_left_msg = None
        
        # 오른팔 처리
        if self.last_right_msg is not None:
            self._apply_trajectory(self.last_right_msg, "right")
            self.last_right_msg = None
    
    def _apply_trajectory(self, msg: JointTrajectory, arm: str):
        """JointTrajectory 메시지를 Isaac Sim에 적용"""
        try:
            if not msg.points:
                return
            
            # 첫 번째 포인트의 positions 사용 (GR00T은 첫 timestep만 보통 사용)
            point = msg.points[0]
            positions = point.positions
            
            if not positions:
                return
            
            # 조인트 이름과 인덱스 매핑
            indices = []
            values = []
            
            if arm == "left":
                isaac_joints = self.isaac_left_joints
                joint_mapping = self.left_joint_mapping
            else:
                isaac_joints = self.isaac_right_joints
                joint_mapping = self.right_joint_mapping
            
            # msg.joint_names가 있는 경우 (이름 기반 매핑)
            if msg.joint_names:
                for i, ros_joint_name in enumerate(msg.joint_names):
                    if i >= len(positions):
                        break
                    
                    # 매핑 사용 또는 직접 이름 사용
                    if joint_mapping and ros_joint_name in joint_mapping:
                        isaac_joint_name = joint_mapping[ros_joint_name]
                    else:
                        isaac_joint_name = ros_joint_name
                    
                    try:
                        idx = self.isaac_articulation.get_dof_index(isaac_joint_name)
                        indices.append(idx)
                        values.append(float(positions[i]))
                    except Exception:
                        # 조인트를 찾지 못하면 스킵
                        pass
            else:
                # joint_names가 없는 경우 순서대로 매핑
                for i, isaac_joint_name in enumerate(isaac_joints):
                    if i >= len(positions):
                        break
                    try:
                        idx = self.isaac_articulation.get_dof_index(isaac_joint_name)
                        indices.append(idx)
                        values.append(float(positions[i]))
                    except Exception:
                        pass
            
            # 조인트 위치 설정
            if indices and values:
                self.isaac_articulation.set_joint_positions(
                    np.array(values, dtype=np.float32),
                    joint_indices=indices
                )
                
        except Exception as e:
            self.node.get_logger().error(
                f"[JointTrajectoryToIsaac] Error applying trajectory ({arm}): {e}",
                throttle_duration_sec=2.0
            )


class IsaacCommunication(Node):
    def __init__(self, **kwargs):
        super().__init__("isaac_communication")
        
        # 시뮬레이션 시간 사용 설정
        self.set_parameters([
            Parameter(name="use_sim_time", value=True)
        ])

        # 데몬 및 실행자 초기화
        self.daemon = {}
        self.my_executor = MultiThreadedExecutor()
        self.th = None
        
        # 각 컴포넌트 초기화
        self._initialize_components(kwargs)

    def _initialize_components(self, kwargs):
        """컴포넌트 초기화 (모듈화)"""
        try:
            # 매니퓰레이터 초기화 (JointState 기반)
            if 'manipulator' in kwargs:
                self._init_manipulator(kwargs['manipulator'])
            
            # GR00T 추론 결과 기반 팔 제어 (JointTrajectory 기반)
            if 'trajectory_control' in kwargs:
                self._init_trajectory_control(kwargs['trajectory_control'])
            
        except Exception as e:
            self.get_logger().error(f"Error initializing components: {e}")
            raise

    def _init_manipulator(self, params):
        """매니퓰레이터 초기화 (JointState 토픽 기반)"""
        isaac_joint_names = params['isaac_joint_names']
        mf_joint_names = params.get('mf_joint_names', None)  # 없으면 None
        manipulator_prim_path = params['manipulator_prim_path']
        self.daemon['manipulator'] = JointStatesToIssacArticulation(
            node=self, 
            articulation_prim_name=manipulator_prim_path, 
            isaac_joint_names=isaac_joint_names,
            mf_joint_names=mf_joint_names
        )

    def _init_trajectory_control(self, params):
        """
        GR00T inference JointTrajectory 기반 팔 제어 초기화.
        
        FFW SG2 로봇용 설정 예시:
        trajectory_control = {
            'articulation_prim_path': '/World/ffw_sg2',
            'left_topic': '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory',
            'right_topic': '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory',
            # 아래는 선택사항 (기본값 사용 가능)
            'isaac_left_joints': ['arm_l_joint1', ..., 'gripper_l_joint1'],
            'isaac_right_joints': ['arm_r_joint1', ..., 'gripper_r_joint1'],
        }
        """
        articulation_prim_path = params['articulation_prim_path']
        left_topic = params.get(
            'left_topic', 
            '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory'
        )
        right_topic = params.get(
            'right_topic', 
            '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory'
        )
        isaac_left_joints = params.get('isaac_left_joints', None)
        isaac_right_joints = params.get('isaac_right_joints', None)
        left_joint_mapping = params.get('left_joint_mapping', None)
        right_joint_mapping = params.get('right_joint_mapping', None)
        
        self.daemon['trajectory_control'] = JointTrajectoryToIsaacArticulation(
            node=self,
            articulation_prim_path=articulation_prim_path,
            left_topic=left_topic,
            right_topic=right_topic,
            left_joint_mapping=left_joint_mapping,
            right_joint_mapping=right_joint_mapping,
            isaac_left_joints=isaac_left_joints,
            isaac_right_joints=isaac_right_joints,
        )
        self.get_logger().info(f"[IsaacCommunication] Trajectory control initialized for {articulation_prim_path}")


    def initialize(self):
        for da in self.daemon.values():
            da.initialize()

    def start_ros2_spin(self):
        self.my_executor.add_node(self)

        def _spin():
            try:
                self.my_executor.spin()
            finally:
                self.my_executor.remove_node(self)
                self.destroy_node()
                rclpy.shutdown()

        self.th = threading.Thread(target=_spin, name="ROS2SpinThread", daemon=True)
        self.th.start()

    def step(self):
        for da in self.daemon.values():
            da.step()

    def destroy(self):
        if self.th:
            self.th.join(timeout=1.0)
            self.th = None
        for da in self.daemon.values():
            da.destroy()
        self.my_executor.shutdown()


class ReplayWorld:
    def __init__(self, simulation_app, env_path, sim_ros_comm_params=None):
        self.env_path = env_path
        self.simulation_app = simulation_app
        self.sim_ros_comm_params = sim_ros_comm_params or {}
        self.sim_ros_comm = None
        self.my_world = None
        self.startEnvs()

    def startEnvs(self):
        """환경 초기화 및 시작"""
        try:
            # 스테이지 열기
            open_stage(self.env_path)
            
            # 월드 생성
            isaac_fps = self.sim_ros_comm_params.get('isaac_fps', 30)
            self.my_world = World(
                stage_units_in_meters=1.0,
                rendering_dt=1.0/isaac_fps
            )
                            
            # 물리 시뮬레이션 설정
            self.setup_physics_scene()

            # 월드 시작
            self.my_world.play()
            self.my_world.step(render=True)
            self.my_world.step(render=True)
            
            # ROS2 통신 초기화
            self.sim_ros_comm = IsaacCommunication(**self.sim_ros_comm_params)
            self.sim_ros_comm.initialize()
            self.sim_ros_comm.start_ros2_spin()
            
        except Exception as e:
            # 로거가 없으므로 print 사용 (시뮬레이션 초기화 단계)
            print(f"Error starting environment: {e}")
            raise

    def setup_physics_scene(self):
        """물리 시뮬레이션 설정 최적화"""
        try:
            stage = omni.usd.get_context().get_stage()
            physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
            
            if physxSceneAPI:
                # 로거가 없으므로 print 사용 (시뮬레이션 초기화 단계)
                print("physicsScene updating...")
                
                # GPU 가속 설정
                physxSceneAPI.GetEnableGPUDynamicsAttr().Set(True)
                physxSceneAPI.GetEnableStabilizationAttr().Set(True)
                physxSceneAPI.GetEnableCCDAttr().Set(True)
                physxSceneAPI.GetSolverTypeAttr().Set("TGS")
                
                # 프레임 레이트 설정
                isaac_fps = self.sim_ros_comm_params.get('isaac_fps', 30)
                physxSceneAPI.GetTimeStepsPerSecondAttr().Set(isaac_fps)
                
                # GPU 메모리 설정 (최적화)
                gpu_capacity = 2 * 1024 * 1024
                physxSceneAPI.GetGpuTotalAggregatePairsCapacityAttr().Set(gpu_capacity)
                physxSceneAPI.GetGpuFoundLostAggregatePairsCapacityAttr().Set(gpu_capacity)
                
                print("physicsScene updated!")
            else:
                print("Warning: physicsScene not found!")

            # 타임라인 설정
            self._setup_timeline()
            
        except Exception as e:
            # 로거가 없으므로 print 사용 (시뮬레이션 초기화 단계)
            print(f"Error setting up physics scene: {e}")

    def _setup_timeline(self):
        """타임라인 설정"""
        try:
            physics_rate = self.sim_ros_comm_params.get('isaac_fps', 30)
            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
            
            stage = omni.usd.get_context().get_stage()
            stage.SetTimeCodesPerSecond(physics_rate)
            timeline.set_target_framerate(physics_rate)
            timeline.play()
            
        except Exception as e:
            # 로거가 없으므로 print 사용 (시뮬레이션 초기화 단계)
            print(f"Error setting up timeline: {e}")

    def play(self):
        """시뮬레이션 실행 루프"""
        reset_needed = False
        
        try:
            while self.simulation_app.is_running():
                if self.my_world.is_playing():
                    if reset_needed:
                        self.startEnvs()
                        reset_needed = False
                    self.sim_ros_comm.step()
                elif self.my_world.is_stopped():
                    reset_needed = True

                self.my_world.step(render=True)
                
        except KeyboardInterrupt:
            # 로거가 없으므로 print 사용 (시뮬레이션 초기화 단계)
            print("Simulation interrupted by user")
        except Exception as e:
            # 로거가 없으므로 print 사용 (시뮬레이션 초기화 단계)
            print(f"Error in simulation loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        try:
            if self.sim_ros_comm:
                self.sim_ros_comm.destroy()
            if self.simulation_app:
                self.simulation_app.close()
        except Exception as e:
            # 로거가 없으므로 print 사용 (시뮬레이션 초기화 단계)
            print(f"Error during cleanup: {e}")
