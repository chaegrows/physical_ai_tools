# GR00T Bridge - ROS2 Inference for Isaac-GR00T

This package provides a ROS2 inference wrapper for running finetuned Isaac-GR00T models on the FFW SG2 Rev1 dual-arm robot.

## Overview

The GR00T Bridge enables running GR00T model inference in a ROS2 environment, designed to work with:
- Real-time robot control
- ros2 bag playback for testing and validation

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │     gr00t_inference_node.py         │
                    │                                     │
  ┌──────────┐      │  ┌───────────────────────────────┐  │
  │ Camera   │──────┼──│  Image Subscriber             │  │
  │ Topic    │      │  │  (/zed/.../compressed)        │  │
  └──────────┘      │  └───────────────────────────────┘  │
                    │              │                      │
  ┌──────────┐      │  ┌───────────▼───────────────────┐  │      ┌──────────────┐
  │ Joint    │──────┼──│  State Extractor              │──┼─────▶│ Gr00tPolicy  │
  │ States   │      │  │  (JointState → state dict)    │  │      │  (CUDA)      │
  └──────────┘      │  └───────────────────────────────┘  │      └──────┬───────┘
                    │              │                      │             │
                    │  ┌───────────▼───────────────────┐  │             │
                    │  │  Action Publisher             │◀─┼─────────────┘
                    │  │  (/gr00t/predicted_action)    │  │
                    └──┴───────────────────────────────┴──┘
```

## Quick Start

### 1. Prerequisites

- Docker container running `robotis/physical-ai-server:latest`
- Isaac-GR00T available at `/workspace/Isaac-GR00T`
- Finetuned GR00T checkpoint

### 2. Inside the Docker Container

```bash
# Navigate to physical_ai_tools
cd /root/ros2_ws/src/physical_ai_tools

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Add gr00t_bridge to Python path
export PYTHONPATH=$PYTHONPATH:/root/ros2_ws/src/physical_ai_tools
```

### 3. Test Model Loading

```bash
# Test that the model loads correctly
python3 -c "
from gr00t_bridge.gr00t_policy import Gr00tInferenceWrapper

wrapper = Gr00tInferenceWrapper(
    checkpoint_path='/workspace/Isaac-GR00T/results/gr00t/checkpoint-100000',
    embodiment_tag='ffw_sg2_rev1',
    device='cuda'
)
print('Model loaded successfully!')
"
```

### 4. Run Inference with ros2 bag

**Terminal 1: Play ros2 bag**
```bash
ros2 bag play /path/to/your/rosbag --clock
```

**Terminal 2: Run inference node**
```bash
cd /root/ros2_ws/src/physical_ai_tools/gr00t_bridge

# Dry-run mode (recommended for testing)
python3 scripts/run_gr00t_inference_offline.py \
    --checkpoint /workspace/Isaac-GR00T/results/gr00t/checkpoint-100000 \
    --use-sim-time \
    --dry-run

# With action publishing
python3 scripts/run_gr00t_inference_offline.py \
    --checkpoint /workspace/Isaac-GR00T/results/gr00t/checkpoint-100000 \
    --use-sim-time \
    --publish
```

## Configuration

### Topics

**Subscribed (Input):**
| Topic | Type | Description |
|-------|------|-------------|
| `/zed/zed_node/left/image_rect_color/compressed` | `sensor_msgs/CompressedImage` | Head camera image (ego_view) |
| `/joint_states` | `sensor_msgs/JointState` | Robot joint positions |

**Published (Output) - when using `--publish`:**
| Topic | Type | Description |
|-------|------|-------------|
| `/leader/joint_trajectory_command_broadcaster_left/joint_trajectory` | `JointTrajectory` | Left arm control (8 DOF) |
| `/leader/joint_trajectory_command_broadcaster_right/joint_trajectory` | `JointTrajectory` | Right arm control (8 DOF) |
| `/gr00t/predicted_action` | `JointTrajectory` | Combined debug output (16 DOF) |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_path` | `/workspace/.../checkpoint-100000` | Path to model checkpoint |
| `embodiment_tag` | `ffw_sg2_rev1` | Embodiment identifier |
| `device` | `cuda` | Inference device |
| `inference_rate` | `10.0` | Inference rate in Hz |
| `dry_run` | `true` | If true, only prints actions |
| `task_instruction` | `Execute the task` | Language instruction |
| `print_rate` | `1.0` | Rate to print actions in Hz |

## State/Action Mapping

### State Keys (Model Expects)
| Key | Dimension | Description |
|-----|-----------|-------------|
| `state.left_arm` | 7 | Left arm joint positions |
| `state.left_hand` | 1 | Left gripper position |
| `state.right_arm` | 7 | Right arm joint positions |
| `state.right_hand` | 1 | Right gripper position |

### Action Keys (Model Outputs)
| Key | Dimension | Description |
|-----|-----------|-------------|
| `action.left_arm` | 7 | Left arm target positions |
| `action.left_hand` | 1 | Left gripper target |
| `action.right_arm` | 7 | Right arm target positions |
| `action.right_hand` | 1 | Right gripper target |

### Joint Order (from config)
```yaml
left_arm:
  - arm_l_joint1..7

left_hand:
  - gripper_l_joint1

right_arm:
  - arm_r_joint1..7

right_hand:
  - gripper_r_joint1
```

## Troubleshooting

### CUDA out of memory
```bash
# Try with CPU
python3 scripts/run_gr00t_inference_offline.py --device cpu --dry-run
```

### Model not found
```
ERROR: Checkpoint not found at: /workspace/Isaac-GR00T/results/gr00t/checkpoint-100000
```
Verify the checkpoint path exists and contains:
- `config.json`
- `experiment_cfg/metadata.json`
- `model.safetensors` or `model-*.safetensors`

### No messages received
```
[WARN] No image received yet
[WARN] No joint state received yet
```
1. Check ros2 bag is playing: `ros2 bag info /path/to/bag`
2. Check topics: `ros2 topic list`
3. Verify topic names match your configuration

### Dimension mismatch
```
ValueError: State left_arm has 6 dims, expected 7
```
The robot joint state doesn't match expected dimensions. Check that:
- The JointState message contains all expected joint names
- Joint names match `JointConfig` in `gr00t_inference_node.py`

### Import errors
```
ModuleNotFoundError: No module named 'gr00t_bridge'
```
Add to Python path:
```bash
export PYTHONPATH=$PYTHONPATH:/root/ros2_ws/src/physical_ai_tools
```

## File Structure

```
gr00t_bridge/
├── __init__.py                    # Package init
├── gr00t_policy.py                # GR00T inference wrapper
├── ffw_sg2_inference_config.py    # Data config for FFW SG2
├── gr00t_inference_node.py        # ROS2 inference node
├── scripts/
│   └── run_gr00t_inference_offline.py  # CLI script
└── README.md                      # This file
```

## Design Notes

### Single Camera (Current Implementation)
The current implementation uses only the head camera (`video.ego_view`) as the model was trained with this configuration. This matches the `FFWSG2DataConfig` used during training.

### Future Multi-Camera Extension
To extend to multi-camera inference:
1. Update `ffw_sg2_inference_config.py` to include additional cameras
2. Add subscribers for wrist camera topics in `gr00t_inference_node.py`
3. Use `FFWSG2MultiCamInferenceConfig` (see `ffw_sg2_data_config.py` in Isaac-GR00T)

The architecture is designed to make this a configuration change rather than a refactor.

## 기존 physical_ai_server 시스템 통합

### 방법 1: Python 스크립트 (권장)

```bash
# Docker 컨테이너 내부에서
cd /root/ros2_ws/src/physical_ai_tools/gr00t_bridge

# Inference 시작
python3 scripts/start_inference_via_service.py \
    --checkpoint /workspace/Isaac-GR00T/results/gr00t/checkpoint-100000 \
    --robot-type ffw_sg2_rev1 \
    --fps 10

# Inference 중지
python3 scripts/start_inference_via_service.py --stop
```

### 방법 2: ROS2 서비스 직접 호출

```bash
# 로봇 타입 설정
ros2 service call /set_robot_type physical_ai_interfaces/srv/SetRobotType \
    "{robot_type: 'ffw_sg2_rev1'}"

# Inference 시작 (command: 3 = START_INFERENCE)
ros2 service call /task/command physical_ai_interfaces/srv/SendCommand "{
  command: 3,
  task_info: {
    policy_path: '/workspace/Isaac-GR00T/results/gr00t/checkpoint-100000',
    fps: 10,
    task_instruction: ['Execute the task'],
    record_inference_mode: false
  }
}"

# Inference 중지 (command: 5 = FINISH)
ros2 service call /task/command physical_ai_interfaces/srv/SendCommand \
    "{command: 5}"
```

### 방법 3: Web UI

1. 브라우저에서 `http://<server-ip>:3000` 접속
2. Robot Type: `ffw_sg2_rev1` 선택
3. Inference 탭 → Policy Path 입력
4. Start Inference 클릭

### 기존 시스템 vs gr00t_bridge 비교

| 기능 | physical_ai_server | gr00t_bridge |
|------|-------------------|--------------|
| 토픽 퍼블리시 | ✅ 올바른 토픽 | ✅ 올바른 토픽 |
| Web UI 통합 | ✅ | ❌ |
| 데이터 수집 동시 | ✅ | ❌ |
| 독립 실행 | ❌ (서버 필요) | ✅ |
| ros2 bag 테스트 | 복잡 | ✅ 간단 |

## Related Files

- Training config: `/workspace/Isaac-GR00T/ffw_sg2_data_config.py`
- Robot config: `physical_ai_server/config/ffw_sg2_rev1_config.yaml`
- Original inference manager: `physical_ai_server/inference/inference_manager.py`




## Test
```bash
# terminal 1
ros2 bag play 123/ --clock -l --exclude-topics /leader/joint_trajectory_command_broadcaster_left/joint_trajectory

# terminal 2
python3 scripts/run_gr00t_inference_offline.py     --checkpoint /workspace/Isaac-GR00T/results/gr00t/checkpoint-100000     --publish     --use-sim-time

ros2 topic echo /leader/joint_trajectory_command_broadcaster_left/joint_trajectory

```