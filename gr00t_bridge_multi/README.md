# GR00T Multi-Camera Inference Bridge

Multi-camera GR00T inference for FFW SG2 dual-arm robot with ROS2 Jazzy.

## Overview

This package provides multi-camera GR00T inference support with **3 camera inputs**:

| Camera Key | Topic | Native Resolution |
|-----------|-------|-------------------|
| `video.ego_view` | `/zed/zed_node/left/image_rect_color/compressed` | 672×376 |
| `video.cam_wrist_left` | `/camera_left/camera_left/color/image_rect_raw/compressed` | 424×240 |
| `video.cam_wrist_right` | `/camera_right/camera_right/color/image_rect_raw/compressed` | 424×240 |

**Key Feature**: Handles different camera resolutions at runtime using `VideoResizeIndependent` transform that resizes each camera to 224×224 **before** concatenation.

---

## Quick Start

### 1. Enter Docker Container

```bash
cd /media/metafarmers/7440017c-39b5-476a-a24b-4fb0a9c24140/chae/dualArm_rosbag/physical_ai_tools
./docker/container.sh enter
```

### 2. Install Dependencies

```bash
# Inside Docker
cd /workspace/Isaac-GR00T
pip install -e .

# Install gr00t_bridge_multi
cd ~/gr00t_bridge_multi
pip install -e .
```

### 3. Test with ros2 bag

**Terminal 1** (ros2 bag playback):
```bash
ros2 bag play /path/to/rosbag --clock -l \
    --exclude-topics \
    /leader/joint_trajectory_command_broadcaster_left/joint_trajectory \
    /leader/joint_trajectory_command_broadcaster_right/joint_trajectory
```

**Terminal 2** (inference - dry-run):
```bash
cd ~/gr00t_bridge_multi
python3 scripts/run_gr00t_inference_offline.py \
    --checkpoint /workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000 \
    --use-sim-time \
    --dry-run
```

**Terminal 3** (monitor output - optional):
```bash
ros2 topic echo /gr00t/predicted_action
```

---

## Detailed Usage

### Dry-Run Mode (Default)

Prints predicted actions without publishing:

```bash
python3 scripts/run_gr00t_inference_offline.py \
    --checkpoint /workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000 \
    --use-sim-time \
    --dry-run
```

### Publish Mode

Actually publishes actions to robot control topics:

```bash
python3 scripts/run_gr00t_inference_offline.py \
    --checkpoint /workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000 \
    --use-sim-time \
    --publish
```

### Custom Camera Topics

```bash
python3 scripts/run_gr00t_inference_offline.py \
    --checkpoint /workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000 \
    --ego-view-topic /zed/zed_node/left/image_rect_color/compressed \
    --wrist-left-topic /camera_left/camera_left/color/image_rect_raw/compressed \
    --wrist-right-topic /camera_right/camera_right/color/image_rect_raw/compressed \
    --use-sim-time \
    --dry-run
```

### All Options

```bash
python3 scripts/run_gr00t_inference_offline.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `/workspace/.../checkpoint-100000` | Model checkpoint path |
| `--embodiment` | `new_embodiment` | Embodiment tag |
| `--device` | `cuda` | Device (cuda/cpu) |
| `--rate` | `10.0` | Inference rate (Hz) |
| `--dry-run` | False | Print-only mode |
| `--publish` | False | Enable publishing |
| `--use-sim-time` | False | Use sim time for ros2 bag |
| `--ego-view-topic` | `/zed/.../compressed` | Head camera topic |
| `--wrist-left-topic` | `/camera_left/.../compressed` | Left wrist camera |
| `--wrist-right-topic` | `/camera_right/.../compressed` | Right wrist camera |
| `--joint-topic` | `/joint_states` | Joint state topic |

---

## Topics

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/zed/zed_node/left/image_rect_color/compressed` | `sensor_msgs/CompressedImage` | Head camera (ego_view) |
| `/camera_left/camera_left/color/image_rect_raw/compressed` | `sensor_msgs/CompressedImage` | Left wrist camera |
| `/camera_right/camera_right/color/image_rect_raw/compressed` | `sensor_msgs/CompressedImage` | Right wrist camera |
| `/joint_states` | `sensor_msgs/JointState` | Robot joint states |

### Published Topics (when `--publish`)

| Topic | Type | Description |
|-------|------|-------------|
| `/leader/joint_trajectory_command_broadcaster_left/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Left arm commands |
| `/leader/joint_trajectory_command_broadcaster_right/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Right arm commands |
| `/gr00t/predicted_action` | `trajectory_msgs/JointTrajectory` | Debug: combined actions |

---

## State/Action Mapping

### State Keys (Input)

| Key | Dimension | Source |
|-----|-----------|--------|
| `state.left_arm` | 7 | `arm_l_joint1` ~ `arm_l_joint7` |
| `state.left_hand` | 1 | `gripper_l_joint1` |
| `state.right_arm` | 7 | `arm_r_joint1` ~ `arm_r_joint7` |
| `state.right_hand` | 1 | `gripper_r_joint1` |

### Action Keys (Output)

| Key | Dimension | Target |
|-----|-----------|--------|
| `action.left_arm` | 7 | Left arm joint positions |
| `action.left_hand` | 1 | Left gripper position |
| `action.right_arm` | 7 | Right arm joint positions |
| `action.right_hand` | 1 | Right gripper position |

---

## Architecture

### Key Difference from Single-Camera

The multi-camera config uses `VideoResizeIndependent` to handle different native resolutions:

```python
# Single-cam: Standard pipeline
VideoToTensor → VideoCrop → VideoResize → VideoToNumpy

# Multi-cam: Independent resize FIRST
VideoResizeIndependent → VideoToNumpy
```

This allows:
- ego_view: 672×376 → 224×224
- cam_wrist_left: 424×240 → 224×224  
- cam_wrist_right: 424×240 → 224×224

All cameras become 224×224 **before** concatenation.

### File Structure

```
gr00t_bridge_multi/
├── __init__.py
├── ffw_sg2_multicam_inference_config.py  # Multi-cam data config
├── gr00t_policy.py                        # Inference wrapper
├── gr00t_inference_node.py                # ROS2 node
├── setup.py
├── README.md
└── scripts/
    └── run_gr00t_inference_offline.py     # Runner script
```

---

## Troubleshooting

### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Try CPU inference:
```bash
python3 scripts/run_gr00t_inference_offline.py --device cpu --dry-run
```

### 2. Missing Camera Topics

```
[WARN] Missing cameras: ['cam_wrist_left', 'cam_wrist_right']
```

**Solution**: Check that ros2 bag contains all 3 camera topics:
```bash
ros2 bag info /path/to/bag
```

Or check topic names:
```bash
ros2 topic list | grep -E "camera|zed"
```

### 3. No Inference Happening

If status shows all data ready but no inference:

```bash
# Check if messages are being received
ros2 topic hz /zed/zed_node/left/image_rect_color/compressed
ros2 topic hz /camera_left/camera_left/color/image_rect_raw/compressed
ros2 topic hz /joint_states
```

### 4. Dimension Mismatch

```
ValueError: State left_arm has X dims, expected 7
```

**Solution**: Verify joint names in `/joint_states` match config:
```bash
ros2 topic echo /joint_states --once | grep name
```

Expected joints: `arm_l_joint1-7`, `gripper_l_joint1`, `arm_r_joint1-7`, `gripper_r_joint1`

### 5. Checkpoint Not Found

```
ERROR: Checkpoint not found at: /workspace/...
```

**Solution**: Verify path inside Docker:
```bash
ls -la /workspace/Isaac-GR00T/results/gr00t_multi_cam/
```

### 6. Import Error for Isaac-GR00T

```
ImportError: Failed to import Isaac-GR00T
```

**Solution**: Install Isaac-GR00T:
```bash
cd /workspace/Isaac-GR00T
pip install -e .
```

### 7. Different Resolution Warning (OK to ignore)

```
VideoResizeIndependent: Resizing ego_view from (376, 672) to (224, 224)
```

This is expected and handled automatically.

---

## Comparison: Single-cam vs Multi-cam

| Feature | gr00t_bridge | gr00t_bridge_multi |
|---------|--------------|-------------------|
| Cameras | 1 (ego_view) | 3 (ego + wrists) |
| Data Config | `FFWSG2InferenceConfig` | `FFWSG2MultiCamInferenceConfig` |
| Resolution Handling | Standard | VideoResizeIndependent |
| Checkpoint | `gr00t/checkpoint-*` | `gr00t_multi_cam/checkpoint-*` |

---

## Isaac Sim Integration

For Isaac Sim replay of inference results, use the same topic setup:

**Terminal 1** (ros2 bag):
```bash
ros2 bag play /path/to/rosbag --clock -l \
    --exclude-topics \
    /leader/joint_trajectory_command_broadcaster_left/joint_trajectory \
    /leader/joint_trajectory_command_broadcaster_right/joint_trajectory

```

**Terminal 2** (GR00T inference):
```bash
python3 scripts/run_gr00t_inference_offline.py --publish --use-sim-time
```

**Terminal 3** (Isaac Sim):
```bash
cd /path/to/mf_isaac
conda activate isaacsim
python ffw_sg2_inference_replay.py
```

---

## Test Commands Summary

```bash
### Docker enter
cd /media/metafarmers/7440017c-39b5-476a-a24b-4fb0a9c24140/chae/dualArm_rosbag/physical_ai_tools
./docker/container.sh enter

# terminal 1 - ros2 bag
ros2 bag play /workspace/rosbag2/dkim/ffw_sg2_rev1_dkim/123 --clock -l \
    --exclude-topics \
    /leader/joint_trajectory_command_broadcaster_left/joint_trajectory \
    /leader/joint_trajectory_command_broadcaster_right/joint_trajectory

# terminal 2 - setup
cd /workspace/Isaac-GR00T && pip install -e .
cd ~/gr00t_bridge_multi && pip install -e .

# terminal 2 - inference
python3 scripts/run_gr00t_inference_offline.py \
    --checkpoint /workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000 \
    --publish \
    --use-sim-time

# terminal 3 - verify
ros2 topic echo /leader/joint_trajectory_command_broadcaster_left/joint_trajectory
```

---

## Notes

- Model was trained with `FFWSG2MultiCamDataConfig` (pre-resized dataset)
- Inference uses `FFWSG2MultiCamInferenceConfig` (runtime resize)
- Action horizon: 16 timesteps, only first timestep is used for real-time control
- Inference waits for ALL 3 cameras + joint states before running
