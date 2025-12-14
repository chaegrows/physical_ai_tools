# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Inference-specific data config for FFW SG2 Rev1 robot.
# This matches the FFWSG2DataConfig used during training, but with
# inference-appropriate settings (no color jitter, etc.).
#
# IMPORTANT: This config must match the training config exactly in terms of:
# - video_keys (camera names)
# - state_keys (state variable names)
# - action_keys (action variable names)
# - observation_indices and action_indices
#
# The only difference is that we skip training-specific augmentations like
# color jitter during inference.

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Add Isaac-GR00T to path if not already there
ISAAC_GROOT_PATH = Path("/workspace/Isaac-GR00T")
if ISAAC_GROOT_PATH.exists() and str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform
from gr00t.experiment.data_config import BaseDataConfig


@dataclass
class FFWSG2InferenceConfig(BaseDataConfig):
    """Inference config for FFW SG2 Rev1 dual-arm robot.
    
    This config is designed for real-time inference with a single head camera.
    It matches the FFWSG2DataConfig used during training, but without
    training-specific augmentations (color jitter).
    
    DESIGN NOTES:
    - Primary goal: Support stable single-camera (ego_view) inference
    - Future extension: Multi-camera support can be added by extending video_keys
      to ["video.ego_view", "video.cam_wrist_left", "video.cam_wrist_right"]
      with corresponding changes to use FFWSG2MultiCamInferenceConfig
    
    Camera Configuration (single camera for now):
    - video.ego_view: Head camera (672x376 native, resized to 224x224)
    
    State Configuration (16 DOF arms + grippers):
    - state.left_arm: 7 joints
    - state.left_hand: 1 gripper
    - state.right_arm: 7 joints
    - state.right_hand: 1 gripper
    
    Action Configuration (same as state):
    - action.left_arm: 7 joints
    - action.left_hand: 1 gripper
    - action.right_arm: 7 joints
    - action.right_hand: 1 gripper
    """
    
    # Camera keys - single camera (ego_view) for now
    # NOTE: To switch to multi-camera, extend this list and use FFWSG2MultiCamInferenceConfig
    video_keys: List[str] = None
    
    # State and action keys for the dual-arm configuration
    state_keys: List[str] = None
    action_keys: List[str] = None
    
    # Language key for task instruction
    language_keys: List[str] = None
    
    # Observation and action horizons (same as training)
    observation_indices: List[int] = None
    action_indices: List[int] = None
    
    def __post_init__(self):
        # Default values (can't use mutable defaults in dataclass)
        if self.video_keys is None:
            self.video_keys = ["video.ego_view"]
        if self.state_keys is None:
            self.state_keys = [
                "state.left_arm",
                "state.left_hand",
                "state.right_arm",
                "state.right_hand",
            ]
        if self.action_keys is None:
            self.action_keys = [
                "action.left_arm",
                "action.left_hand",
                "action.right_arm",
                "action.right_hand",
            ]
        if self.language_keys is None:
            self.language_keys = ["annotation.human.action.task_description"]
        if self.observation_indices is None:
            self.observation_indices = [0]
        if self.action_indices is None:
            self.action_indices = list(range(16))

    def modality_config(self) -> dict:
        """Return modality configuration for the model."""
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        """Return the modality transform for inference.
        
        This is similar to the training transform but WITHOUT color jitter
        since we don't want augmentation during inference.
        """
        transforms = [
            # Video transforms (no color jitter for inference)
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # NOTE: No VideoColorJitter for inference
            VideoToNumpy(apply_to=self.video_keys),
            # State transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # Action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # Concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # Model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


# For future multi-camera support, use this config:
# 
# @dataclass
# class FFWSG2MultiCamInferenceConfig(FFWSG2InferenceConfig):
#     """Multi-camera inference config - handles different camera resolutions."""
#     
#     def __post_init__(self):
#         super().__post_init__()
#         self.video_keys = ["video.ego_view", "video.cam_wrist_left", "video.cam_wrist_right"]
#     
#     def transform(self) -> ModalityTransform:
#         # Use VideoResizeIndependent to handle different camera resolutions
#         # See ffw_sg2_data_config.py for implementation details
#         pass
