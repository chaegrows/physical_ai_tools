#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# GR00T Multi-Camera Inference Wrapper for ROS2 integration
#
# This module provides a clean wrapper around the Isaac-GR00T Gr00tPolicy
# for use in ROS2-based robotics applications with MULTIPLE cameras.
#
# Supported Cameras:
# - video.ego_view (head camera)
# - video.cam_wrist_left (left wrist camera)
# - video.cam_wrist_right (right wrist camera)
#
# Key Features:
# - Handles different camera resolutions via VideoResizeIndependent
# - Configurable video_keys for single or multi-camera
# - Reuses the same transforms as training
# - Clear interface for ROS2 integration

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import cv2
import warnings

# Suppress warnings during import
warnings.filterwarnings('ignore', category=UserWarning)

# Add Isaac-GR00T to path if not already there
ISAAC_GROOT_PATH = Path("/workspace/Isaac-GR00T")
if ISAAC_GROOT_PATH.exists() and str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))


@dataclass
class ActionOutput:
    """Structured output from GR00T inference.
    
    The model outputs joint-space actions for the FFW SG2 robot:
    - left_arm: 7 joint positions (radians)
    - left_hand: 1 gripper position
    - right_arm: 7 joint positions (radians)
    - right_hand: 1 gripper position
    
    Total: 16 action dimensions per timestep, with 16 timesteps in action horizon
    """
    left_arm: np.ndarray    # Shape: (16, 7) - action horizon x joint dims
    left_hand: np.ndarray   # Shape: (16, 1)
    right_arm: np.ndarray   # Shape: (16, 7)
    right_hand: np.ndarray  # Shape: (16, 1)
    raw_dict: Dict[str, np.ndarray]  # Original action dict from model
    
    def get_first_timestep(self) -> Dict[str, np.ndarray]:
        """Get actions for the first timestep only (most common use case)."""
        return {
            "left_arm": self.left_arm[0] if len(self.left_arm.shape) > 1 else self.left_arm,
            "left_hand": self.left_hand[0] if len(self.left_hand.shape) > 1 else self.left_hand,
            "right_arm": self.right_arm[0] if len(self.right_arm.shape) > 1 else self.right_arm,
            "right_hand": self.right_hand[0] if len(self.right_hand.shape) > 1 else self.right_hand,
        }
    
    def get_concatenated_action(self, timestep: int = 0) -> np.ndarray:
        """Get concatenated action vector for a specific timestep.
        
        Returns: np.ndarray of shape (16,) in order:
            [left_arm(7), left_hand(1), right_arm(7), right_hand(1)]
        """
        first = self.get_first_timestep() if timestep == 0 else {
            "left_arm": self.left_arm[timestep],
            "left_hand": self.left_hand[timestep],
            "right_arm": self.right_arm[timestep],
            "right_hand": self.right_hand[timestep],
        }
        return np.concatenate([
            first["left_arm"].flatten(),
            first["left_hand"].flatten(),
            first["right_arm"].flatten(),
            first["right_hand"].flatten(),
        ])


class Gr00tMultiCamInferenceWrapper:
    """Wrapper for Isaac-GR00T model inference with MULTI-CAMERA support.
    
    This wrapper:
    1. Loads the GR00T model from a checkpoint
    2. Configures the data transforms for multi-camera inference
    3. Handles different camera resolutions via independent resizing
    4. Provides a clean interface for inference
    
    Example usage (3 cameras):
        wrapper = Gr00tMultiCamInferenceWrapper(
            checkpoint_path="/workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000",
            embodiment_tag="new_embodiment",
            device="cuda"
        )
        
        action = wrapper.predict(
            images={
                "ego_view": head_camera_image,      # (H1, W1, C) uint8
                "cam_wrist_left": left_wrist_image, # (H2, W2, C) uint8
                "cam_wrist_right": right_wrist_image, # (H3, W3, C) uint8
            },
            state={
                "left_arm": np.array([...]),   # 7 joints
                "left_hand": np.array([...]),  # 1 gripper
                "right_arm": np.array([...]),  # 7 joints
                "right_hand": np.array([...]), # 1 gripper
            },
            task_instruction="Pick up the object"
        )
    """
    
    # Expected dimensions for state components
    STATE_DIMS = {
        "left_arm": 7,
        "left_hand": 1,
        "right_arm": 7,
        "right_hand": 1,
    }
    
    # Default camera configuration
    DEFAULT_VIDEO_KEYS = [
        "video.ego_view",
        "video.cam_wrist_left",
        "video.cam_wrist_right",
    ]
    
    # Target resolution for all cameras (after independent resize)
    TARGET_HEIGHT = 224
    TARGET_WIDTH = 224
    
    def __init__(
        self,
        checkpoint_path: str,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        data_config: Optional[Any] = None,
        denoising_steps: Optional[int] = None,
        video_keys: Optional[List[str]] = None,
    ):
        """Initialize the GR00T multi-camera inference wrapper.
        
        Args:
            checkpoint_path: Path to the finetuned GR00T checkpoint
            embodiment_tag: Embodiment identifier (must match training)
            device: Device for inference ("cuda" or "cpu")
            data_config: Optional custom data config. If None, uses FFWSG2MultiCamInferenceConfig
            denoising_steps: Optional override for denoising steps (default: model's setting)
            video_keys: Optional override for video keys (for single-cam vs multi-cam switching)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.embodiment_tag = embodiment_tag
        self.device = device
        self.denoising_steps = denoising_steps
        
        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at: {self.checkpoint_path}\n"
                f"Please verify the path is correct."
            )
        
        # Initialize data config
        if data_config is None:
            from .ffw_sg2_multicam_inference_config import FFWSG2MultiCamInferenceConfig
            self.data_config = FFWSG2MultiCamInferenceConfig()
        else:
            self.data_config = data_config
        
        # Override video_keys if specified
        if video_keys is not None:
            self.data_config.video_keys = video_keys
        
        # Store video keys for reference
        self._video_keys = self.data_config.video_keys
        
        # Load the model
        self._load_model()
        
        print(f"[Gr00tMultiCamWrapper] Model loaded successfully!")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Embodiment: {self.embodiment_tag}")
        print(f"  Device: {self.device}")
        print(f"  Video keys: {self.data_config.video_keys}")
        print(f"  State keys: {self.data_config.state_keys}")
        print(f"  Action keys: {self.data_config.action_keys}")
        print(f"  Cameras: {len(self._video_keys)}")
    
    def _load_model(self):
        """Load the GR00T model and configure transforms."""
        try:
            from gr00t.model.policy import Gr00tPolicy
        except ImportError as e:
            raise ImportError(
                f"Failed to import Isaac-GR00T. Make sure it's installed:\n"
                f"  pip install -e /workspace/Isaac-GR00T\n"
                f"Original error: {e}"
            )
        
        # Get modality config and transform from data config
        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()
        
        # Load the policy
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.policy = Gr00tPolicy(
                model_path=str(self.checkpoint_path),
                embodiment_tag=self.embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=self.denoising_steps,
                device=self.device,
            )
    
    def preprocess_image(
        self,
        image: np.ndarray,
        camera_key: str = "ego_view"
    ) -> Dict[str, np.ndarray]:
        """Preprocess a single camera image for GR00T.
        
        The image will be resized to 224x224 by the transform pipeline.
        
        Args:
            image: Image array of shape (H, W, C) in BGR or RGB format, uint8
            camera_key: Camera identifier (e.g., "ego_view", "cam_wrist_left")
        
        Returns:
            Dict with key "video.{camera_key}" and value of shape (1, H, W, C)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        
        # Ensure correct shape
        if len(image.shape) != 3:
            raise ValueError(f"Expected (H, W, C) image, got shape {image.shape}")
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Add time dimension: (H, W, C) -> (1, H, W, C)
        image = image[np.newaxis, ...]
        
        return {f"video.{camera_key}": image}
    
    def preprocess_images(
        self,
        images: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Preprocess multiple camera images for GR00T.
        
        Args:
            images: Dict mapping camera names to images (H, W, C) uint8
                    Keys should be: "ego_view", "cam_wrist_left", "cam_wrist_right"
        
        Returns:
            Dict with keys like "video.ego_view" and values of shape (1, H, W, C)
        """
        observation = {}
        
        for camera_key, image in images.items():
            obs_image = self.preprocess_image(image, camera_key)
            observation.update(obs_image)
        
        return observation
    
    def preprocess_state(
        self,
        state: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Preprocess robot state for GR00T.
        
        Args:
            state: Dictionary with keys like "left_arm", "right_arm", etc.
                   Each value should be a 1D numpy array of joint positions.
        
        Returns:
            Dict with keys like "state.left_arm" and values of shape (1, D)
        """
        observation = {}
        
        for key, expected_dim in self.STATE_DIMS.items():
            if key not in state:
                raise ValueError(
                    f"Missing required state key: {key}\n"
                    f"Required keys: {list(self.STATE_DIMS.keys())}"
                )
            
            value = state[key]
            
            # Convert to numpy if needed
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=np.float64)
            else:
                value = value.astype(np.float64)
            
            # Flatten if needed
            value = value.flatten()
            
            # Validate dimension
            if len(value) != expected_dim:
                raise ValueError(
                    f"State {key} has {len(value)} dims, expected {expected_dim}"
                )
            
            # Add time dimension: (D,) -> (1, D)
            value = value[np.newaxis, ...]
            
            observation[f"state.{key}"] = value
        
        return observation
    
    def preprocess(
        self,
        images: Dict[str, np.ndarray],
        state: Dict[str, np.ndarray],
        task_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Preprocess all inputs for GR00T inference.
        
        Args:
            images: Dict mapping camera names to images (H, W, C) uint8
                    Expected keys: "ego_view", "cam_wrist_left", "cam_wrist_right"
            state: Dict mapping state component names to joint positions
            task_instruction: Optional language instruction for the task
        
        Returns:
            Observation dict ready for model inference
        """
        # Validate we have all required cameras
        required_cameras = [k.replace("video.", "") for k in self._video_keys]
        missing = set(required_cameras) - set(images.keys())
        if missing:
            raise ValueError(
                f"Missing required camera images: {missing}\n"
                f"Required cameras: {required_cameras}\n"
                f"Provided cameras: {list(images.keys())}"
            )
        
        observation = {}
        
        # Process images
        for camera_key, image in images.items():
            obs_image = self.preprocess_image(image, camera_key)
            observation.update(obs_image)
        
        # Process state
        obs_state = self.preprocess_state(state)
        observation.update(obs_state)
        
        # Add task instruction if provided
        if task_instruction is not None:
            observation["annotation.human.action.task_description"] = [task_instruction]
        
        return observation
    
    def predict(
        self,
        images: Dict[str, np.ndarray],
        state: Dict[str, np.ndarray],
        task_instruction: Optional[str] = None,
    ) -> ActionOutput:
        """Run inference and return predicted actions.
        
        Args:
            images: Dict mapping camera names to images (H, W, C) uint8
                    Example: {
                        "ego_view": head_camera_image,
                        "cam_wrist_left": left_wrist_image,
                        "cam_wrist_right": right_wrist_image,
                    }
            state: Dict mapping state component names to joint positions
                   Example: {"left_arm": np.array([...]), ...}
            task_instruction: Optional language instruction
        
        Returns:
            ActionOutput with predicted joint-space actions
        """
        # Preprocess inputs
        observation = self.preprocess(images, state, task_instruction)
        
        # Run inference
        action_dict = self.policy.get_action(observation)
        
        # Parse action output
        return self._parse_action_output(action_dict)
    
    def _parse_action_output(self, action_dict: Dict[str, np.ndarray]) -> ActionOutput:
        """Parse the raw action dict from the model into structured output."""
        
        # Extract action components
        left_arm = action_dict.get("action.left_arm", np.zeros((16, 7)))
        left_hand = action_dict.get("action.left_hand", np.zeros((16, 1)))
        right_arm = action_dict.get("action.right_arm", np.zeros((16, 7)))
        right_hand = action_dict.get("action.right_hand", np.zeros((16, 1)))
        
        return ActionOutput(
            left_arm=left_arm,
            left_hand=left_hand,
            right_arm=right_arm,
            right_hand=right_hand,
            raw_dict=action_dict,
        )
    
    def get_modality_config(self) -> Dict[str, Any]:
        """Get the modality configuration used by the model."""
        return self.policy.get_modality_config()
    
    @property
    def action_horizon(self) -> int:
        """Get the action horizon (number of future timesteps predicted)."""
        return len(self.data_config.action_indices)
    
    @property
    def video_keys(self) -> List[str]:
        """Get the expected video/camera keys."""
        return self._video_keys
    
    @property
    def state_keys(self) -> List[str]:
        """Get the expected state keys."""
        return self.data_config.state_keys
    
    @property
    def action_keys(self) -> List[str]:
        """Get the expected action keys."""
        return self.data_config.action_keys
    
    @property
    def num_cameras(self) -> int:
        """Get the number of cameras expected."""
        return len(self._video_keys)


def test_inference():
    """Simple test to verify the wrapper works with multi-camera."""
    print("=" * 60)
    print("GR00T Multi-Camera Inference Wrapper Test")
    print("=" * 60)
    
    # Check if checkpoint exists
    checkpoint_path = "/workspace/Isaac-GR00T/results/gr00t_multi_cam/checkpoint-100000"
    if not Path(checkpoint_path).exists():
        print(f"[SKIP] Checkpoint not found at: {checkpoint_path}")
        print("       This test requires a trained multi-cam model checkpoint.")
        return False
    
    # Try to load the model
    try:
        wrapper = Gr00tMultiCamInferenceWrapper(
            checkpoint_path=checkpoint_path,
            embodiment_tag="new_embodiment",
            device="cuda",
        )
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create dummy inputs with different resolutions
    dummy_images = {
        "ego_view": np.random.randint(0, 255, (376, 672, 3), dtype=np.uint8),
        "cam_wrist_left": np.random.randint(0, 255, (240, 424, 3), dtype=np.uint8),
        "cam_wrist_right": np.random.randint(0, 255, (240, 424, 3), dtype=np.uint8),
    }
    dummy_state = {
        "left_arm": np.zeros(7),
        "left_hand": np.zeros(1),
        "right_arm": np.zeros(7),
        "right_hand": np.zeros(1),
    }
    
    # Run inference
    try:
        action = wrapper.predict(
            images=dummy_images,
            state=dummy_state,
            task_instruction="Test task"
        )
        
        # Verify output
        first_action = action.get_first_timestep()
        concat_action = action.get_concatenated_action()
        
        print(f"[PASS] Multi-camera inference successful!")
        print(f"       Action shapes:")
        print(f"         left_arm:  {action.left_arm.shape}")
        print(f"         left_hand: {action.left_hand.shape}")
        print(f"         right_arm: {action.right_arm.shape}")
        print(f"         right_hand: {action.right_hand.shape}")
        print(f"       First timestep concatenated: {concat_action.shape}")
        print(f"       First action values: {concat_action[:4]}...")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_inference()
