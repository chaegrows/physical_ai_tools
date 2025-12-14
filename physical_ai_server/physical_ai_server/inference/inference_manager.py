#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

import os
import sys
from pathlib import Path
import warnings
import logging

# Suppress HuggingFace transformers warnings about untrained models
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)

from lerobot.policies.pretrained import PreTrainedPolicy
import numpy as np
from physical_ai_server.utils.file_utils import read_json_file
import torch


class InferenceManager:

    def __init__(
            self,
            device: str = 'cuda'):

        self.device = device
        self.policy_type = None
        self.policy_path = None
        self.policy = None
        self.is_groot = False
        self.groot_data_config = None
        self.groot_embodiment_tag = None
        self._load_policy_error_logged = False  # Track if we've already logged the error
        self._last_load_attempt_time = None  # Track last load attempt time

    def validate_policy(self, policy_path: str) -> bool:
        result_message = ''
        if not os.path.exists(policy_path) or not os.path.isdir(policy_path):
            result_message = f'Policy path {policy_path} does not exist or is not a directory.'
            return False, result_message

        # Check if path points to pretrained_model folder, if not try to find it
        original_path = policy_path
        pretrained_model_path = policy_path
        
        # If path doesn't end with pretrained_model, check if pretrained_model subfolder exists
        if not policy_path.endswith('pretrained_model'):
            potential_pretrained_path = os.path.join(policy_path, 'pretrained_model')
            if os.path.exists(potential_pretrained_path) and os.path.isdir(potential_pretrained_path):
                pretrained_model_path = potential_pretrained_path
                print(f'Found pretrained_model folder at: {pretrained_model_path}')

        config_path = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.exists(config_path):
            # Also check in the original path
            config_path_original = os.path.join(original_path, 'config.json')
            if os.path.exists(config_path_original):
                config_path = config_path_original
                pretrained_model_path = original_path
            else:
                result_message = f'config.json file does not exist in {pretrained_model_path} or {original_path}.'
                return False, result_message

        config = read_json_file(config_path)
        if (config is None or
                ('type' not in config and 'model_type' not in config)):
            result_message = f'config.json malformed or missing fields in {pretrained_model_path}.'
            return False, result_message

        available_policies = self.__class__.get_available_policies()
        policy_type = config.get('type') or config.get('model_type')
        
        # Map GR00T model types to standard policy type names
        groot_type_mapping = {
            'gr00t_n1_5': 'gr00t',
            'gr00t_n1': 'groot-n1',
            'groot_n1_5': 'gr00t',
            'groot_n1': 'groot-n1',
        }
        
        if policy_type in groot_type_mapping:
            policy_type = groot_type_mapping[policy_type]
        
        # Check if it's a GR00T policy (groot or groot-n1)
        if policy_type in ['groot', 'groot-n1', 'gr00t']:
            self.is_groot = True
            # Get data_config and embodiment_tag from config if available
            # For GR00T N1.5, try to get from config, otherwise use defaults
            self.groot_data_config = config.get('data_config') or config.get('data_config_path')
            self.groot_embodiment_tag = config.get('embodiment_tag') or config.get('embodiment')
            
            # If not specified in config, try to detect from model path or use FFW SG2 defaults
            if not self.groot_data_config:
                # Check if ffw_sg2 data config is available
                if 'ffw_sg2' in policy_path.lower() or 'ffw-sg2' in policy_path.lower():
                    self.groot_data_config = 'ffw_sg2_data_config:FFWSG2DataConfig'
                else:
                    # Default to FFW SG2 config for ROBOTIS robots
                    self.groot_data_config = 'ffw_sg2_data_config:FFWSG2DataConfig'
                print(f'[GR00T] Using FFW SG2 data config: {self.groot_data_config}')
            
            if not self.groot_embodiment_tag:
                # Default to new_embodiment for finetuned models
                self.groot_embodiment_tag = 'new_embodiment'
                print(f'[GR00T] Using default embodiment tag: {self.groot_embodiment_tag}')
            
            # Log the detected GR00T configuration
            print(f'Detected GR00T model type: {policy_type}')
            print(f'  - Data config: {self.groot_data_config}')
            print(f'  - Embodiment tag: {self.groot_embodiment_tag}')
        elif policy_type not in available_policies:
            result_message = f'Policy type {policy_type} is not supported.'
            return False, result_message

        # Use the pretrained_model path for loading
        self.policy_path = pretrained_model_path
        self.policy_type = policy_type
        return True, f'Policy {policy_type} is valid.'

    def load_policy(self):
        import time
        
        # Prevent too frequent retry attempts (at least 5 seconds between attempts)
        current_time = time.time()
        if self._last_load_attempt_time is not None:
            time_since_last_attempt = current_time - self._last_load_attempt_time
            if time_since_last_attempt < 5.0:
                # Too soon to retry, skip silently
                return False
        
        self._last_load_attempt_time = current_time
        
        try:
            if self.is_groot:
                # Load GR00T policy
                result = self._load_groot_policy()
                if result:
                    self._load_policy_error_logged = False  # Reset on success
                return result
            else:
                # Load LeRobot policy
                policy_cls = self._get_policy_class(self.policy_type)
                self.policy = policy_cls.from_pretrained(self.policy_path)
                self._load_policy_error_logged = False  # Reset on success
                return True
        except Exception as e:
            # Only log error once to avoid spam, or if it's a different error
            if not self._load_policy_error_logged:
                print(f'Failed to load policy from {self.policy_path}: {e}')
                import traceback
                traceback.print_exc()
                self._load_policy_error_logged = True
            return False

    def clear_policy(self):
        if hasattr(self, 'policy'):
            del self.policy
            self.policy = None
        else:
            print('No policy to clear.')
        # Reset error tracking when clearing policy
        self._load_policy_error_logged = False
        self._last_load_attempt_time = None

    def get_policy_config(self):
        if self.is_groot:
            # GR00T policy doesn't have a standard config attribute
            return {
                'policy_type': self.policy_type,
                'data_config': self.groot_data_config,
                'embodiment_tag': self.groot_embodiment_tag,
            }
        return self.policy.config

    def predict(
            self,
            images: dict[str, np.ndarray],
            state: list[float],
            task_instruction: str = None) -> list:

        if self.is_groot:
            # Use GR00T-specific prediction
            return self._predict_groot(images, state, task_instruction)
        else:
            # Use LeRobot prediction
            observation = self._preprocess(images, state, task_instruction)
            with torch.inference_mode():
                action = self.policy.select_action(observation)
                action = action.squeeze(0).to('cpu').numpy()
            return action

    def _preprocess(
            self,
            images: dict[str, np.ndarray],
            state: list,
            task_instruction: str = None) -> dict:

        observation = self._convert_images2tensors(images)
        observation['observation.state'] = self._convert_np2tensors(state)
        for key in observation.keys():
            observation[key] = observation[key].to(self.device)

        if task_instruction is not None:
            observation['task'] = [task_instruction]

        return observation

    def _convert_images2tensors(
            self,
            images: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:

        processed_images = {}
        for key, value in images.items():
            image = torch.from_numpy(value)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.to(self.device, non_blocking=True)
            image = image.unsqueeze(0)
            processed_images['observation.images.' + key] = image

        return processed_images

    def _convert_np2tensors(
            self,
            data):
        if isinstance(data, list):
            data = np.array(data)
        tensor_data = torch.from_numpy(data)
        tensor_data = tensor_data.to(torch.float32)
        tensor_data = tensor_data.to(self.device, non_blocking=True)
        tensor_data = tensor_data.unsqueeze(0)

        return tensor_data

    def _load_groot_policy(self):
        """Load GR00T policy with modality config and transform."""
        try:
            # Add gr00t_bridge to path for custom data configs
            gr00t_bridge_path = Path(__file__).parent.parent.parent.parent / 'gr00t_bridge'
            if gr00t_bridge_path.exists():
                parent_path = str(gr00t_bridge_path.parent)
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)
                    print(f'[GR00T] Added gr00t_bridge parent to path: {parent_path}')
            
            # First, try importing gr00t directly (if installed via pip)
            try:
                from gr00t.model.policy import Gr00tPolicy
                from gr00t.experiment.data_config import load_data_config
                print('Using pip-installed gr00t package')
            except ImportError:
                # If pip installation not found, try to find Isaac-GR00T directory
                print('gr00t not found in pip packages, searching for Isaac-GR00T directory...')
                possible_paths = [
                    # Docker workspace path (mounted from ./workspace:/workspace)
                    Path('/workspace/Isaac-GR00T/'),
                    # Alternative workspace paths
                    Path('/workspace/gr00t'),
                    Path('/workspace/isaac-gr00t'),
                    # Relative path from current file (for local development)
                    Path(__file__).parent.parent.parent.parent.parent / 'Isaac-GR00T',
                    # Alternative relative path
                    Path(__file__).parent.parent.parent.parent.parent.parent / 'Isaac-GR00T',
                    # Absolute path from workspace root
                    Path('/root/ros2_ws/src/physical_ai_tools/Isaac-GR00T'),
                    # Path via docker workspace mount (../:/root/ros2_ws/src/physical_ai_tools/)
                    Path('/root/ros2_ws/src/physical_ai_tools/docker/workspace/Isaac-GR00T'),
                ]
                
                groot_path = None
                checked_paths = []
                
                # First, try direct paths
                for path in possible_paths:
                    checked_paths.append(path)
                    if path.exists() and path.is_dir():
                        # Verify it's actually GR00T by checking for gr00t subdirectory or pyproject.toml
                        has_gr00t_dir = (path / 'gr00t').exists() and (path / 'gr00t').is_dir()
                        has_pyproject = (path / 'pyproject.toml').exists()
                        if has_gr00t_dir or has_pyproject:
                            groot_path = path
                            print(f'Found Isaac-GR00T at: {groot_path}')
                            break
                
                # If not found, check workspace directory for GR00T
                if groot_path is None:
                    workspace_path = Path('/workspace')
                    if workspace_path.exists() and workspace_path.is_dir():
                        checked_paths.append(workspace_path)
                        # Look for GR00T directories in workspace
                        try:
                            for item in workspace_path.iterdir():
                                if item.is_dir() and 'gr00t' in item.name.lower():
                                    # Verify it's GR00T
                                    has_gr00t_dir = (item / 'gr00t').exists() and (item / 'gr00t').is_dir()
                                    has_pyproject = (item / 'pyproject.toml').exists()
                                    if has_gr00t_dir or has_pyproject:
                                        groot_path = item
                                        print(f'Found Isaac-GR00T at: {groot_path}')
                                        break
                        except Exception as e:
                            print(f'Warning: Could not iterate workspace directory: {e}')
                
                if groot_path is None:
                    # Provide more detailed error message
                    error_msg = 'Could not find Isaac-GR00T directory. Tried paths:\n'
                    for p in checked_paths:
                        exists = p.exists() if p else False
                        is_dir = p.is_dir() if exists else False
                        error_msg += f'  - {p} (exists: {exists}, is_dir: {is_dir})\n'
                    
                    # Check workspace directory contents if it exists
                    workspace_path = Path('/workspace')
                    if workspace_path.exists() and workspace_path.is_dir():
                        try:
                            workspace_contents = list(workspace_path.iterdir())
                            error_msg += f'\nContents of /workspace:\n'
                            for item in workspace_contents[:20]:  # Limit to first 20 items
                                item_type = 'dir' if item.is_dir() else 'file'
                                error_msg += f'  - {item.name} ({item_type})\n'
                            if len(workspace_contents) > 20:
                                error_msg += f'  ... and {len(workspace_contents) - 20} more items\n'
                        except Exception as e:
                            error_msg += f'\nCould not list /workspace contents: {e}\n'
                    else:
                        error_msg += f'\n/workspace directory does not exist or is not a directory.\n'
                    
                    raise FileNotFoundError(error_msg)
                
                # Add to sys.path if not already there
                groot_path_str = str(groot_path.resolve())
                if groot_path_str not in sys.path:
                    sys.path.insert(0, groot_path_str)
                    print(f'Added Isaac-GR00T to sys.path: {groot_path_str}')
                
                from gr00t.model.policy import Gr00tPolicy
                from gr00t.experiment.data_config import load_data_config
            
            # Ensure policy_path points to pretrained_model folder
            # GR00T expects the model path to be the pretrained_model directory
            model_path = self.policy_path
            if not model_path.endswith('pretrained_model'):
                potential_pretrained_path = os.path.join(model_path, 'pretrained_model')
                if os.path.exists(potential_pretrained_path) and os.path.isdir(potential_pretrained_path):
                    model_path = potential_pretrained_path
                    print(f'Using pretrained_model path: {model_path}')
            
            # Convert to absolute path to avoid path resolution issues
            model_path = os.path.abspath(model_path)
            print(f'Loading GR00T model from absolute path: {model_path}')
            
            # Verify GR00T model structure
            config_path = os.path.join(model_path, 'config.json')
            experiment_cfg_path = os.path.join(model_path, 'experiment_cfg')
            metadata_path = os.path.join(experiment_cfg_path, 'metadata.json')
            model_index_path = os.path.join(model_path, 'model.safetensors.index.json')
            model_single_path = os.path.join(model_path, 'model.safetensors')
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f'config.json not found at {model_path}')
            
            if not os.path.exists(experiment_cfg_path):
                raise FileNotFoundError(
                    f'experiment_cfg folder not found at {experiment_cfg_path}\n'
                    f'GR00T model requires experiment_cfg directory with metadata.json file.'
                )
            
            if not os.path.exists(metadata_path):
                # Check what files are in experiment_cfg
                experiment_cfg_contents = []
                if os.path.exists(experiment_cfg_path):
                    try:
                        experiment_cfg_contents = os.listdir(experiment_cfg_path)
                    except Exception:
                        pass
                
                error_msg = (
                    f'metadata.json not found at {metadata_path}\n'
                    f'Required for embodiment tag: {self.groot_embodiment_tag}\n'
                )
                if experiment_cfg_contents:
                    error_msg += f'Contents of experiment_cfg directory: {experiment_cfg_contents}\n'
                else:
                    error_msg += f'experiment_cfg directory is empty or cannot be read.\n'
                error_msg += (
                    f'\nTo fix this issue:\n'
                    f'1. Ensure the model was trained with GR00T N1.5 and includes experiment_cfg/metadata.json\n'
                    f'2. Check that the embodiment tag "{self.groot_embodiment_tag}" exists in metadata.json\n'
                    f'3. Verify the model path is correct: {model_path}\n'
                    f'4. If using a different embodiment tag, update config.json with the correct embodiment_tag'
                )
                raise FileNotFoundError(error_msg)
            
            # Check if the required embodiment tag exists in metadata.json
            try:
                metadata_content = read_json_file(metadata_path)
                if metadata_content:
                    available_tags = list(metadata_content.keys())
                    if self.groot_embodiment_tag not in available_tags:
                        error_msg = (
                            f'Embodiment tag "{self.groot_embodiment_tag}" not found in metadata.json\n'
                            f'Available embodiment tags: {available_tags}\n'
                            f'Metadata file: {metadata_path}\n'
                            f'\nTo fix this issue:\n'
                            f'1. Update config.json with the correct embodiment_tag (one of: {available_tags})\n'
                            f'2. Or ensure the model was trained with embodiment tag "{self.groot_embodiment_tag}"'
                        )
                        raise ValueError(error_msg)
                    print(f'Found embodiment tag "{self.groot_embodiment_tag}" in metadata.json')
                    print(f'Available embodiment tags: {available_tags}')
            except ValueError:
                raise  # Re-raise ValueError
            except Exception as e:
                print(f'Warning: Could not verify embodiment tag in metadata.json: {e}')
                print('Attempting to load anyway...')
            
            # Check for model files (sharded or single)
            has_sharded_model = os.path.exists(model_index_path)
            has_single_model = os.path.exists(model_single_path)
            
            print(f'Model file check:')
            print(f'  - Sharded model (model.safetensors.index.json): {has_sharded_model}')
            print(f'  - Single model (model.safetensors): {has_single_model}')
            
            if has_sharded_model:
                # List all sharded model files
                import glob
                sharded_files = glob.glob(os.path.join(model_path, 'model-*.safetensors'))
                print(f'  - Found {len(sharded_files)} sharded safetensors files')
                for f in sharded_files:
                    print(f'    * {os.path.basename(f)}')
            
            if not (has_sharded_model or has_single_model):
                # List available files for debugging
                available_files = os.listdir(model_path)
                print(f'Warning: No standard model files found.')
                print(f'Available files in {model_path}:')
                for f in available_files:
                    print(f'  - {f}')
                print('Attempting to load anyway - HuggingFace may find the files...')
            
            # Load data config
            # Try to load external FFW SG2 config first, then fall back to built-in configs
            try:
                if 'ffw_sg2' in self.groot_data_config.lower():
                    # Try to load from gr00t_bridge first (our custom config)
                    try:
                        from gr00t_bridge.ffw_sg2_inference_config import FFWSG2InferenceConfig
                        data_config = FFWSG2InferenceConfig()
                        print(f'[GR00T] Loaded FFWSG2InferenceConfig from gr00t_bridge')
                    except ImportError:
                        # Fall back to Isaac-GR00T's ffw_sg2_data_config
                        try:
                            # Add Isaac-GR00T path to find ffw_sg2_data_config
                            groot_config_path = Path('/workspace/Isaac-GR00T')
                            if groot_config_path.exists() and str(groot_config_path) not in sys.path:
                                sys.path.insert(0, str(groot_config_path))
                            data_config = load_data_config(self.groot_data_config)
                            print(f'[GR00T] Loaded data config from Isaac-GR00T: {self.groot_data_config}')
                        except Exception as e:
                            print(f'[GR00T] Warning: Could not load {self.groot_data_config}: {e}')
                            # Final fallback: use fourier_gr1_arms_only (similar to FFW SG2)
                            data_config = load_data_config('fourier_gr1_arms_only')
                            print(f'[GR00T] Falling back to fourier_gr1_arms_only')
                else:
                    data_config = load_data_config(self.groot_data_config)
            except Exception as e:
                print(f'[GR00T] Error loading data config: {e}')
                # Fallback to fourier_gr1_arms_only
                data_config = load_data_config('fourier_gr1_arms_only')
                print(f'[GR00T] Using fallback data config: fourier_gr1_arms_only')
            
            modality_config = data_config.modality_config()
            modality_transform = data_config.transform()
            
            print(f'Loading GR00T policy...')
            print(f'  - Model path: {model_path}')
            print(f'  - Embodiment tag: {self.groot_embodiment_tag}')
            print(f'  - Data config: {self.groot_data_config}')
            print(f'  - Device: {self.device}')
            
            # Load GR00T policy
            # GR00T expects the path to the pretrained_model folder directly
            # HuggingFace's from_pretrained will automatically handle sharded models via model.safetensors.index.json
            # Suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                import logging
                old_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.ERROR)
                try:
                    self.policy = Gr00tPolicy(
                        model_path=model_path,
                        embodiment_tag=self.groot_embodiment_tag,
                        modality_config=modality_config,
                        modality_transform=modality_transform,
                        device=self.device,
                    )
                finally:
                    logging.getLogger().setLevel(old_level)
            
            # Set model to eval mode (important for inference)
            # GR00T policy might have nested models, so we need to set eval mode recursively
            def set_eval_mode(module):
                """Recursively set all modules to eval mode"""
                module.eval()
                for child in module.children():
                    set_eval_mode(child)
            
            try:
                set_eval_mode(self.policy)
                print('[GR00T] Set all modules to eval mode')
            except Exception as e:
                print(f'[GR00T] Warning: Could not set eval mode: {e}')
                # Try alternative methods
                if hasattr(self.policy, 'eval'):
                    self.policy.eval()
                if hasattr(self.policy, 'model'):
                    if hasattr(self.policy.model, 'eval'):
                        self.policy.model.eval()
                    if hasattr(self.policy.model, 'backbone'):
                        if hasattr(self.policy.model.backbone, 'eval'):
                            self.policy.model.backbone.eval()
                    if hasattr(self.policy.model, 'action_head'):
                        if hasattr(self.policy.model.action_head, 'eval'):
                            self.policy.model.action_head.eval()
            
            # Verify eval mode
            training_status = []
            if hasattr(self.policy, 'training'):
                training_status.append(f'policy.training={self.policy.training}')
            if hasattr(self.policy, 'model'):
                if hasattr(self.policy.model, 'training'):
                    training_status.append(f'model.training={self.policy.model.training}')
                if hasattr(self.policy.model, 'backbone'):
                    if hasattr(self.policy.model.backbone, 'training'):
                        training_status.append(f'backbone.training={self.policy.model.backbone.training}')
                if hasattr(self.policy.model, 'action_head'):
                    if hasattr(self.policy.model.action_head, 'training'):
                        training_status.append(f'action_head.training={self.policy.model.action_head.training}')
            
            print('GR00T policy loaded successfully!')
            if training_status:
                print(f'[GR00T] Model training status: {", ".join(training_status)}')
            else:
                print('[GR00T] Could not determine model training status')
            
            return True
        except Exception as e:
            print(f'Failed to load GR00T policy: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def _predict_groot(
            self,
            images: dict[str, np.ndarray],
            state: list[float],
            task_instruction: str = None) -> list:
        """Predict using GR00T policy."""
        try:
            # Convert to GR00T observation format
            observation = self._preprocess_groot(images, state, task_instruction)
            
            # Debug: print observation keys and validate (reduced verbosity)
            # print(f'Observation keys: {list(observation.keys())}')
            observation_valid = True
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    has_nan = np.isnan(value).any() if value.size > 0 else False
                    has_inf = np.isinf(value).any() if value.size > 0 else False
                    if has_nan or has_inf:
                        print(f'  ERROR: {key}: shape={value.shape}, dtype={value.dtype}, has_nan={has_nan}, has_inf={has_inf}')
                        observation_valid = False
                    # Only print if there's an issue
                    # if value.size > 0:
                    #     print(f'    value range: [{value.min():.4f}, {value.max():.4f}]')
                # else:
                #     print(f'  {key}: type={type(value)}, value={value}')
            
            if not observation_valid:
                raise ValueError('Invalid observation detected (NaN/Inf)')
            
            # Validate observation before passing to policy
            for key, value in observation.items():
                if isinstance(value, np.ndarray) and value.size > 0:
                    if np.isnan(value).any():
                        raise ValueError(f'NaN detected in observation[{key}] before policy inference')
                    if np.isinf(value).any():
                        raise ValueError(f'Inf detected in observation[{key}] before policy inference')
            
            # Ensure model is in eval mode before inference
            # Recursively set all modules to eval mode
            def ensure_eval_mode(module):
                if hasattr(module, 'eval'):
                    module.eval()
                for child in module.children():
                    ensure_eval_mode(child)
            
            try:
                ensure_eval_mode(self.policy)
            except Exception:
                # Fallback to simple eval
                if hasattr(self.policy, 'eval'):
                    self.policy.eval()
                if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'eval'):
                    self.policy.model.eval()
            
            # GR00T's get_action applies transform internally, so we should NOT apply it here
            # Just pass the observation as-is (with video.* and state.* keys)
            
            # Debug: print observation structure before calling get_action
            import sys
            sys.stdout.flush()  # Ensure output is flushed
            print(f'[GR00T] Observation keys before get_action: {list(observation.keys())}', flush=True)
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    print(f'  {key}: shape={value.shape}, dtype={value.dtype}, min={value.min() if value.size > 0 else "N/A"}, max={value.max() if value.size > 0 else "N/A"}', flush=True)
                else:
                    print(f'  {key}: type={type(value)}, value={value}', flush=True)
            
            # Get action from GR00T policy
            # Use torch.no_grad() to disable gradient computation during inference
            with torch.no_grad():
                try:
                    print('[GR00T] Calling policy.get_action...', flush=True)
                    action_dict = self.policy.get_action(observation)
                    print(f'[GR00T] get_action succeeded, action_dict keys: {list(action_dict.keys()) if action_dict else "empty"}', flush=True)
                except Exception as e:
                    print(f'[GR00T] ERROR in policy.get_action: {e}', flush=True)
                    print(f'[GR00T] Observation keys: {list(observation.keys())}', flush=True)
                    print(f'[GR00T] Observation details:', flush=True)
                    for key, value in observation.items():
                        if isinstance(value, np.ndarray):
                            print(f'    {key}: shape={value.shape}, dtype={value.dtype}', flush=True)
                        else:
                            print(f'    {key}: type={type(value)}', flush=True)
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Debug: check for NaN in action dict and print action details
            import sys
            if action_dict:
                print(f'[GR00T] Action dict has {len(action_dict)} keys', flush=True)
                for key, value in action_dict.items():
                    if isinstance(value, np.ndarray):
                        has_nan = value.size > 0 and np.isnan(value).any()
                        print(f'  {key}: shape={value.shape}, dtype={value.dtype}, has_nan={has_nan}', flush=True)
                        if has_nan:
                            print(f'    ERROR: NaN detected! First few values: {value.flat[:10] if value.size >= 10 else value.flat[:]}', flush=True)
                        elif value.size > 0:
                            print(f'    Value range: [{value.min():.4f}, {value.max():.4f}]', flush=True)
                    elif isinstance(value, torch.Tensor):
                        value_np = value.cpu().numpy()
                        has_nan = value_np.size > 0 and np.isnan(value_np).any()
                        print(f'  {key}: shape={value_np.shape}, dtype={value_np.dtype}, has_nan={has_nan}', flush=True)
                        if has_nan:
                            print(f'    ERROR: NaN detected! First few values: {value_np.flat[:10] if value_np.size >= 10 else value_np.flat[:]}', flush=True)
                        elif value_np.size > 0:
                            print(f'    Value range: [{value_np.min():.4f}, {value_np.max():.4f}]', flush=True)
                    else:
                        print(f'  {key}: type={type(value)}', flush=True)
            else:
                print('[GR00T] Warning: action_dict is empty', flush=True)
                return []
            
            # Convert action dict to list format
            # GR00T returns action as dict with keys like "action.left_arm", "action.right_arm", etc.
            # Extract the first timestep action and concatenate all action components
            action_list = []
            for key in sorted(action_dict.keys()):
                if key.startswith('action.'):
                    # Get first timestep action (action horizon is typically > 1)
                    action_component = action_dict[key]
                    
                    if isinstance(action_component, np.ndarray):
                        if action_component.size == 0:
                            print(f'Warning: action_component {key} is empty')
                            continue
                        # Take first timestep
                        if len(action_component.shape) > 1:
                            if action_component.shape[0] > 0:
                                action_slice = action_component[0]
                                # Check for NaN
                                if np.isnan(action_slice).any():
                                    print(f'Warning: NaN detected in {key}, shape={action_slice.shape}')
                                action_list.append(action_slice)
                            else:
                                print(f'Warning: action_component {key} has empty first dimension')
                        else:
                            # Check for NaN
                            if np.isnan(action_component).any():
                                print(f'Warning: NaN detected in {key}')
                            action_list.append(action_component)
                    elif isinstance(action_component, torch.Tensor):
                        action_np = action_component.cpu().numpy()
                        if action_np.size == 0:
                            print(f'Warning: action_component {key} is empty')
                            continue
                        if len(action_np.shape) > 1:
                            if action_np.shape[0] > 0:
                                action_slice = action_np[0]
                                # Check for NaN
                                if np.isnan(action_slice).any():
                                    print(f'Warning: NaN detected in {key}, shape={action_slice.shape}')
                                action_list.append(action_slice)
                            else:
                                print(f'Warning: action_component {key} has empty first dimension')
                        else:
                            # Check for NaN
                            if np.isnan(action_np).any():
                                print(f'Warning: NaN detected in {key}')
                            action_list.append(action_np)
            
            # Concatenate all action components
            if action_list:
                try:
                    concatenated = np.concatenate(action_list)
                    # Check for NaN in final result
                    if np.isnan(concatenated).any():
                        print(f'Warning: NaN detected in concatenated action, shape={concatenated.shape}')
                    return concatenated.tolist()
                except Exception as e:
                    print(f'Error concatenating action_list: {e}')
                    print(f'  action_list lengths: {[len(a) if hasattr(a, "__len__") else "N/A" for a in action_list]}')
                    raise
            else:
                # Fallback: try to get any action value
                if len(action_dict) == 0:
                    print('Error: action_dict is empty and no action components found')
                    return []
                
                try:
                    first_action = list(action_dict.values())[0]
                    if isinstance(first_action, np.ndarray):
                        if first_action.size == 0:
                            print('Warning: first_action is empty')
                            return []
                        if len(first_action.shape) > 1:
                            if first_action.shape[0] > 0:
                                result = first_action[0].tolist()
                            else:
                                print('Warning: first_action has empty first dimension')
                                return []
                        else:
                            result = first_action.tolist()
                        # Check for NaN
                        if any(np.isnan(np.array(result))):
                            print(f'Warning: NaN detected in fallback action')
                        return result
                    elif isinstance(first_action, torch.Tensor):
                        action_np = first_action.cpu().numpy()
                        if action_np.size == 0:
                            print('Warning: first_action is empty')
                            return []
                        if len(action_np.shape) > 1:
                            if action_np.shape[0] > 0:
                                result = action_np[0].tolist()
                            else:
                                print('Warning: first_action has empty first dimension')
                                return []
                        else:
                            result = action_np.tolist()
                        # Check for NaN
                        if any(np.isnan(np.array(result))):
                            print(f'Warning: NaN detected in fallback action')
                        return result
                    else:
                        print(f'Warning: Unknown action type: {type(first_action)}')
                        return []
                except IndexError as e:
                    print(f'Error accessing action_dict values: {e}')
                    print(f'  action_dict length: {len(action_dict)}')
                    return []
        except Exception as e:
            print(f'Error in _predict_groot: {e}')
            import traceback
            traceback.print_exc()
            return []
    
    def _preprocess_groot(
            self,
            images: dict[str, np.ndarray],
            state: list,
            task_instruction: str = None) -> dict:
        """Preprocess observation for GR00T policy.
        
        GR00T expects:
        - video.<key>: (T, H, W, C) uint8 or (B, T, H, W, C) uint8
        - state.<key>: (T, D) float64 or (B, T, D) float64
        - annotation.human.task_description: list[str] or (B,) list[str]
        """
        # Validate inputs
        if not images:
            raise ValueError('images dict is empty')
        
        # Check for NaN/Inf in input images
        for key, img in images.items():
            if not isinstance(img, np.ndarray):
                raise ValueError(f'Image {key} is not a numpy array, got {type(img)}')
            if img.size == 0:
                raise ValueError(f'Image {key} is empty')
            if np.isnan(img).any():
                raise ValueError(f'NaN detected in input image {key}')
            if np.isinf(img).any():
                raise ValueError(f'Inf detected in input image {key}')
        
        # Check for NaN/Inf in input state
        if state:
            state_array_check = np.array(state, dtype=np.float64)
            if np.isnan(state_array_check).any():
                raise ValueError(f'NaN detected in input state')
            if np.isinf(state_array_check).any():
                raise ValueError(f'Inf detected in input state')
        
        observation = {}
        
        # Get expected video keys from modality config
        expected_video_keys = []
        if hasattr(self.policy, '_modality_config') and 'video' in self.policy._modality_config:
            expected_video_keys = self.policy._modality_config['video'].modality_keys
        elif hasattr(self.policy, 'modality_config') and 'video' in self.policy.modality_config:
            expected_video_keys = self.policy.modality_config['video'].modality_keys
        
        # Create mapping from actual camera keys to expected keys
        # Common mappings: cam_head -> ego_view, or use first available camera
        video_key_mapping = {}
        if expected_video_keys:
            # Extract base key names (remove 'video.' prefix)
            expected_base_keys = [k.replace('video.', '') for k in expected_video_keys]
            
            # Try to map actual keys to expected keys
            # Priority: cam_head -> ego_view, cam_wrist_left -> wrist_left, etc.
            key_mapping_rules = {
                'cam_head': 'ego_view',
                'head': 'ego_view',
                'ego': 'ego_view',
                'cam_wrist_left': 'wrist_left',
                'wrist_left': 'wrist_left',
                'cam_wrist_right': 'wrist_right',
                'wrist_right': 'wrist_right',
            }
            
            # Map each expected key
            for expected_base in expected_base_keys:
                mapped = False
                # Try direct match first
                if expected_base in images:
                    video_key_mapping[expected_base] = expected_base
                    mapped = True
                else:
                    # Try mapping rules
                    for actual_key, mapped_key in key_mapping_rules.items():
                        if actual_key in images and mapped_key == expected_base:
                            video_key_mapping[expected_base] = actual_key
                            mapped = True
                            break
                
                # If still not mapped, use first available camera
                if not mapped and images:
                    first_key = list(images.keys())[0]
                    video_key_mapping[expected_base] = first_key
                    print(f'Warning: Mapping first available camera "{first_key}" to expected key "{expected_base}"')
        else:
            # No expected keys from config, use actual keys as-is
            for key in images.keys():
                video_key_mapping[key] = key
        
        # Convert images to GR00T format: (1, H, W, C) uint8
        # GR00T expects video keys like 'video.ego_view', 'video.wrist_left', etc.
        for expected_key, actual_key in video_key_mapping.items():
            if actual_key not in images:
                print(f'Warning: Camera key "{actual_key}" not found in images. Available keys: {list(images.keys())}')
                continue
                
            value = images[actual_key].copy()  # Make a copy to avoid modifying original
            
            # Validate image values
            if np.isnan(value).any():
                raise ValueError(f'NaN detected in image {actual_key} before processing')
            if np.isinf(value).any():
                raise ValueError(f'Inf detected in image {actual_key} before processing')
            
            # Ensure uint8 and correct shape
            if value.dtype != np.uint8:
                if value.max() <= 1.0:
                    value = (value * 255).astype(np.uint8)
                else:
                    value = value.astype(np.uint8)
            
            # Validate after conversion
            if np.isnan(value).any():
                raise ValueError(f'NaN detected in image {actual_key} after uint8 conversion')
            if np.isinf(value).any():
                raise ValueError(f'Inf detected in image {actual_key} after uint8 conversion')
            
            # Reshape to (1, H, W, C) if needed
            # GR00T expects (T, H, W, C) or (B, T, H, W, C) format
            if len(value.shape) == 3:  # (H, W, C)
                value = value[np.newaxis, ...]  # (1, H, W, C)
            elif len(value.shape) == 4:
                if value.shape[0] == 1:  # Already (1, H, W, C)
                    pass
                else:
                    # Might be (T, H, W, C), ensure first dim is 1
                    value = value[np.newaxis, ...]  # (1, T, H, W, C)
            else:
                # If (C, H, W), convert to (H, W, C)
                if len(value.shape) == 3 and (value.shape[0] == 3 or value.shape[0] == 1):
                    value = np.transpose(value, (1, 2, 0))
                    value = value[np.newaxis, ...]  # (1, H, W, C)
            
            # Store with 'video.' prefix as GR00T expects
            video_key = f'video.{expected_key}'
            observation[video_key] = value
            # print(f'Added {video_key} with shape {value.shape}')  # Reduced verbosity
        
        # Convert state to GR00T format: (1, D) float64
        state_array = np.array(state, dtype=np.float64)
        if len(state_array.shape) == 0:
            # Empty state array
            state_array = np.zeros((1, 0), dtype=np.float64)
        elif len(state_array.shape) == 1:
            state_array = state_array[np.newaxis, ...]
        
        # Validate state array shape
        if len(state_array.shape) < 2:
            raise ValueError(f'Invalid state array shape: {state_array.shape}. Expected at least 2 dimensions.')
        
        state_dim = state_array.shape[1]
        if state_dim == 0:
            print('Warning: State array is empty (0 dimensions)')
        
        # Get state keys from modality config if available
        expected_state_keys = []
        state_modality_config = None
        if hasattr(self.policy, '_modality_config') and 'state' in self.policy._modality_config:
            state_modality_config = self.policy._modality_config['state']
            expected_state_keys = state_modality_config.modality_keys
        elif hasattr(self.policy, 'modality_config') and 'state' in self.policy.modality_config:
            state_modality_config = self.policy.modality_config['state']
            expected_state_keys = state_modality_config.modality_keys
        
        # Try to get metadata for state dimensions
        state_metadata = None
        if hasattr(self.policy, 'metadata') and self.policy.metadata:
            if hasattr(self.policy.metadata, 'modalities'):
                modalities = self.policy.metadata.modalities
                # DatasetModalities is an object, not a dict, so use getattr or direct attribute access
                if hasattr(modalities, 'state'):
                    state_metadata = modalities.state
                else:
                    # Try to access as attribute
                    try:
                        state_metadata = getattr(modalities, 'state', None)
                    except AttributeError:
                        state_metadata = None
        
        # Extract base key names (remove 'state.' prefix)
        expected_state_base_keys = []
        if expected_state_keys:
            expected_state_base_keys = [k.replace('state.', '') for k in expected_state_keys]
            print(f'Expected state keys from modality config: {expected_state_base_keys}')
        
        print(f'[GR00T] Input state dimension: {state_dim}')
        print(f'[GR00T] Input state array sample (first 22): {state_array[0, :min(22, state_dim)].tolist() if state_dim > 0 else "empty"}')
        
        # Map state data to expected keys
        if expected_state_base_keys:
            # If only one state key expected
            if len(expected_state_base_keys) == 1:
                state_key = expected_state_base_keys[0]
                
                # Try to get expected dimension from metadata
                expected_dim = None
                if state_metadata:
                    # Check if state_metadata is a dict or object
                    key_metadata = None
                    if isinstance(state_metadata, dict):
                        key_metadata = state_metadata.get(state_key)
                    elif hasattr(state_metadata, state_key):
                        key_metadata = getattr(state_metadata, state_key)
                    
                    if key_metadata:
                        if hasattr(key_metadata, 'shape'):
                            shape = key_metadata.shape
                            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                                expected_dim = shape[-1]
                            elif not isinstance(shape, (list, tuple)):
                                expected_dim = shape
                        elif isinstance(key_metadata, dict) and 'shape' in key_metadata:
                            shape = key_metadata['shape']
                            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                                expected_dim = shape[-1]
                            elif not isinstance(shape, (list, tuple)):
                                expected_dim = shape
                
                if expected_dim and state_dim != expected_dim:
                    print(f'Warning: State dimension mismatch. Expected {expected_dim} for state.{state_key}, got {state_dim}')
                    # Try to pad or truncate to match expected dimension
                    if state_dim < expected_dim:
                        # Pad with zeros
                        padding = np.zeros((state_array.shape[0], expected_dim - state_dim), dtype=np.float64)
                        state_array = np.concatenate([state_array, padding], axis=1)
                        print(f'Padded state array from {state_dim} to {expected_dim}')
                    else:
                        # Truncate
                        state_array = state_array[:, :expected_dim]
                        print(f'Truncated state array from {state_dim} to {expected_dim}')
                
                observation[f'state.{state_key}'] = state_array
            else:
                # Multiple state keys expected - need to split state array
                # Try to get dimensions from metadata
                state_key_dims = {}
                total_expected_dim = 0
                
                for state_key in expected_state_base_keys:
                    expected_dim = None
                    if state_metadata:
                        # Check if state_metadata is a dict or object
                        key_metadata = None
                        if isinstance(state_metadata, dict):
                            key_metadata = state_metadata.get(state_key)
                        elif hasattr(state_metadata, state_key):
                            key_metadata = getattr(state_metadata, state_key)
                        
                        if key_metadata:
                            if hasattr(key_metadata, 'shape'):
                                shape = key_metadata.shape
                                if isinstance(shape, (list, tuple)) and len(shape) > 0:
                                    expected_dim = shape[-1]
                                elif not isinstance(shape, (list, tuple)):
                                    expected_dim = shape
                            elif isinstance(key_metadata, dict) and 'shape' in key_metadata:
                                shape = key_metadata['shape']
                                if isinstance(shape, (list, tuple)) and len(shape) > 0:
                                    expected_dim = shape[-1]
                                elif not isinstance(shape, (list, tuple)):
                                    expected_dim = shape
                    
                    if expected_dim:
                        state_key_dims[state_key] = expected_dim
                        total_expected_dim += expected_dim
                
                # If we have dimension info, use it; otherwise split evenly
                if state_key_dims and total_expected_dim > 0:
                    # If state has more dimensions than expected, truncate to expected dimension
                    # This happens when input includes extra joints (e.g., head, mobile) that model doesn't expect
                    if state_dim > total_expected_dim:
                        print(f'[GR00T] WARNING: Input state has {state_dim} dims, but model expects {total_expected_dim} dims.')
                        print(f'[GR00T] Truncating state from {state_dim} to {total_expected_dim} dimensions.')
                        print(f'[GR00T] Truncated values (indices {total_expected_dim}:{state_dim}): {state_array[0, total_expected_dim:].tolist()}')
                        state_array = state_array[:, :total_expected_dim]
                        state_dim = total_expected_dim
                    
                    start_idx = 0
                    for state_key in expected_state_base_keys:
                        if state_key in state_key_dims:
                            key_dim = state_key_dims[state_key]
                            end_idx = start_idx + key_dim
                            
                            if end_idx <= state_dim:
                                state_slice = state_array[:, start_idx:end_idx]
                            else:
                                # Not enough dimensions, pad with zeros
                                state_slice = state_array[:, start_idx:]
                                padding = np.zeros((state_slice.shape[0], key_dim - state_slice.shape[1]), dtype=np.float64)
                                state_slice = np.concatenate([state_slice, padding], axis=1)
                            
                            observation[f'state.{state_key}'] = state_slice
                            start_idx = end_idx
                            print(f'Mapped state[{start_idx-key_dim}:{min(end_idx, state_dim)}] (dim={key_dim}) to state.{state_key}')
                        else:
                            # No dimension info, split remaining evenly
                            try:
                                key_index = expected_state_base_keys.index(state_key)
                                remaining_keys = len(expected_state_base_keys) - key_index
                            except ValueError:
                                # state_key not in list, skip
                                print(f'Warning: state_key "{state_key}" not found in expected_state_base_keys')
                                continue
                            
                            if remaining_keys <= 0:
                                print(f'Warning: No remaining keys to process for "{state_key}"')
                                continue
                            
                            remaining_dim = max(0, state_dim - start_idx)
                            if remaining_dim == 0:
                                # No more dimensions left, pad with zeros
                                key_dim = 1  # Default to 1 dimension
                                state_slice = np.zeros((state_array.shape[0], key_dim), dtype=np.float64)
                            else:
                                key_dim = max(1, remaining_dim // remaining_keys)
                                end_idx = min(start_idx + key_dim, state_dim)
                                
                                if start_idx >= state_dim:
                                    # Already past the end, pad with zeros
                                    state_slice = np.zeros((state_array.shape[0], key_dim), dtype=np.float64)
                                else:
                                    state_slice = state_array[:, start_idx:end_idx]
                            
                            observation[f'state.{state_key}'] = state_slice
                            start_idx = start_idx + key_dim
                            print(f'Mapped state[{start_idx-key_dim}:{min(start_idx, state_dim)}] (dim={key_dim}) to state.{state_key}')
                    
                    # Check if there are remaining dimensions not mapped
                    if start_idx < state_dim:
                        remaining_dim = state_dim - start_idx
                        print(f'[GR00T] WARNING: {remaining_dim} dimensions not mapped! State has {state_dim} dims, but only {start_idx} were used.')
                        print(f'[GR00T] Remaining state values: {state_array[0, start_idx:].tolist()}')
                        print(f'[GR00T] This might cause model to produce NaN outputs!')
                else:
                    # No dimension info, split evenly
                    if len(expected_state_base_keys) == 0:
                        print('Warning: No expected state keys, skipping state mapping')
                    else:
                        dims_per_key = state_dim // len(expected_state_base_keys) if state_dim > 0 else 0
                        remainder = state_dim % len(expected_state_base_keys) if state_dim > 0 else 0
                        
                        start_idx = 0
                        for i, state_key in enumerate(expected_state_base_keys):
                            key_dim = dims_per_key + (1 if i < remainder else 0)
                            end_idx = min(start_idx + key_dim, state_dim)
                            
                            if start_idx >= state_dim:
                                # Past the end, pad with zeros
                                state_slice = np.zeros((state_array.shape[0], key_dim), dtype=np.float64)
                            else:
                                state_slice = state_array[:, start_idx:end_idx]
                            
                            observation[f'state.{state_key}'] = state_slice
                            
                            start_idx = end_idx
                            print(f'Mapped state[{start_idx-key_dim}:{end_idx}] (dim={key_dim}) to state.{state_key}')
        else:
            # No expected keys from config, try common mappings
            # Common pattern: left_arm is typically 7 dimensions (6 joints + 1 gripper)
            # If state_dim is 12, it might be left_arm (7) + something else (5)
            # Or it might be a different configuration
            
            # Try common patterns
            if state_dim == 7:
                observation['state.left_arm'] = state_array
                print(f'Using state.left_arm (dim={state_dim})')
            elif state_dim == 14:
                # Likely left_arm (7) + right_arm (7)
                observation['state.left_arm'] = state_array[:, :7]
                observation['state.right_arm'] = state_array[:, 7:]
                print(f'Split state array (dim={state_dim}) into left_arm (7) and right_arm (7)')
            elif state_dim == 12:
                # Could be left_arm (7) + partial right_arm (5)
                # Or something else - try left_arm only with first 7 dimensions
                observation['state.left_arm'] = state_array[:, :7]
                print(f'Using first 7 dimensions for state.left_arm (total dim={state_dim})')
            else:
                # Fallback: use left_arm with all dimensions
                observation['state.left_arm'] = state_array
                print(f'Using state.left_arm as fallback (dim={state_dim})')
        
        # Add task instruction if provided
        if task_instruction is not None:
            observation['annotation.human.task_description'] = [task_instruction]
        
        # Debug: print final observation keys (only if there's an issue)
        # print(f'Final observation keys: {list(observation.keys())}')
        # for key in observation.keys():
        #     if isinstance(observation[key], np.ndarray):
        #         print(f'  {key}: shape={observation[key].shape}, dtype={observation[key].dtype}')
        #     else:
        #         print(f'  {key}: type={type(observation[key])}, value={observation[key]}')
        
        return observation

    def _get_policy_class(self, name: str) -> PreTrainedPolicy:
        if name == 'tdmpc':
            from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

            return TDMPCPolicy
        elif name == 'diffusion':
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

            return DiffusionPolicy
        elif name == 'act':
            from lerobot.policies.act.modeling_act import ACTPolicy

            return ACTPolicy
        elif name == 'vqbet':
            from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

            return VQBeTPolicy
        elif name == 'pi0':
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy

            return PI0Policy
        elif name == 'pi0fast':
            from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
            return PI0FASTPolicy
        elif name == 'smolvla':
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            return SmolVLAPolicy
        else:
            raise NotImplementedError(
                f'Policy with name {name} is not implemented.')

    @staticmethod
    def get_available_policies() -> list[str]:
        return [
            'tdmpc',
            'diffusion',
            'act',
            'vqbet',
            'pi0',
            'pi0fast',
            'smolvla',
            'groot',
            'groot-n1',
            'gr00t',
        ]

    @staticmethod
    def get_saved_policies():
        import os
        import json

        home_dir = os.path.expanduser('~')
        hub_dir = os.path.join(home_dir, '.cache/huggingface/hub')
        models_folder_list = [d for d in os.listdir(hub_dir) if d.startswith('models--')]

        saved_policy_path = []
        saved_policy_type = []

        for model_folder in models_folder_list:
            model_path = os.path.join(hub_dir, model_folder)
            snapshots_path = os.path.join(model_path, 'snapshots')

            # Check if snapshots directory exists
            if os.path.exists(snapshots_path) and os.path.isdir(snapshots_path):
                # Get list of folders inside snapshots directory
                snapshot_folders = [
                    d for d in os.listdir(snapshots_path)
                    if os.path.isdir(os.path.join(snapshots_path, d))
                ]

            # Check if pretrained_model folder exists in each snapshot folder
            for snapshot_folder in snapshot_folders:
                snapshot_path = os.path.join(snapshots_path, snapshot_folder)
                pretrained_model_path = os.path.join(snapshot_path, 'pretrained_model')

                # If pretrained_model folder exists, add to saved_policies
                if os.path.exists(pretrained_model_path) and os.path.isdir(pretrained_model_path):
                    config_path = os.path.join(pretrained_model_path, 'config.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                if 'type' in config:
                                    saved_policy_path.append(pretrained_model_path)
                                    saved_policy_type.append(config['type'])
                                elif 'model_type' in config:
                                    saved_policy_path.append(pretrained_model_path)
                                    saved_policy_type.append(config['model_type'])
                        except (json.JSONDecodeError, IOError):
                            # If config.json cannot be read, store path only
                            print('File IO Errors : ', IOError)

        return saved_policy_path, saved_policy_type
