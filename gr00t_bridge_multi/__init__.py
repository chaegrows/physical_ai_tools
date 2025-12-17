# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# GR00T Multi-Camera Inference Bridge for ROS2
# 
# This package provides multi-camera GR00T inference support:
# - video.ego_view (head camera)
# - video.cam_wrist_left (left wrist camera)
# - video.cam_wrist_right (right wrist camera)

from .gr00t_policy import Gr00tMultiCamInferenceWrapper, ActionOutput
from .ffw_sg2_multicam_inference_config import FFWSG2MultiCamInferenceConfig

__all__ = [
    "Gr00tMultiCamInferenceWrapper",
    "ActionOutput",
    "FFWSG2MultiCamInferenceConfig",
]
