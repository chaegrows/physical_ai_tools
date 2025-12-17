#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages

setup(
    name="gr00t_bridge_multi",
    version="0.1.0",
    description="GR00T Multi-Camera Inference Bridge for ROS2",
    author="ROBOTIS",
    author_email="support@robotis.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "gr00t_multicam_inference_node = gr00t_bridge_multi.gr00t_inference_node:main",
        ],
    },
    python_requires=">=3.8",
)
