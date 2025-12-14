#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages

setup(
    name="gr00t_bridge",
    version="0.1.0",
    description="ROS2 inference wrapper for Isaac-GR00T models",
    author="ROBOTIS",
    author_email="",
    packages=find_packages(exclude=["scripts"]),
    py_modules=[
        "gr00t_policy",
        "gr00t_inference_node",
        "ffw_sg2_inference_config",
    ],
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "gr00t_inference_node=gr00t_bridge.gr00t_inference_node:main",
        ],
    },
)
