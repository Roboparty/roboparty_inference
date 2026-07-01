# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

##launch file
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import re

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

def launch_setup(context, *args, **kwargs):
    robot = LaunchConfiguration("robot").perform(context)
    policy = LaunchConfiguration("policy").perform(context)
    if not NAME_PATTERN.fullmatch(robot):
        raise ValueError(f"Invalid robot name: {robot}")
    if not NAME_PATTERN.fullmatch(policy):
        raise ValueError(f"Invalid policy name: {policy}")

    policy_file = policy if policy.endswith(".yaml") else f"{policy}.yaml"

    robot_dir = os.path.join(
        get_package_share_directory("roboparty_inference"),
        "robots",
        robot,
    )
    robot_config = os.path.join(robot_dir, "robot.yaml")
    policy_config = os.path.join(robot_dir, "configs", policy_file)

    if not os.path.isfile(robot_config):
        raise FileNotFoundError(f"Robot config not found: {robot_config}")
    if not os.path.isfile(policy_config):
        raise FileNotFoundError(f"Inference config not found: {policy_config}")

    return [
        Node(
            package="roboparty_inference",
            executable="inference_node",
            name="inference_node",
            parameters=[
                policy_config,
                {
                    "robot_name": robot,
                    "policy_name": policy,
                    "robot_config": robot_config,
                    "model_dir": os.path.join(robot_dir, "models"),
                    "motion_dir": os.path.join(robot_dir, "motions"),
                },
            ],
            output="screen",
            # prefix=["xterm -e gdb -ex run --args"],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot", default_value="rpo"),
            DeclareLaunchArgument("policy", default_value="default"),
            OpaqueFunction(function=launch_setup),
        ]
    )
