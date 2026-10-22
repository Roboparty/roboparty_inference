# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import Parameter
from ament_index_python.packages import get_package_prefix, get_package_share_directory
import os
import re
import shlex
import sys

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

    mode = os.environ.get("ROBOT_CONTROL_ISOLATION", "isolated")
    if mode not in ("affinity", "isolated"):
        raise ValueError(f"Unknown ROBOT_CONTROL_ISOLATION: {mode}")
    # Resolve the deployment layout once; C++ threads apply it themselves.
    defaults = {"ROBOT_INFERENCE_CPU": 4, "ROBOT_CONTROL_CPU": 5, "ROBOT_CAN_CPU": 6}
    thread_env = {}
    for name, default in defaults.items():
        thread_env[name] = str(int(os.environ.get(name, default)))
    prefix = None
    if mode == "isolated":
        import yaml
        with open(robot_config) as stream:
            motors = yaml.safe_load(stream)["motors"]
        interfaces = set(zip(motors["motor_interface_type"], motors["motor_interface"]))
        runtime = os.path.join(get_package_prefix("roboparty_inference"),
                               "lib", "roboparty_inference", "realtime_run.py")
        command = [sys.executable, runtime,
                   "--can-count", str(sum(kind == "can" for kind, _ in interfaces)),
                   "--canfd-count", str(sum(kind == "canfd" for kind, _ in interfaces)),
                   "--"]
        if os.geteuid() != 0:
            command = ["sudo", "-n", "-E", "env",
                       "LD_LIBRARY_PATH=" + os.environ.get("LD_LIBRARY_PATH", "")] + command
        prefix = shlex.join(command)

    return [
        Node(
            package="roboparty_inference",
            executable="inference_node",
            name="inference_node",
            parameters=[
                policy_config,
                Parameter("robot_name", robot, value_type=str),
                Parameter("policy_name", policy, value_type=str),
                Parameter("robot_config", robot_config, value_type=str),
                Parameter("model_dir", os.path.join(robot_dir, "models"), value_type=str),
                Parameter("motion_dir", os.path.join(robot_dir, "motions"), value_type=str),
                Parameter("latent_dir", os.path.join(robot_dir, "latents"), value_type=str),
            ],
            output="screen",
            additional_env=thread_env,
            prefix=prefix,
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
