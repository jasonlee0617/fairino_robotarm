#!/usr/bin/env python3
"""Gazebo demo for MPC dynamic-obstacle avoidance and tube-BiRRT replanning."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
)

THIS_DIR = os.path.dirname(__file__)
SHARE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SCRIPTS_DIR = os.path.join(os.path.dirname(THIS_DIR), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
if SHARE_DIR not in sys.path:
    sys.path.append(SHARE_DIR)
from mpc_demo_models import build_spawn_actions, load_spawn_config  # noqa: E402
from launch_utils.moveit_stack import build_moveit_config  # noqa: E402
from launch_utils.robot_profiles import load_robot_profile  # noqa: E402


_LAUNCH_ARGUMENT_SPECS = (
    ("robot_profile", "fairino3_v6", "机器人配置名称。", ("fairino3_v6", "fairino_arm_gripper_onbase")),
    ("use_sim_time", "true", "是否使用 Gazebo 的 /clock 仿真时间。", None),
    ("world_name", "empty", "Gazebo 世界名称。", None),
    ("ik_plugin", "fairino", "MoveIt IK 客户端。", ("fairino", "kdl")),
    ("planning_pipeline_id", "fairino", "MoveIt 规划流水线。", ("fairino", "ompl")),
    ("planner_id", "tube_birrt*", "初始和重规划使用的规划器。", None),
    ("arm_max_velocity", "1.0", "MoveIt 演示速度比例。", None),
    ("arm_max_acceleration", "1.0", "MoveIt 演示加速度比例。", None),
    ("allowed_planning_time", "15.0", "MoveIt 演示规划时限。", None),
    ("position_tolerance", "0.005", "MoveIt 演示位置容差。", None),
    ("orientation_tolerance", "0.05", "MoveIt 演示姿态容差。", None),
    ("enable_rviz", "true", "是否启动 RViz。", None),
)
_YAML_LAUNCH_DEFAULTS = launch_defaults_as_strings(
    load_launch_parameters_yaml(
        "myrobot_mpc_avoidance", "config/mpc_avoidance_params.yaml", "sim"
    )
)
_LAUNCH_ARGUMENT_SPECS = tuple(
    (name, _YAML_LAUNCH_DEFAULTS.get(name, default), description, choices)
    for name, default, description, choices in _LAUNCH_ARGUMENT_SPECS
)
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name, *_ in _LAUNCH_ARGUMENT_SPECS
}


def _declare_launch_arguments():
    return [
        DeclareLaunchArgument(
            name,
            default_value=default,
            description=description,
            choices=list(choices) if choices else None,
        )
        for name, default, description, choices in _LAUNCH_ARGUMENT_SPECS
    ]


def _launch_setup(context, *args, **kwargs):
    mpc_share = get_package_share_directory("myrobot_mpc_avoidance")
    simulation_share = get_package_share_directory("myrobot_simulation")
    config = _LAUNCH_CONFIGURATIONS
    profile = load_robot_profile(config["robot_profile"].perform(context))
    obstacle_config = os.path.join(mpc_share, "config", "obstacle_stack.yaml")
    spawn_actions = build_spawn_actions(mpc_share, load_spawn_config(obstacle_config))
    moveit_config = build_moveit_config(profile, enable_camera_model=False)
    controller_topic = f"{profile.arm_controller}/joint_trajectory"

    gazebo_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(simulation_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "use_sim_time": config["use_sim_time"],
            "rviz_config": os.path.join(mpc_share, "rviz", "mpc_avoidance.rviz"),
            "enable_rviz": config["enable_rviz"],
            "robot_profile": profile.name,
            "enable_camera_model": "false",
            "world": config["world_name"],
        }.items(),
    )

    obstacle_sim = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="myrobot_mpc_avoidance",
                executable="obstacle_simulator",
                name="obstacle_simulator",
                output="screen",
                parameters=[
                    {"use_sim_time": True, "world_name": "empty"},
                    load_node_parameters_yaml(
                        "myrobot_mpc_avoidance",
                        "config/mpc_avoidance_params.yaml",
                        "obstacle_simulator",
                        "sim",
                    ),
                    {
                        "use_sim_time": ParameterValue(config["use_sim_time"], value_type=bool),
                        "world_name": config["world_name"],
                        "scenario_config": obstacle_config,
                    },
                ],
            )
        ],
    )
    mpc_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="myrobot_mpc_avoidance",
                executable="mpc_avoidance_node",
                output="screen",
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    {
                        "use_sim_time": True,
                        "solver_type": "nmpc",
                        "robot_profile": profile.name,
                        "group_name": profile.group_name,
                        "joint_names": profile.arm_joints,
                        "controller_topic": controller_topic,
                        "enable_moveit_scene": True,
                        "scenario_config": obstacle_config,
                    },
                    load_node_parameters_yaml(
                        "myrobot_mpc_avoidance",
                        "config/mpc_avoidance_params.yaml",
                        "mpc_avoidance_node",
                        "sim",
                    ),
                    {
                        "use_sim_time": ParameterValue(config["use_sim_time"], value_type=bool),
                        "robot_profile": config["robot_profile"],
                    },
                ],
            )
        ],
    )
    demo_node = TimerAction(
        period=12.0,
        actions=[
            Node(
                package="myrobot_simulation",
                executable="mpc_avoidance_node_sim.py",
                name="mpc_avoidance_demo_node",
                output="screen",
                parameters=[
                    moveit_config.robot_description_semantic,
                    {
                        "use_sim_time": True,
                        "planner_id": "tube_birrt*",
                        "ik_plugin": "fairino",
                        "planning_pipeline_id": "fairino",
                        "robot_profile": profile.name,
                        "group_name": profile.group_name,
                        "ee_link": profile.ee_frame_name,
                        "base_frame": profile.planning_frame,
                        "joint_names": profile.arm_joints,
                        "controller_topic": controller_topic,
                        "move_group_ns_fairino": "/move_group_fairino",
                        "move_group_ns_kdl": "/move_group_kdl",
                        "planning_attempts": 5,
                        "allowed_planning_time": 15.0,
                        "position_tolerance": 0.005,
                        "orientation_tolerance": 0.05,
                        "max_velocity": 1.0,
                        "max_acceleration": 1.0,
                    },
                    load_node_parameters_yaml(
                        "myrobot_mpc_avoidance",
                        "config/mpc_avoidance_params.yaml",
                        "mpc_avoidance_demo_node",
                        "sim",
                    ),
                    {
                        "use_sim_time": ParameterValue(config["use_sim_time"], value_type=bool),
                        "ik_plugin": config["ik_plugin"],
                        "planning_pipeline_id": config["planning_pipeline_id"],
                        "planner_id": config["planner_id"],
                        "max_velocity": ParameterValue(
                            config["arm_max_velocity"], value_type=float
                        ),
                        "max_acceleration": ParameterValue(
                            config["arm_max_acceleration"], value_type=float
                        ),
                        "allowed_planning_time": ParameterValue(
                            config["allowed_planning_time"], value_type=float
                        ),
                        "position_tolerance": ParameterValue(
                            config["position_tolerance"], value_type=float
                        ),
                        "orientation_tolerance": ParameterValue(
                            config["orientation_tolerance"], value_type=float
                        ),
                        "robot_profile": config["robot_profile"],
                    },
                ],
            )
        ],
    )
    return [
        LogInfo(
            msg=(
                f"[mpc_avoidance_demo_sim] profile={profile.name}, "
                f"group={profile.group_name}, controller={controller_topic}"
            )
        ),
        gazebo_moveit_launch,
        *spawn_actions,
        obstacle_sim,
        mpc_node,
        demo_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        *_declare_launch_arguments(),
        OpaqueFunction(function=_launch_setup),
    ])
