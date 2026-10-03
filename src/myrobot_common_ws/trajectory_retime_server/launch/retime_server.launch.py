#!/usr/bin/env python3
import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import xacro


def _build_node(context):
    xacro_file = os.path.join(
        get_package_share_directory("fairino_arm_moveit_config"),
        "config",
        "fairino_arm_moveit_descriptions.urdf.xacro",
    )
    srdf_file = os.path.join(
        get_package_share_directory("fairino_arm_moveit_config"),
        "config",
        "fairino_arm_moveit_descriptions.srdf",
    )
    kin_file = os.path.join(
        get_package_share_directory("fairino_arm_moveit_config"),
        "config",
        "kinematics.yaml",
    )
    limits_file = os.path.join(
        get_package_share_directory("fairino_arm_moveit_config"),
        "config",
        "joint_limits.yaml",
    )

    robot_description_xml = LaunchConfiguration("robot_description").perform(context)
    robot_semantic_xml = LaunchConfiguration("robot_description_semantic").perform(context)
    robot_kinematics_str = LaunchConfiguration("robot_description_kinematics").perform(context)

    if not robot_description_xml:
        if not os.path.exists(xacro_file):
            raise RuntimeError(f"Xacro file not found: {xacro_file}")
        robot_description_xml = xacro.process_file(xacro_file).toxml()

    if not robot_semantic_xml:
        if not os.path.exists(srdf_file):
            raise RuntimeError(f"SRDF file not found: {srdf_file}")
        with open(srdf_file, "r", encoding="utf-8") as f:
            robot_semantic_xml = f.read()

    robot_kinematics = None
    if robot_kinematics_str:
        robot_kinematics = yaml.safe_load(robot_kinematics_str)
    if not isinstance(robot_kinematics, dict):
        if not os.path.exists(kin_file):
            raise RuntimeError(f"Kinematics file not found: {kin_file}")
        with open(kin_file, "r", encoding="utf-8") as f:
            robot_kinematics = yaml.safe_load(f)
    if not isinstance(robot_kinematics, dict):
        robot_kinematics = {}
    with open(limits_file, "r", encoding="utf-8") as f:
        robot_description_planning = yaml.safe_load(f) or {}

    retime_server = Node(
        package="trajectory_retime_server",
        executable="retime_server",
        name="trajectory_retime_server",
        output="screen",
        parameters=[
            {"service_name": LaunchConfiguration("service_name")},
            {"robot_description": robot_description_xml},
            {"robot_description_semantic": robot_semantic_xml},
            {"robot_description_kinematics": robot_kinematics},
            {"robot_description_planning": robot_description_planning},
        ],
    )
    return [retime_server]


def generate_launch_description():
    xacro_file = os.path.join(
        get_package_share_directory("fairino_arm_moveit_config"),
        "config",
        "fairino_arm_moveit_descriptions.urdf.xacro",
    )
    srdf_file = os.path.join(
        get_package_share_directory("fairino_arm_moveit_config"),
        "config",
        "fairino_arm_moveit_descriptions.srdf",
    )
    kin_file = os.path.join(
        get_package_share_directory("fairino_arm_moveit_config"),
        "config",
        "kinematics.yaml",
    )

    robot_description_default = ""
    if os.path.exists(xacro_file):
        robot_description_default = xacro.process_file(xacro_file).toxml()

    robot_semantic_default = ""
    if os.path.exists(srdf_file):
        with open(srdf_file, "r", encoding="utf-8") as f:
            robot_semantic_default = f.read()

    robot_kinematics_default = ""
    if os.path.exists(kin_file):
        with open(kin_file, "r", encoding="utf-8") as f:
            robot_kinematics_default = yaml.safe_dump(yaml.safe_load(f))

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "service_name",
                default_value="/retime_trajectory",
                description="Service name for trajectory_retime_server",
            ),
            DeclareLaunchArgument(
                "robot_description",
                default_value=robot_description_default,
                description="URDF XML string for trajectory_retime_server",
            ),
            DeclareLaunchArgument(
                "robot_description_semantic",
                default_value=robot_semantic_default,
                description="SRDF XML string for trajectory_retime_server",
            ),
            DeclareLaunchArgument(
                "robot_description_kinematics",
                default_value=robot_kinematics_default,
                description="YAML string for robot_description_kinematics parameter",
            ),
            OpaqueFunction(function=_build_node),
        ]
    )
