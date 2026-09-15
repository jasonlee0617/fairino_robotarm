#!/usr/bin/env python3
"""真实相机感知与手眼 TF 演示入口."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


_LAUNCH_ARGUMENT_SPECS = (
    ("use_sim_time", "false", "是否使用仿真时间。"),
    ("depth_profile", "640x480x30", "D435 深度流配置。"),
    ("color_profile", "640x480x30", "D435 彩色流配置。"),
    ("calibration_name", "robot_calibration", "手眼标定名称。"),
    (
        "storage_directory",
        os.path.expandvars("$HOME/my-workspace/fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real"),
        "标定结果保存目录。",
    ),
)
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name, *_ in _LAUNCH_ARGUMENT_SPECS
}


def _declare_launch_arguments():
    return [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in _LAUNCH_ARGUMENT_SPECS
    ]


def generate_launch_description():
    grasping_share = get_package_share_directory("visual_grasping_bringup")
    moveit_share = get_package_share_directory("fairino_arm_moveit_config")

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
            "depth_module.profile": _LAUNCH_CONFIGURATIONS["depth_profile"],
            "rgb_camera.profile": _LAUNCH_CONFIGURATIONS["color_profile"],
            "pointcloud.enable": "true",
            "temporal_filter.enable": "true",
            "spatial_filter.enable": "true",
        }.items(),
    )
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, "launch", "demo.launch.py")
        ),
        launch_arguments={
            "use_rviz": "true",
            "rviz_config": os.path.join(
                grasping_share, "rviz", "perception.demo.rviz"
            ),
        }.items(),
    )
    handeye_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
                "calibration_name": _LAUNCH_CONFIGURATIONS["calibration_name"],
                "storage_directory": _LAUNCH_CONFIGURATIONS["storage_directory"],
            }
        ],
    )
    return LaunchDescription(
        [*_declare_launch_arguments(), realsense_launch, moveit_launch, handeye_publisher]
    )
