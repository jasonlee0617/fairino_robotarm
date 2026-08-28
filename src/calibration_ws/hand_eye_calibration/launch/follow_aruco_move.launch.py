#!/usr/bin/env python3
"""Real-hardware low-rate global MoveIt tracking of ArUco marker ID 1."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
)

_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

from handeye_launch_utils import (  # noqa: E402
    default_storage_directory,
    load_handeye_profile,
    value,
)
from myrobot_common.camera.launch import camera_launch


_LAUNCH_ARGUMENT_SPECS = (
    ("use_sim_time", "false", "实机默认不使用 /clock。"),
    ("calibration_type", "eye_in_hand", "手眼标定模式。"),
    ("camera_serial_no", "", "RealSense 设备序列号；留空时自动选择。"),
    ("color_profile", "1280x720x30", "RealSense 彩色流 profile。"),
    ("depth_profile", "848x480x30", "RealSense 深度流 profile。"),
    ("use_rviz", "true", "是否启动 MoveIt RViz。"),
    ("rviz_config", "", "跟随 RViz 配置绝对路径。"),
    ("debug", "false", "是否开启 MoveIt 调试。"),
    ("allow_trajectory_execution", "true", "是否允许真实轨迹执行。"),
    ("publish_monitored_planning_scene", "true", "是否发布 monitored planning scene。"),
    ("monitor_dynamics", "false", "是否监控机器人动力学。"),
    ("capabilities", "", "附加 MoveIt capabilities。"),
    ("disable_capabilities", "", "禁用的 MoveIt capabilities。"),
    ("ik_plugin", "fairino", "选择执行 MoveIt IK 客户端。"),
    ("planning_pipeline_id", "fairino", "选择规划流水线。"),
    ("planner_id", "tube_birrt*", "选择规划器。"),
    ("move_group_ready_timeout_sec", "10.0", "MoveIt 就绪等待时间。"),
    ("arm_max_velocity", "0.2", "机械臂最大速度比例。"),
    ("arm_max_acceleration", "0.2", "机械臂最大加速度比例。"),
    ("allowed_planning_time", "15.0", "规划时间上限。"),
    ("position_tolerance", "0.005", "位置容差。"),
    ("orientation_tolerance", "0.005", "姿态容差。"),
    ("allowed_start_tolerance", "0.1", "起点容差。"),
)
_YAML_DEFAULTS = launch_defaults_as_strings(
    load_launch_parameters_yaml(
        "hand_eye_calibration", "config/follow_aruco_move_params.yaml", None
    )
)
_DEFAULT_RVIZ_CONFIG = os.path.join(
    get_package_share_directory("hand_eye_calibration"),
    "rviz",
    "follow_aruco_move.rviz",
)
_LAUNCH_ARGUMENT_SPECS = tuple(
    (name, _YAML_DEFAULTS.get(name, _DEFAULT_RVIZ_CONFIG if name == "rviz_config" else default), description)
    for name, default, description in _LAUNCH_ARGUMENT_SPECS
)


def _declare_launch_arguments():
    return [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in _LAUNCH_ARGUMENT_SPECS
    ]


def _launch_setup(context, *_args, **_kwargs):
    calibration_type = value(context, "calibration_type")
    if calibration_type not in {"eye_in_hand", "eye_on_base"}:
        raise ValueError("calibration_type must be eye_in_hand or eye_on_base")
    profile = load_handeye_profile(calibration_type)
    follower_params = load_node_parameters_yaml(
        "hand_eye_calibration",
        "config/follow_aruco_move_params.yaml",
        "aruco_marker_follower",
    )
    for name in (
        "ik_plugin", "planning_pipeline_id", "planner_id", "move_group_ready_timeout_sec",
        "arm_max_velocity", "arm_max_acceleration", "allowed_planning_time",
        "position_tolerance", "orientation_tolerance", "allowed_start_tolerance",
    ):
        follower_params[name] = launch_parameter_value(value(context, name), follower_params[name])
    source_params = {
        "marker_id": profile["marker_id"],
        "aruco_topic": "/aruco_markers",
        "output_topic": "/aruco_marker/pose",
    }
    handeye_params = {
        "camera_link_frame": profile["camera_link_frame"],
        "publish_child_frame": profile["publish_camera_link_frame"],
        "calibration_name": profile["calibration_name"],
        "storage_directory": default_storage_directory("real"),
        "publish_rate_hz": 10.0,
        "use_tracking_to_camera_link_compensation": profile[
            "use_tracking_to_camera_link_compensation"
        ],
        "use_sim_time": LaunchConfiguration("use_sim_time"),
    }
    use_sim_time = LaunchConfiguration("use_sim_time")

    camera = camera_launch(
        "realsense",
        realsense_args={
            "serial_no": value(context, "camera_serial_no"),
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.color_profile": value(context, "color_profile"),
            "depth_module.depth_profile": value(context, "depth_profile"),
            "align_depth.enable": "true",
            "enable_sync": "true",
            "temporal_filter.enable": "true",
            "spatial_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        },
    )
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("fairino_arm_moveit_config"),
                "launch",
                "moveit_hardware.launch.py",
            )
        ),
        launch_arguments={
            **{
                name: value(context, name)
                for name in (
                    "use_rviz",
                    "rviz_config",
                    "debug",
                    "allow_trajectory_execution",
                    "publish_monitored_planning_scene",
                    "monitor_dynamics",
                    "capabilities",
                    "disable_capabilities",
                )
            },
            "execution_ik": value(context, "ik_plugin"),
            "execution_pipeline": value(context, "planning_pipeline_id"),
        }.items(),
    )
    return [
        camera,
        moveit,
        Node(
            package="hand_eye_calibration",
            executable="handeye_publisher.py",
            name="handeye_publisher",
            output="screen",
            parameters=[handeye_params],
        ),
        Node(
            package="ros2_aruco",
            executable="aruco_node",
            name="aruco_node",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time},
                os.path.join(
                    get_package_share_directory("hand_eye_calibration"),
                    "config",
                    "aruco_parameters.yaml",
                ),
            ],
        ),
        Node(
            package="hand_eye_calibration",
            executable="aruco_marker_pose_publisher.py",
            name="aruco_marker_pose_publisher",
            output="screen",
            parameters=[source_params, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="hand_eye_calibration",
            executable="follow_aruco_marker.py",
            name="aruco_marker_follower",
            output="screen",
            parameters=[follower_params, {"use_sim_time": use_sim_time}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        *_declare_launch_arguments(),
        OpaqueFunction(function=_launch_setup),
    ])
