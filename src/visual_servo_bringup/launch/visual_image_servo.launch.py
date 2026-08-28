#!/usr/bin/env python3
"""Hardware entry point for ArUco four-corner IBVS."""

import os
import sys
from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import load_launch_parameters_yaml, load_yaml
from moveit_configs_utils import MoveItConfigsBuilder
from visual_servo_bringup.image_servo_config import (
    NODE_PARAMETER_DESCRIPTIONS,
    image_servo_moveit_parameters,
    image_servo_parameters,
)


_HANDEYE_LAUNCH_DIR = os.path.join(get_package_share_directory("hand_eye_calibration"), "launch")
if _HANDEYE_LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _HANDEYE_LAUNCH_DIR)

from myrobot_common.camera.launch import camera_launch
from handeye_launch_utils import value  # noqa: E402


_REFERENCE_PATH = str(
    Path.home()
    / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real"
    / "image_servo_aruco_id1.yaml"
)

_LAUNCH_FALLBACKS = {
    "use_sim_time": "false",
    "camera_serial_no": "",
    "color_profile": "640x480x60",
    "depth_profile": "640x480x60",
    "pointcloud_enable": "false",
    "use_rviz": "true",
    "rviz_config": os.path.join(
        get_package_share_directory("visual_servo_bringup"), "rviz", "visual_image_servo.rviz"
    ),
    "reference_path": _REFERENCE_PATH,
    "debug": "false",
    "allow_trajectory_execution": "true",
    "publish_monitored_planning_scene": "true",
    "monitor_dynamics": "false",
    "capabilities": "",
    "disable_capabilities": "",
    "publish_frequency": "100.0",
}
_NODE_PARAMETER_FALLBACKS = {
    "image_topic": "/camera/camera/color/image_raw",
    "camera_info_topic": "/camera/camera/color/camera_info",
    "debug_image_topic": "/visual_image_servo/debug_image",
    "error_topic": "/visual_image_servo/error",
    "marker_dictionary": "DICT_5X5_250",
    "marker_id": 1,
    "marker_size_m": 0.07,
    "base_frame": "base_link",
    "camera_frame": "camera_color_optical_frame",
    "ee_frame": "tool0",
    "servo_ns": "/servo_node",
    "control_rate_hz": 100.0,
    "detector_rate_hz": 20.0,
    "tracker_max_error_px": 12.0,
    "debug_image_rate_hz": 10.0,
    "enable_subpixel_refinement": True,
    "lambda_gain": 0.9,
    "damping": 0.06,
    "max_linear_speed": 0.2,
    "max_angular_speed": 0.2,
    "feature_timeout_sec": 0.15,
    "servo_stop_timeout_sec": 2.0,
    "image_error_tolerance": 0.003,
    "servo_status_halt_codes": [2, 4, 5],
    "auto_start": True,
}
_IMAGE_SERVO_DEFAULTS = {
    **_NODE_PARAMETER_FALLBACKS,
    **image_servo_parameters(),
}
_IMAGE_LAUNCH_DEFAULTS = load_launch_parameters_yaml(
    "visual_servo_bringup", "config/visual_image_servo_params.yaml", "real"
)


def _launch_default(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    return yaml.safe_dump(value, default_flow_style=True).strip()


DEFAULTS = {
    **_LAUNCH_FALLBACKS,
    **{
        name: _launch_default(value)
        for name, value in _IMAGE_LAUNCH_DEFAULTS.items()
        if name in _LAUNCH_FALLBACKS
    },
    "reference_path": _launch_default(_IMAGE_SERVO_DEFAULTS.get("reference_path") or _REFERENCE_PATH),
    **{
        name: _launch_default(default)
        for name, default in _IMAGE_SERVO_DEFAULTS.items()
        if name != "reference_path"
    },
}

DESCRIPTIONS = {
    "use_sim_time": "实机必须保持 false。",
    "camera_serial_no": "RealSense 相机序列号；留空时由驱动选择。",
    "color_profile": "RGB 图像 profile。",
    "depth_profile": "深度图像 profile；保持 RGB-D 对齐。",
    "pointcloud_enable": "是否启用驱动点云。",
    "use_rviz": "是否启动 MoveIt RViz。",
    "rviz_config": "RViz 配置文件。",
    "reference_path": "记录 id=1 目标图像角点的 YAML 文件。",
    "debug": "MoveIt 调试模式。",
    "allow_trajectory_execution": "是否允许 MoveIt 执行轨迹。",
    "publish_monitored_planning_scene": "是否发布监控规划场景。",
    "monitor_dynamics": "是否监控机器人动力学。",
    "capabilities": "额外 MoveIt capabilities。",
    "disable_capabilities": "禁用的 MoveIt capabilities。",
    "publish_frequency": "MoveIt 状态发布频率。",
    **NODE_PARAMETER_DESCRIPTIONS,
}


def _argument(name, default):
    kwargs = {"default_value": default, "description": DESCRIPTIONS[name]}
    return DeclareLaunchArgument(name, **kwargs)


def _node_parameter(context, name):
    raw = value(context, name)
    return raw if isinstance(_IMAGE_SERVO_DEFAULTS[name], str) else yaml.safe_load(raw)


def _hardware_moveit_config(kinematics_file: str):
    return (
        MoveItConfigsBuilder(
            "fairino_arm_moveit_descriptions", package_name="fairino_arm_moveit_config"
        )
        .robot_description_kinematics(file_path=f"config/{kinematics_file}")
        .planning_pipelines(default_planning_pipeline="fairino")
        .trajectory_execution(
            file_path="config/moveit_controllers_real.yaml", moveit_manage_controllers=False
        )
        .to_moveit_configs()
    )


def _launch_setup(context):
    use_sim_time = LaunchConfiguration("use_sim_time")
    image_params = {name: _node_parameter(context, name) for name in _IMAGE_SERVO_DEFAULTS}
    image_moveit_params = image_servo_moveit_parameters()
    moveit_config = _hardware_moveit_config(
        f"kinematics_{image_moveit_params['ik_moveit_servo']}.yaml"
    )
    servo_yaml = load_yaml("fairino_arm_moveit_config", "config/servo_parameters_real.yaml")
    servo_yaml.update({
        "move_group_name": "robot_arm",
        "planning_frame": "base_link",
        "ee_frame_name": "tool0",
        "command_out_topic": "/robot_arm_controller/joint_trajectory",
        "command_out_type": "trajectory_msgs/JointTrajectory",
        "cartesian_command_in_topic": "/servo_node/delta_twist_cmds",
        "joint_command_in_topic": "/servo_node/delta_joint_cmds",
        "monitored_planning_scene_topic": "/move_group_fairino/monitored_planning_scene",
        "use_gazebo": False,
    })

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
            "pointcloud.enable": value(context, "pointcloud_enable"),
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
            "use_rviz": value(context, "use_rviz"),
            "rviz_config": value(context, "rviz_config"),
            "execution_ik": image_moveit_params["ik_plugin"],
            "execution_pipeline": image_moveit_params["planning_pipeline_id"],
            **{
                name: value(context, name)
                for name in (
                    "debug", "allow_trajectory_execution",
                    "publish_monitored_planning_scene", "monitor_dynamics",
                    "capabilities", "disable_capabilities", "publish_frequency",
                )
            },
        }.items(),
    )
    handeye = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "calibration_name": "robot_calibration",
            "storage_directory": str(
                Path.home()
                / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real"
            ),
        }],
    )
    servo = TimerAction(
        period=4.0,
        actions=[
            Node(
                package="moveit_servo",
                executable="servo_node_main",
                name="servo_node",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {"use_sim_time": use_sim_time, "moveit_servo": servo_yaml},
                ],
            )
        ],
    )
    ibvs = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="visual_servo_bringup",
                executable="visual_image_servo",
                name="visual_image_servo",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}, image_params],
            )
        ],
    )
    return [camera, moveit, handeye, servo, ibvs]


def generate_launch_description():
    return LaunchDescription([
        *(_argument(name, default) for name, default in DEFAULTS.items()),
        OpaqueFunction(function=_launch_setup),
    ])
