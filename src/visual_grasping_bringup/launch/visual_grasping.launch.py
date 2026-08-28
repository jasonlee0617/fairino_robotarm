#!/usr/bin/env python3
"""实机 RGB-D YOLO-OBB 视觉抓取入口。"""

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
)

_HANDEYE_LAUNCH_DIR = os.path.join(
    get_package_share_directory("hand_eye_calibration"), "launch"
)
if _HANDEYE_LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _HANDEYE_LAUNCH_DIR)

from myrobot_common.camera.launch import camera_launch
from handeye_launch_utils import value  # noqa: E402


_TASK_PARAMETERS = load_node_parameters_yaml(
    "visual_grasping_bringup", "config/visual_grasping_params.yaml", "visual_grasping", "real"
)
_PUBLIC_TASK_PARAMETER_NAMES = (
    "ik_plugin", "planning_pipeline_id", "planner_id", "move_group_ready_timeout_sec",
    "allow_cross_client_fallback", "arm_max_velocity", "arm_max_acceleration",
    "allowed_planning_time", "position_tolerance", "orientation_tolerance",
    "allowed_start_tolerance", "max_step_size",
)
_PUBLIC_TASK_FALLBACKS = {
    name: _TASK_PARAMETERS[name] for name in _PUBLIC_TASK_PARAMETER_NAMES
}
_ENVIRONMENT_FALLBACKS = {
    "use_sim_time": "false", "camera_type": "realsense", "camera_serial_no": "",
    "color_profile": "1280x720x30", "depth_profile": "848x480x30",
    "pointcloud_enable": "false", "use_rviz": "true", "debug": "false",
    "allow_trajectory_execution": "true", "publish_monitored_planning_scene": "true",
    "monitor_dynamics": "false", "capabilities": "", "disable_capabilities": "",
    "publish_frequency": "100.0",
}
_YAML_LAUNCH_DEFAULTS = load_launch_parameters_yaml(
    "visual_grasping_bringup", "config/visual_grasping_params.yaml", "real"
)
DEFAULTS = {
    **_ENVIRONMENT_FALLBACKS,
    "rviz_config": os.path.join(
        get_package_share_directory("visual_grasping_bringup"), "rviz", "visual_grasping.rviz"
    ),
    **launch_defaults_as_strings(_PUBLIC_TASK_FALLBACKS),
}
DEFAULTS.update(launch_defaults_as_strings({
    name: _YAML_LAUNCH_DEFAULTS[name]
    for name in DEFAULTS.keys() & _YAML_LAUNCH_DEFAULTS.keys()
}))

DESCRIPTIONS = {
    "use_sim_time": "实机为 false；仿真入口单独设置为 true。",
    "camera_type": "相机类型：realsense 或 oak。",
    "camera_serial_no": "RealSense 相机序列号；留空时由驱动选择。",
    "color_profile": "彩色 profile，例如 1280x720x30。",
    "depth_profile": "深度 profile，例如 848x480x30。",
    "pointcloud_enable": "是否启用驱动点云。",
    "use_rviz": "是否启动 MoveIt RViz。",
    "rviz_config": "RViz 配置文件绝对路径。",
    "debug": "MoveIt 调试模式。",
    "allow_trajectory_execution": "是否允许 MoveIt 执行轨迹。",
    "publish_monitored_planning_scene": "是否发布监控规划场景。",
    "monitor_dynamics": "是否监控机器人动力学。",
    "capabilities": "额外 MoveIt capabilities。",
    "disable_capabilities": "禁用的 MoveIt capabilities。",
    "publish_frequency": "MoveIt 状态发布频率。",
    "ik_plugin": "离散规划与执行使用的 IK client。",
    "planning_pipeline_id": "离散规划管线。",
    "planner_id": "离散规划器 ID。",
    "move_group_ready_timeout_sec": "MoveIt client 就绪等待秒数。",
    "allow_cross_client_fallback": "是否允许任务跨 MoveIt client 回退。",
    "arm_max_velocity": "离散规划最大关节速度比例。",
    "arm_max_acceleration": "离散规划最大关节加速度比例。",
    "allowed_planning_time": "离散规划最长耗时，单位秒。",
    "position_tolerance": "离散规划位置容差，单位米。",
    "orientation_tolerance": "离散规划姿态容差，单位弧度。",
    "allowed_start_tolerance": "离散规划起点容差。",
    "max_step_size": "笛卡尔路径离散步长，单位米。",
}


def _argument(name, default):
    kwargs = {"default_value": default, "description": DESCRIPTIONS[name]}
    if name == "camera_type":
        kwargs["choices"] = ["realsense", "oak"]
    return DeclareLaunchArgument(name, **kwargs)


def _public_task_parameters(context):
    return {
        name: launch_parameter_value(value(context, name), fallback)
        for name, fallback in _PUBLIC_TASK_FALLBACKS.items()
    }


def _launch_setup(context):
    task_params = dict(_TASK_PARAMETERS)
    task_params.update(_public_task_parameters(context))
    use_sim_time = LaunchConfiguration("use_sim_time")
    camera = camera_launch(
        value(context, "camera_type"),
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
        oak_args={
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.color_profile": value(context, "color_profile"),
            "depth_module.depth_profile": value(context, "depth_profile"),
            "pointcloud.enable": value(context, "pointcloud_enable"),
        },
    )
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("fairino_arm_moveit_config"),
            "launch", "moveit_hardware.launch.py",
        )),
        launch_arguments={
            **{
                name: LaunchConfiguration(name)
                for name in (
                    "use_rviz", "rviz_config", "debug",
                    "allow_trajectory_execution", "publish_monitored_planning_scene",
                    "monitor_dynamics", "capabilities", "disable_capabilities",
                    "publish_frequency",
                )
            },
            "execution_ik": task_params["ik_plugin"],
            "execution_pipeline": task_params["planning_pipeline_id"],
        }.items(),
    )
    detector = TimerAction(period=3.0, actions=[Node(
        package="visual_perception", executable="yolo_detector_obb.py",
        name="yolo_detector_obb", parameters=[
            load_node_parameters_yaml(
                "visual_grasping_bringup", "config/visual_grasping_params.yaml", "yolo_detector_obb", "real"
            ),
            {
                "model_path": os.path.join(get_package_share_directory("visual_perception"), "models", "yolo-obb-1280.pt"),
                "use_sim_time": use_sim_time,
            },
        ],
    )])
    handeye = Node(
        package="hand_eye_calibration", executable="handeye_publisher.py",
        name="handeye_publisher", output="screen", parameters=[{
            "use_sim_time": use_sim_time,
            "calibration_name": "robot_calibration",
            "storage_directory": str(Path.home() / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real"),
        }],
    )
    retime = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory("trajectory_retime_server"), "launch", "retime_server.launch.py"
    )))
    grasp = TimerAction(period=8.0, actions=[Node(
        package="visual_grasping_bringup", executable="visual_grasping", name="visual_grasping",
        output="screen", parameters=[
            task_params,
            {"use_sim_time": use_sim_time},
        ],
    )])
    motion_control = Node(
        package="myrobot_common", executable="motion_control", name="motion_control", output="screen"
    )
    return [camera, moveit, detector, handeye, retime, grasp, motion_control]


def generate_launch_description():
    return LaunchDescription([
        *(_argument(name, default) for name, default in DEFAULTS.items()),
        OpaqueFunction(function=_launch_setup),
    ])
