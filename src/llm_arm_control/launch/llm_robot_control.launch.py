#!/usr/bin/env python3
"""Real-hardware LLM Robot entry point with YOLO perception."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
)
from llm_arm_control_nodes.voice_launch import build_voice_launch_actions

_HANDEYE_LAUNCH_DIR = os.path.join(
    get_package_share_directory("hand_eye_calibration"), "launch"
)
if _HANDEYE_LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _HANDEYE_LAUNCH_DIR)
from myrobot_common.camera.launch import camera_launch  # noqa: E402
from handeye_launch_utils import default_storage_directory, value  # noqa: E402


_TASK_PARAMETERS = load_node_parameters_yaml(
    "llm_arm_control", "config/llm_robot_control_params.yaml",
    "llm_control_task_server", "real",
)
_LLM_PERCEPTION_PARAMETERS = load_node_parameters_yaml(
    "llm_arm_control", "config/llm_robot_control_params.yaml", "llm_visual_perception", "real"
)
_PUBLIC_TASK_PARAMETER_NAMES = (
    "ik_plugin", "planning_pipeline_id", "planner_id", "move_group_ready_timeout_sec",
    "allow_cross_client_fallback", "arm_max_velocity", "arm_max_acceleration",
)
_PUBLIC_TASK_FALLBACKS = {name: _TASK_PARAMETERS[name] for name in _PUBLIC_TASK_PARAMETER_NAMES}
_LAUNCH_FALLBACKS = {
    "use_sim_time": "false",
    "camera_type": "realsense",
    "camera_serial_no": "",
    "color_profile": "1280x720x30",
    "depth_profile": "848x480x30",
    "pointcloud_enable": "false",
    "use_rviz": "true",
    "debug": "false",
    "allow_trajectory_execution": "true",
    "publish_monitored_planning_scene": "true",
    "monitor_dynamics": "false",
    "capabilities": "",
    "disable_capabilities": "",
    "publish_frequency": "100.0",
    "enable_voice": "true",
    "audio_input_device": "auto",
    "audio_input_volume_percent": "100",
    "rviz_config": os.path.join(
        get_package_share_directory("llm_arm_control"), "rviz", "llm_robot_control.rviz"
    ),
}
_YAML_LAUNCH_DEFAULTS = load_launch_parameters_yaml(
    "llm_arm_control", "config/llm_robot_control_params.yaml", "real"
)
DEFAULTS = {**_LAUNCH_FALLBACKS, **launch_defaults_as_strings(_PUBLIC_TASK_FALLBACKS)}
DEFAULTS.update(launch_defaults_as_strings({
    name: _YAML_LAUNCH_DEFAULTS[name]
    for name in DEFAULTS.keys() & _YAML_LAUNCH_DEFAULTS.keys()
}))


def _argument(name: str, default: str) -> DeclareLaunchArgument:
    """Create a launch argument with optional choices."""
    kwargs = {"default_value": default, "description": name}

    if name == "camera_type":
        kwargs["choices"] = ["realsense", "oak"]
    return DeclareLaunchArgument(name, **kwargs)


def _public_task_parameters(context):
    return {
        name: launch_parameter_value(value(context, name), fallback)
        for name, fallback in _PUBLIC_TASK_FALLBACKS.items()
    }


def _launch_setup(context):
    """Assemble the real-hardware launch description."""
    task_params = dict(_TASK_PARAMETERS)
    task_params.update(_public_task_parameters(context))
    use_sim_time = LaunchConfiguration("use_sim_time")

    # 相机驱动
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
        },
        oak_args={
            "enable_color": "true",
            "enable_depth": "true",
        },
    )

    # MoveIt 真实机械臂
    moveit_launch_args = {
        name: LaunchConfiguration(name)
        for name in (
            "use_rviz",
            "debug",
            "allow_trajectory_execution",
            "publish_monitored_planning_scene",
            "monitor_dynamics",
            "capabilities",
            "disable_capabilities",
            "publish_frequency",
            "rviz_config",
        )
    }
    moveit_launch_args["execution_ik"] = task_params["ik_plugin"]
    moveit_launch_args["execution_pipeline"] = task_params["planning_pipeline_id"]
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("fairino_arm_moveit_config"),
                "launch",
                "moveit_hardware.launch.py",
            )
        ),
        launch_arguments=moveit_launch_args.items(),
    )

    # 手眼标定发布器
    handeye = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "calibration_name": "robot_calibration",
            "storage_directory": default_storage_directory("real"),
        }],
    )

    # 轨迹时间重规划服务
    retime = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )

    # YOLO 感知
    yolo_obb = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("visual_perception"),
                "launch",
                "llm_visual_perception.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "require_cuda_for_visual_perception": launch_defaults_as_strings(
                _LLM_PERCEPTION_PARAMETERS
            )["require_cuda_for_visual_perception"],
            "sync_slop": launch_defaults_as_strings(
                _LLM_PERCEPTION_PARAMETERS
            )["sync_slop"],
            "sync_watchdog_sec": launch_defaults_as_strings(
                _LLM_PERCEPTION_PARAMETERS
            )["sync_watchdog_sec"],
            "expected_camera_rate_hz": launch_defaults_as_strings(
                _LLM_PERCEPTION_PARAMETERS
            )["expected_camera_rate_hz"],
        }.items(),
    )

    # LLM 任务服务器
    task = Node(
        package="llm_arm_control",
        executable="llm_control_task_server",
        output="screen",
        parameters=[
            task_params,
            {
                "use_sim_time": use_sim_time,
            },
        ],
    )

    # 保留本地 stop/reset/resume 安全控制；自然语言输入只由语音链提供。
    motion_control = Node(
        package="myrobot_common",
        executable="motion_control",
        name="motion_control",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "command_burst_count": int(_YAML_LAUNCH_DEFAULTS["command_burst_count"]),
            "enable_grasp_trigger_key": False,
        }],
    )

    voice_actions = build_voice_launch_actions(
        context,
        environment="real",
        enable_voice=LaunchConfiguration("enable_voice"),
        audio_input_device=LaunchConfiguration("audio_input_device"),
        audio_input_volume_percent=LaunchConfiguration(
            "audio_input_volume_percent"
        ),
        use_sim_time=use_sim_time,
    )

    return [
        camera,
        moveit,
        handeye,
        retime,
        yolo_obb,
        task,
        motion_control,
        *voice_actions,
    ]


def generate_launch_description():
    """Return the real-hardware LLM robot launch description."""
    launch_arguments = [
        _argument(name, default) for name, default in DEFAULTS.items()
    ]
    return LaunchDescription([
        *launch_arguments,
        OpaqueFunction(function=_launch_setup),
    ])
