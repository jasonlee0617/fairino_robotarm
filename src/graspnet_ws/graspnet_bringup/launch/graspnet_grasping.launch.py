#!/usr/bin/env python3
"""实机 RGB-D GraspNet 视觉抓取入口。"""

import os
import shlex
import sys
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
    write_node_parameters_ros_file,
)

_HANDEYE_LAUNCH_DIR = os.path.join(
    get_package_share_directory("hand_eye_calibration"), "launch"
)
if _HANDEYE_LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _HANDEYE_LAUNCH_DIR)

from myrobot_common.camera.launch import camera_launch
from handeye_launch_utils import value  # noqa: E402


_TASK_PARAMETERS = load_node_parameters_yaml(
    "graspnet_bringup", "config/graspnet_grasping_params.yaml", "graspnet_visual_grasping", "real"
)
_PUBLIC_TASK_PARAMETER_NAMES = (
    "ik_plugin", "planning_pipeline_id", "planner_id", "move_group_ready_timeout_sec",
    "allow_cross_client_fallback", "arm_max_velocity", "arm_max_acceleration",
    "allowed_planning_time", "position_tolerance", "orientation_tolerance",
    "allowed_start_tolerance",
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
    "rviz_config": os.path.join(
        get_package_share_directory("graspnet_bringup"), "rviz", "graspnet_grasping.rviz"
    ),
}
_YAML_LAUNCH_DEFAULTS = load_launch_parameters_yaml(
    "graspnet_bringup", "config/graspnet_grasping_params.yaml", "real"
)
DEFAULTS = {**_LAUNCH_FALLBACKS, **launch_defaults_as_strings(_PUBLIC_TASK_FALLBACKS)}
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


def _graspnet_inference_process(context):
    model_profile = _YAML_LAUNCH_DEFAULTS["model_profile"]
    install_setup = str(Path(get_package_prefix("graspnet_bringup")).parent / "setup.bash")
    config_path = write_node_parameters_ros_file(
        "graspnet_bringup", "config/graspnet_grasping_params.yaml", "graspnet_inference", "real"
    )
    source_share = get_package_share_directory("graspnet_source")
    baseline_dir = os.path.join(source_share, "graspnet_baseline")
    checkpoint_path = os.path.join(source_share, "models", f"checkpoint-{model_profile}.tar")
    conda_setup = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    command_prefix = (
        "set -e; "
        f"source {shlex.quote(conda_setup)}; "
        "conda activate graspnet; "
        "source /opt/ros/humble/setup.bash; "
        f"source {shlex.quote(install_setup)}; "
        "export PYTHONUNBUFFERED=1; "
        "export MPLCONFIGDIR=/tmp/graspnet_mpl_config; "
        "export XDG_CACHE_HOME=/tmp/graspnet_xdg_cache; "
        "mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME; "
        "exec python -m graspnet_bringup.graspnet_inference_node "
        "--ros-args "
        f"--params-file {shlex.quote(config_path)} "
        "-r __node:=graspnet_inference "
        "-p use_sim_time:="
    )
    command_suffix = (
        " "
        f"-p baseline_dir:={shlex.quote(baseline_dir)} "
        f"-p checkpoint_path:={shlex.quote(checkpoint_path)}"
    )
    return [
        ExecuteProcess(
            cmd=["bash", "-lc", command_prefix + value(context, "use_sim_time") + command_suffix],
            output="screen",
        )
    ]


def _launch_setup(context):
    package_share = get_package_share_directory("graspnet_bringup")
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
            "hole_filling_filter.enable": "false",
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
    grasp = Node(
        package="graspnet_bringup", executable="graspnet_visual_grasping",
        name="graspnet_visual_grasping", output="screen", parameters=[
            task_params,
            {"use_sim_time": use_sim_time},
        ],
    )
    motion_control = Node(
        package="myrobot_common", executable="motion_control",
        name="motion_control", output="screen",
    )
    return [
        camera,
        moveit,
        handeye,
        retime,
        TimerAction(period=3.0, actions=[OpaqueFunction(function=_graspnet_inference_process)]),
        TimerAction(period=8.0, actions=[grasp, motion_control]),
    ]


def generate_launch_description():
    return LaunchDescription([
        *(_argument(name, default) for name, default in DEFAULTS.items()),
        OpaqueFunction(function=_launch_setup),
    ])
