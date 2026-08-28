#!/usr/bin/env python3
"""实机 RGB-D YOLO-OBB 抓取与可选语义 OctoMap 入口。"""

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    load_moveit_parameters_yaml,
    load_node_parameters_yaml,
)

_HANDEYE_LAUNCH_DIR = os.path.join(
    get_package_share_directory("hand_eye_calibration"), "launch"
)
if _HANDEYE_LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _HANDEYE_LAUNCH_DIR)

from myrobot_common.camera.launch import camera_launch
from handeye_launch_utils import value  # noqa: E402


DEFAULTS = {
    "use_sim_time": "false",
    "camera_type": "realsense",
    "camera_serial_no": "",
    "color_profile": "1280x720x30",
    "depth_profile": "848x480x30",
    "pointcloud_enable": "true",
    "use_rviz": "true",
    "rviz_config": os.path.join(
        get_package_share_directory("visual_grasping_bringup"), "rviz", "visual_octmap.rviz"
    ),
    "debug": "false",
    "allow_trajectory_execution": "true",
    "publish_monitored_planning_scene": "true",
    "monitor_dynamics": "false",
    "capabilities": "",
    "disable_capabilities": "",
    "publish_frequency": "100.0",
    "device": "auto",
    "conf": "0.5",
    "imgsz": "1024",
    "enable_semantic_cloud_filter": "false",
    "enable_dynamic_collision_objects": "false",
}

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
    "device": "YOLO 推理设备，例如 auto、cpu、0。",
    "conf": "YOLO OBB 置信度阈值。",
    "imgsz": "YOLO OBB 推理尺寸。",
    "enable_semantic_cloud_filter": "是否启动语义点云过滤节点。",
    "enable_dynamic_collision_objects": "是否启动动态碰撞体节点。",
}


def _argument(name, default):
    kwargs = {"default_value": default, "description": DESCRIPTIONS[name]}
    if name == "camera_type":
        kwargs["choices"] = ["realsense", "oak"]
    return DeclareLaunchArgument(name, **kwargs)


def _launch_setup(context):
    package_share = get_package_share_directory("visual_grasping_bringup")
    task_moveit_params = load_moveit_parameters_yaml(
        "visual_grasping_bringup", "config/visual_grasping_params.yaml", "visual_grasping", "real"
    )
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
        launch_arguments=(
            {
                name: LaunchConfiguration(name)
                for name in (
                    "use_rviz", "rviz_config", "debug",
                    "allow_trajectory_execution", "publish_monitored_planning_scene",
                    "monitor_dynamics", "capabilities", "disable_capabilities",
                    "publish_frequency",
                )
            }
            | {
                "execution_ik": task_moveit_params["ik_plugin"],
                "execution_pipeline": task_moveit_params["planning_pipeline_id"],
            }
        ).items(),
    )
    detector = TimerAction(period=3.0, actions=[Node(
        package="visual_perception", executable="yolo_detector_obb.py",
        name="yolo_detector_obb", parameters=[{
            "model_path": os.path.join(get_package_share_directory("visual_perception"), "models", "yolo-obb-1280.pt"),
            "use_sim_time": use_sim_time,
            "device": LaunchConfiguration("device"),
            "conf": LaunchConfiguration("conf"),
            "imgsz": LaunchConfiguration("imgsz"),
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            "enable_visualization": True,
            "enable_ema_smoothing": True,
            "ema_alpha": 0.35,
            "sync_slop": 0.03,
        }],
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
            load_node_parameters_yaml(
                "visual_grasping_bringup", "config/visual_grasping_params.yaml", "visual_grasping", "real"
            ),
            {"use_sim_time": use_sim_time, **task_moveit_params, "allow_cross_client_fallback": False},
        ],
    )])
    octomap_nodes = TimerAction(period=5.0, actions=[
        Node(
            package="visual_perception", executable="semantic_octomap_cloud_filter.py",
            name="semantic_octomap_cloud_filter_node", output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(LaunchConfiguration("enable_semantic_cloud_filter")),
        ),
        Node(
            package="visual_grasping_bringup", executable="dynamic_collision_objects",
            name="dynamic_collision_objects_node", output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(LaunchConfiguration("enable_dynamic_collision_objects")),
        ),
    ])
    return [camera, moveit, detector, handeye, retime, grasp, octomap_nodes]


def generate_launch_description():
    return LaunchDescription([
        *(_argument(name, default) for name, default in DEFAULTS.items()),
        OpaqueFunction(function=_launch_setup),
    ])
