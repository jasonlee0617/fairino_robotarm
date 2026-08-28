"""Real calibration environment and the fixed-profile ArUco pipeline."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

from myrobot_common.camera.launch import camera_launch
from handeye_launch_utils import load_handeye_profile  # noqa: E402


PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}
_DEFAULTS = {
    "calibration_type": "eye_in_hand", "camera_type": "realsense",
    "camera_serial_no": "", "color_profile": "1280x720x30",
    "depth_profile": "848x480x30", "use_sim_time": "false", "use_rviz": "true",
    "rviz_config": "", "debug": "false", "allow_trajectory_execution": "true",
    "publish_monitored_planning_scene": "true", "monitor_dynamics": "false",
    "capabilities": "", "disable_capabilities": "", "ik_plugin": "fairino",
    "planning_pipeline_id": "fairino",
}
_PUBLIC_ARGUMENTS = tuple(_DEFAULTS)
_CHOICES = {
    "calibration_type": ("eye_in_hand", "eye_on_base"),
    "camera_type": ("realsense", "oak"),
    "ik_plugin": ("fairino", "kdl"),
    "planning_pipeline_id": ("fairino", "ompl"),
}


def _value(context, name):
    return LaunchConfiguration(name).perform(context).strip()


def _bool(context, name):
    return _value(context, name).lower() in {"1", "true", "yes", "on"}


def _setup(context, *_args, **_kwargs):
    profile = load_handeye_profile(_value(context, "calibration_type"))
    handeye_share = get_package_share_directory("hand_eye_calibration")
    use_sim_time = _bool(context, "use_sim_time")
    rviz_config = _value(context, "rviz_config") or os.path.join(
        handeye_share, "rviz", profile["rviz_config"]
    )
    aruco_parameters = os.path.join(handeye_share, "config", "aruco_parameters.yaml")
    camera = camera_launch(
        _value(context, "camera_type"),
        realsense_args={
            "serial_no": _value(context, "camera_serial_no"), "enable_color": "true",
            "enable_depth": "true", "rgb_camera.color_profile": _value(context, "color_profile"),
            "depth_module.depth_profile": _value(context, "depth_profile"), "align_depth.enable": "true",
        },
    )
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("fairino_arm_moveit_config"), "launch", "moveit_hardware.launch.py"
        )),
        launch_arguments={
            **{name: _value(context, name) for name in (
                "use_rviz", "debug", "allow_trajectory_execution", "publish_monitored_planning_scene",
                "monitor_dynamics", "capabilities", "disable_capabilities",
            )},
            "rviz_config": rviz_config, "execution_ik": _value(context, "ik_plugin"),
            "execution_pipeline": _value(context, "planning_pipeline_id"),
        }.items(),
    )
    return [
        camera, moveit,
        Node(package="ros2_aruco", executable="aruco_node", output="screen",
             additional_env=PYTHON_NO_USER_SITE_ENV,
             parameters=[{"use_sim_time": use_sim_time}, aruco_parameters]),
        Node(package="hand_eye_calibration", executable="aruco_marker_pose_publisher.py",
             name="aruco_marker_pose_publisher", output="screen", additional_env=PYTHON_NO_USER_SITE_ENV,
             parameters=[{"marker_id": profile["marker_id"], "aruco_topic": "/aruco_markers",
                          "output_topic": "/aruco_marker/pose", "use_sim_time": use_sim_time}]),
        Node(package="hand_eye_calibration", executable="calibration_aruco_publisher.py",
             name="calibration_aruco_publisher", output="screen", additional_env=PYTHON_NO_USER_SITE_ENV,
             parameters=[{"tracking_base_frame": profile["tracking_base_frame"],
                          "tracking_marker_frame": profile["tracking_marker_frame"],
                          "marker_pose_topic": "/aruco_marker/pose", "use_sim_time": use_sim_time}]),
        Node(package="hand_eye_calibration", executable="visualize_aruco_marker.py",
             name="aruco_pose_estimator", output="screen", additional_env=PYTHON_NO_USER_SITE_ENV,
             parameters=[{"image_topic": "/camera/camera/color/image_raw",
                          "camera_info_topic": "/camera/camera/color/camera_info", "output_topic": "/aruco_image",
                          "marker_size": 0.07, "aruco_dictionary_id": "DICT_5X5_250",
                          "use_sim_time": use_sim_time}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=_DEFAULTS[name], choices=_CHOICES.get(name))
          for name in _PUBLIC_ARGUMENTS],
        OpaqueFunction(function=_setup),
    ])
