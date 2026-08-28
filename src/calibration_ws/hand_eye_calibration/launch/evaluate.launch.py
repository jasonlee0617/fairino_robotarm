"""Evaluate a saved calibration with the fixed-profile ArUco pipeline."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

from myrobot_common.camera.launch import camera_launch
from handeye_launch_utils import default_storage_directory, load_handeye_profile  # noqa: E402


_DEFAULTS = {
    "calibration_type": "eye_in_hand", "camera_type": "realsense", "use_sim_time": "false",
    "use_rviz": "true", "storage_directory": default_storage_directory("real"), "calibration_name": "",
}


def _value(context, name):
    return LaunchConfiguration(name).perform(context).strip()


def _bool(context, name):
    return _value(context, name).lower() in {"1", "true", "yes", "on"}


def _setup(context, *_args, **_kwargs):
    profile = load_handeye_profile(_value(context, "calibration_type"))
    use_sim_time = _bool(context, "use_sim_time")
    calibration_name = _value(context, "calibration_name") or profile["calibration_name"]
    storage_directory = os.path.expanduser(os.path.expandvars(_value(context, "storage_directory")))
    handeye_share = get_package_share_directory("hand_eye_calibration")
    aruco_parameters = os.path.join(handeye_share, "config", "aruco_parameters.yaml")
    handeye_params = {
        "calibration_name": calibration_name, "camera_link_frame": profile["camera_link_frame"],
        "publish_child_frame": profile["publish_camera_link_frame"], "publish_rate_hz": 10.0,
        "use_tracking_to_camera_link_compensation": profile["use_tracking_to_camera_link_compensation"],
        "storage_directory": storage_directory, "use_sim_time": use_sim_time,
    }
    return [
        camera_launch(_value(context, "camera_type")),
        Node(package="ros2_aruco", executable="aruco_node", output="screen",
             parameters=[{"use_sim_time": use_sim_time}, aruco_parameters]),
        Node(package="hand_eye_calibration", executable="aruco_marker_pose_publisher.py",
             name="aruco_marker_pose_publisher", output="screen",
             parameters=[{"marker_id": profile["marker_id"], "aruco_topic": "/aruco_markers",
                          "output_topic": "/aruco_marker/pose", "use_sim_time": use_sim_time}]),
        Node(package="hand_eye_calibration", executable="calibration_aruco_publisher.py",
             name="calibration_aruco_publisher", output="screen",
             parameters=[{"tracking_base_frame": profile["tracking_base_frame"],
                          "tracking_marker_frame": profile["tracking_marker_frame"],
                          "marker_pose_topic": "/aruco_marker/pose", "use_sim_time": use_sim_time}]),
        Node(package="hand_eye_calibration", executable="handeye_publisher.py",
             name="handeye_publisher", output="screen", parameters=[handeye_params]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([PathJoinSubstitution([
            FindPackageShare("easy_handeye2"), "launch/evaluate.launch.py"
        ])]), launch_arguments={"name": calibration_name, "storage_directory": storage_directory}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("fairino_arm_moveit_config"), "launch", "demo.launch.py"
        )), launch_arguments={"include_gripper": "False", "use_rviz": _value(context, "use_rviz"),
                              "rviz_config": os.path.join(handeye_share, "rviz", "follow_aruco_move.rviz")}.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=default,
                                choices=("eye_in_hand", "eye_on_base") if name == "calibration_type" else None)
          for name, default in _DEFAULTS.items()],
        OpaqueFunction(function=_setup),
    ])
