"""One Gazebo hand-eye calibration environment for both camera topologies."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_yaml,
)


PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}
_CALIBRATION_PROFILES = (
    "fairino_arm_gripper_inhand",
    "fairino_arm_gripper_calibration_onbase",
)
_PUBLIC_ARGUMENTS = ("robot_profile", "use_sim_time", "use_rviz", "rviz_config")
_LAUNCH_FALLBACKS = {
    "robot_profile": "fairino_arm_gripper_inhand",
    "use_sim_time": True,
    "use_rviz": True,
    "rviz_config": "rviz/calibration_sim.rviz",
}
_CONFIG = load_yaml("myrobot_simulation", "config/calibration_sim_params.yaml")
_YAML_LAUNCH = _CONFIG.get("launch", {})
if not isinstance(_YAML_LAUNCH, dict):
    raise RuntimeError("calibration_sim_params.yaml launch must be a mapping")
_LAUNCH_DEFAULTS = launch_defaults_as_strings({**_LAUNCH_FALLBACKS, **_YAML_LAUNCH})
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _PUBLIC_ARGUMENTS
}
_GAZEBO_SHARED_ARGUMENTS = (
    "world", "camera_profile", "camera_profile_file", "camera_noise_mode",
    "camera_depth_far_m", "camera_fps", "camera_image_width", "camera_image_height",
    "enable_camera_model", "enable_camera_bridge", "enable_servo", "publish_frequency",
    "initial_positions_file", "robot_spawn_delay", "controller_spawn_delay", "spawn_name",
    "spawn_x", "spawn_y", "spawn_z", "spawn_roll", "spawn_pitch", "spawn_yaw",
)


def _value(context, name):
    return _LAUNCH_CONFIGURATIONS[name].perform(context).strip()


def _public_value(context, name):
    return launch_parameter_value(_value(context, name), _YAML_LAUNCH.get(name, _LAUNCH_FALLBACKS[name]))


def _selected_profile(profile_name):
    profiles = _CONFIG.get("profiles", {})
    if profile_name not in _CALIBRATION_PROFILES or not isinstance(profiles, dict):
        raise RuntimeError(
            "robot_profile must be fairino_arm_gripper_inhand or "
            "fairino_arm_gripper_calibration_onbase"
        )
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict) or not isinstance(profile.get("board"), dict):
        raise RuntimeError(f"calibration profile '{profile_name}' is incomplete")
    board = profile["board"]
    source = board.get("source")
    required = (
        ("model", "entity_name", "spawn_delay_sec", "x", "y", "z", "roll", "pitch", "yaw")
        if source == "world"
        else (
            "calibration_board_x", "calibration_board_y", "calibration_board_z",
            "calibration_board_roll", "calibration_board_pitch", "calibration_board_yaw",
        )
        if source == "flange"
        else ()
    )
    if not required or any(name not in board for name in required):
        raise RuntimeError(f"calibration profile '{profile_name}' has an invalid board topology")
    return profile, board


def _setup(context, *_args, **_kwargs):
    gz_share = get_package_share_directory("myrobot_simulation")
    handeye_share = get_package_share_directory("hand_eye_calibration")
    profile_name = _value(context, "robot_profile")
    topology, board = _selected_profile(profile_name)
    calibration_type = topology.get("calibration_type")
    if calibration_type not in {"eye_in_hand", "eye_on_base"}:
        raise RuntimeError(f"calibration profile '{profile_name}' has an invalid calibration_type")
    handeye_launch_dir = os.path.join(handeye_share, "launch")
    if handeye_launch_dir not in sys.path:
        sys.path.insert(0, handeye_launch_dir)
    from handeye_launch_utils import load_handeye_profile  # noqa: PLC0415

    handeye_profile = load_handeye_profile(calibration_type)
    use_sim_time = _public_value(context, "use_sim_time")
    rviz_config = _value(context, "rviz_config")
    if not os.path.isabs(rviz_config):
        rviz_config = os.path.join(gz_share, rviz_config)
    gazebo_arguments = {
        **launch_defaults_as_strings({
            name: _YAML_LAUNCH[name] for name in _GAZEBO_SHARED_ARGUMENTS
        }),
        "robot_profile": profile_name,
        "use_sim_time": str(use_sim_time).lower(),
        "enable_rviz": str(_public_value(context, "use_rviz")).lower(),
        "rviz_config": rviz_config,
    }
    if board["source"] == "flange":
        gazebo_arguments.update({
            name: str(board[name])
            for name in (
                "calibration_board_x", "calibration_board_y", "calibration_board_z",
                "calibration_board_roll", "calibration_board_pitch", "calibration_board_yaw",
            )
        })
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments=gazebo_arguments.items(),
    )
    actions = [simulation]
    if board["source"] == "world":
        actions.append(TimerAction(
            period=float(board["spawn_delay_sec"]),
            actions=[Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-file", os.path.join(gz_share, board["model"]),
                    "-name", str(board["entity_name"]),
                    "-x", str(board["x"]), "-y", str(board["y"]), "-z", str(board["z"]),
                    "-R", str(board["roll"]), "-P", str(board["pitch"]), "-Y", str(board["yaw"]),
                    "-allow_renaming", "false",
                ],
            )],
        ))
    aruco_parameters = os.path.join(handeye_share, "config", "aruco_parameters.yaml")
    actions.extend([
        Node(
            package="ros2_aruco", executable="aruco_node", output="screen",
            additional_env=PYTHON_NO_USER_SITE_ENV,
            parameters=[{"use_sim_time": use_sim_time}, aruco_parameters],
        ),
        Node(
            package="hand_eye_calibration", executable="aruco_marker_pose_publisher.py",
            name="aruco_marker_pose_publisher", output="screen", additional_env=PYTHON_NO_USER_SITE_ENV,
            parameters=[{
                "marker_id": handeye_profile["marker_id"],
                "aruco_topic": "/aruco_markers",
                "output_topic": "/aruco_marker/pose",
                "use_sim_time": use_sim_time,
            }],
        ),
        Node(
            package="hand_eye_calibration", executable="calibration_aruco_publisher.py",
            name="calibration_aruco_publisher", output="screen", additional_env=PYTHON_NO_USER_SITE_ENV,
            parameters=[{
                "tracking_base_frame": handeye_profile["tracking_base_frame"],
                "tracking_marker_frame": handeye_profile["tracking_marker_frame"],
                "marker_pose_topic": "/aruco_marker/pose",
                "use_sim_time": use_sim_time,
            }],
        ),
        Node(
            package="hand_eye_calibration", executable="visualize_aruco_marker.py",
            name="aruco_pose_estimator", output="screen", additional_env=PYTHON_NO_USER_SITE_ENV,
            parameters=[{
                "image_topic": "/camera/camera/color/image_raw",
                "camera_info_topic": "/camera/camera/color/camera_info",
                "output_topic": "/aruco_image",
                "marker_size": 0.07,
                "aruco_dictionary_id": "DICT_5X5_250",
                "use_sim_time": use_sim_time,
            }],
        ),
    ])
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_profile",
            default_value=_LAUNCH_DEFAULTS["robot_profile"],
            choices=_CALIBRATION_PROFILES,
            description="选择眼在手上或眼在手外标定仿真拓扑。",
        ),
        DeclareLaunchArgument("use_sim_time", default_value=_LAUNCH_DEFAULTS["use_sim_time"]),
        DeclareLaunchArgument("use_rviz", default_value=_LAUNCH_DEFAULTS["use_rviz"]),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(
            get_package_share_directory("myrobot_simulation"), _LAUNCH_DEFAULTS["rviz_config"]
        )),
        OpaqueFunction(function=_setup),
    ])
