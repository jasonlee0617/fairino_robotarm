"""Launch the automatic collector with its safe motion overrides."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

from hand_eye_calibration.config import flatten_ros_parameters
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
)


PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}
_PUBLIC_PARAMETERS = (
    "calibration_type", "ik_plugin", "planning_pipeline_id", "planner_id",
    "max_velocity", "max_acceleration", "allowed_planning_time", "max_step_size",
    "position_tolerance", "orientation_tolerance", "allowed_start_tolerance",
    "moveit_ready_timeout", "moveit_ready_poll_interval",
)
_IK_NAMESPACES = {"fairino": "/move_group_fairino", "kdl": "/move_group_kdl"}


def _yaml_parameters():
    path = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "auto_calibration_collector_params.yaml",
    )
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    parameters = document.get("auto_calibration_collector", {}).get("ros__parameters", {})
    return path, flatten_ros_parameters(parameters)


_PARAMETERS_FILE, _YAML_PARAMETERS = _yaml_parameters()
_FALLBACKS = {"ik_plugin": "fairino", **_YAML_PARAMETERS}
_LAUNCH_DEFAULTS = launch_defaults_as_strings({
    name: _FALLBACKS[name] for name in _PUBLIC_PARAMETERS
})
_CONFIGURATIONS = {name: LaunchConfiguration(name) for name in _PUBLIC_PARAMETERS}


def _overrides(context, *_args, **_kwargs):
    ik_plugin = _CONFIGURATIONS["ik_plugin"].perform(context).strip()
    if ik_plugin not in _IK_NAMESPACES:
        raise RuntimeError("ik_plugin must be fairino or kdl")
    overrides = {
        name: launch_parameter_value(
            _CONFIGURATIONS[name].perform(context).strip(), _FALLBACKS[name]
        )
        for name in _PUBLIC_PARAMETERS
        if name != "ik_plugin"
    }
    overrides["move_group_ns_fairino"] = _IK_NAMESPACES[ik_plugin]
    return [Node(
        package="hand_eye_calibration",
        executable="auto_calibration_collector.py",
        name="auto_calibration_collector",
        output="screen",
        emulate_tty=True,
        additional_env=PYTHON_NO_USER_SITE_ENV,
        parameters=[_PARAMETERS_FILE, overrides],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "calibration_type",
            default_value=_LAUNCH_DEFAULTS["calibration_type"],
            choices=("eye_in_hand", "eye_on_base"),
        ),
        DeclareLaunchArgument(
            "ik_plugin", default_value=_LAUNCH_DEFAULTS["ik_plugin"], choices=tuple(_IK_NAMESPACES)
        ),
        *[
            DeclareLaunchArgument(name, default_value=_LAUNCH_DEFAULTS[name])
            for name in _PUBLIC_PARAMETERS
            if name not in {"calibration_type", "ik_plugin"}
        ],
        OpaqueFunction(function=_overrides),
    ])
