"""Gazebo planning/IK demo with a narrow public launch contract."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, LogInfo,
    OpaqueFunction, RegisterEventHandler, TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
)


_PARAMS_FILE = "config/motion_planning_demo_params.yaml"
_PUBLIC_ARGUMENTS = (
    "robot_profile", "enable_rviz", "world", "use_sim_time", "initial_positions_file",
    "enable_camera_model", "rviz_config", "ik_plugin", "planning_pipeline_id",
    "planner_id", "move_group_ready_timeout_sec",
    "run_mode", "benchmark_output_dir",
)
_RAW_DEFAULTS = load_launch_parameters_yaml("myrobot_simulation", _PARAMS_FILE, None)
_DEFAULTS = launch_defaults_as_strings(_RAW_DEFAULTS)
_CONFIG = {name: LaunchConfiguration(name) for name in _PUBLIC_ARGUMENTS}


def _value(context, name):
    return _CONFIG[name].perform(context).strip()


def _node_parameters(context):
    params = load_node_parameters_yaml(
        "myrobot_simulation", _PARAMS_FILE, "motion_planning_node_sim", None
    )
    run_mode = _value(context, "run_mode")
    if run_mode not in ("interactive", "benchmark_execution", "benchmark_algorithm"):
        raise RuntimeError(
            "run_mode must be interactive, benchmark_execution, or benchmark_algorithm"
        )
    output_dir = _value(context, "benchmark_output_dir")
    if run_mode != "interactive" and not output_dir:
        raise RuntimeError("benchmark_output_dir is required for benchmark run_mode")
    return {
        **params,
        "planning_client": _value(context, "ik_plugin"),
        "default_pipeline_id": _value(context, "planning_pipeline_id"),
        "default_planner_id": _value(context, "planner_id"),
        "ik_timeout": launch_parameter_value(
            _value(context, "move_group_ready_timeout_sec"),
            _RAW_DEFAULTS["move_group_ready_timeout_sec"],
        ),
        "use_sim_time": _value(context, "use_sim_time").lower() == "true",
        "sim_world": _value(context, "world"),
        "run_mode": run_mode,
        "benchmark_output_dir": output_dir,
    }


def _setup(context, *_args, **_kwargs):
    gz_share = get_package_share_directory("myrobot_simulation")
    node_params = _node_parameters(context)
    scene_paths = {
        "scene_assets_dir": os.path.join(gz_share, "config", "scenes"),
        "scene_config_file": os.path.join(gz_share, "config", "scenes", "pathplanning_scenes_params.yaml"),
    }
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            **{name: _value(context, name) for name in (
                "robot_profile", "enable_rviz", "world", "use_sim_time",
                "initial_positions_file", "enable_camera_model", "rviz_config",
            )},
            **scene_paths,
            "scene_name": node_params["scene_name"],
            "spawn_sim_scene_models": str(node_params["spawn_sim_scene_models"]).lower(),
            "publish_planning_scene": str(node_params["publish_planning_scene"]).lower(),
            "publish_obstacle_markers": str(node_params["publish_obstacle_markers"]).lower(),
            "obstacle_marker_topic": node_params["obstacle_marker_topic"],
            "planner_random_seed": str(node_params["planner_random_seed"]),
            "moveit_clients": ",".join(
                ("fairino", "kdl") if node_params["run_mode"] == "interactive"
                else (node_params["planning_client"],)
            ),
        }.items(),
    )
    node = Node(
        package="myrobot_simulation", executable="motion_planning_node_sim.py",
        name="motion_planning_node_sim", output="screen", emulate_tty=True,
        parameters=[{**node_params, **scene_paths}],
    )
    actions = [
        simulation,
        TimerAction(period=5.0, actions=[
            LogInfo(msg=["[motion_planning_demo] mode=", node_params["run_mode"]]), node,
        ]),
    ]
    if node_params["run_mode"] != "interactive":
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[
                LogInfo(msg="[motion_planning_demo] benchmark node exited; shutting down launch."),
                EmitEvent(event=Shutdown(reason="planning benchmark completed")),
            ],
        )))
    return actions


def generate_launch_description():
    return LaunchDescription([
        *[
            DeclareLaunchArgument(
                name,
                default_value=(os.path.join(get_package_share_directory("myrobot_simulation"), "rviz", "fairino_planning_test.rviz")
                               if name == "rviz_config" and not _DEFAULTS[name] else _DEFAULTS[name]),
                choices=(['interactive', 'benchmark_execution', 'benchmark_algorithm'] if name == 'run_mode' else None),
                description="业务 launch 参数。",
            )
            for name in _PUBLIC_ARGUMENTS
        ],
        OpaqueFunction(function=_setup),
    ])
