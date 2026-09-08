"""Gazebo planning/IK demo with a narrow public launch contract."""

import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, LogInfo,
    OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable, TimerAction,
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
    load_yaml,
)


_PARAMS_FILE = "config/motion_planning_demo_params.yaml"
_PUBLIC_ARGUMENTS = (
    "robot_profile", "enable_rviz", "world", "use_sim_time", "initial_positions_file",
    "enable_camera_model", "rviz_config", "ik_plugin", "planning_pipeline_id",
    "planner_id", "planner_random_seed", "move_group_ready_timeout_sec", "allowed_planning_time",
    "run_mode", "goal_collection_dir", "benchmark_output_dir",
    "benchmark_scene_name", "benchmark_goal_seed", "benchmark_goal_root_mode", "benchmark_variant",
    "start_joints", "start_id",
)
_RAW_DEFAULTS = load_launch_parameters_yaml("myrobot_simulation", _PARAMS_FILE, None)
_BENCHMARK = load_yaml("myrobot_simulation", _PARAMS_FILE).get("benchmark", {})
_RAW_DEFAULTS["planner_random_seed"] = _BENCHMARK["random"]["planner_seed"]
_RAW_DEFAULTS["benchmark_scene_name"] = _BENCHMARK["scene"]["scene_name"]
_RAW_DEFAULTS["benchmark_goal_seed"] = _BENCHMARK["goals"]["seed"]
_RAW_DEFAULTS["benchmark_goal_root_mode"] = _BENCHMARK["goals"]["goal_root_mode"]
_DEFAULTS = launch_defaults_as_strings(_RAW_DEFAULTS)
_CONFIG = {name: LaunchConfiguration(name) for name in _PUBLIC_ARGUMENTS}


def _value(context, name):
    return _CONFIG[name].perform(context).strip()


def _benchmark_slug(value):
    text = str(value).strip()
    if text.endswith("*"):
        text = f"{text[:-1]}_star"
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in text)


def _benchmark_parameters(context):
    """Project nested benchmark YAML explicitly into flat ROS parameters."""
    comparison = dict(_BENCHMARK["comparison"])
    variant = _value(context, "benchmark_variant")
    if variant not in ("full", "cost_only_queue", "eager_edge_validation", "cost_only_eager"):
        raise ValueError("invalid MIRE ablation variant")
    comparison["mire_ablation_variant"] = variant
    comparison["mire_enable_effort_focal_queue"] = variant not in (
        "cost_only_queue", "cost_only_eager")
    comparison["mire_enable_lazy_edge_validation"] = variant not in (
        "eager_edge_validation", "cost_only_eager")
    scene = dict(_BENCHMARK["scene"])
    goals = dict(_BENCHMARK["goals"])
    scene["scene_name"] = _value(context, "benchmark_scene_name")
    goals["seed"] = int(_value(context, "benchmark_goal_seed"))
    goals["goal_root_mode"] = _value(context, "benchmark_goal_root_mode")
    return comparison, {
        "target_rpy_deg": ",".join(str(value) for value in goals["target_rpy_deg"]),
        "scene_name": scene["scene_name"],
        "spawn_sim_scene_models": bool(scene["spawn_sim_scene_models"]),
        "publish_planning_scene": bool(scene["publish_planning_scene"]),
        "publish_obstacle_markers": bool(scene["publish_obstacle_markers"]),
        "obstacle_marker_topic": scene["obstacle_marker_topic"],
        "planning_scene_obstacle_padding_m": float(scene["planning_scene_obstacle_padding_m"]),
        "benchmark_repetitions": int(goals["repetitions"]),
        "benchmark_goal_mode": goals["mode"],
        "ik_mode": goals["ik_mode"],
        "benchmark_goal_root_mode": goals["goal_root_mode"],
        "benchmark_goal_seed": int(goals["seed"]),
        "benchmark_goal_clearance_min_m": float(goals["clearance_min_m"]),
        "benchmark_goal_clearance_max_m": float(goals["clearance_max_m"]),
        "benchmark_goal_corridor_clearance_max_m": float(goals["corridor_clearance_max_m"]),
        "benchmark_goal_min_separation_m": float(goals["min_separation_m"]),
        "benchmark_goal_state_validity_timeout_s": float(goals["state_validity_timeout_s"]),
    }


def _node_parameters(context):
    params = load_node_parameters_yaml(
        "myrobot_simulation", _PARAMS_FILE, "motion_planning_node_sim", None
    )
    run_mode = _value(context, "run_mode")
    if run_mode not in ("interactive", "goal_collection", "benchmark_execution", "benchmark_algorithm"):
        raise RuntimeError(
            "run_mode must be interactive, goal_collection, benchmark_execution, or benchmark_algorithm"
        )
    goal_collection_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(
        _value(context, "goal_collection_dir")
    ))) if _value(context, "goal_collection_dir") else ""
    output_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(
        _value(context, "benchmark_output_dir")
    ))) if _value(context, "benchmark_output_dir") else ""
    if run_mode == "goal_collection" and not goal_collection_dir:
        raise RuntimeError("goal_collection_dir is required for goal_collection run_mode")
    if run_mode in ("benchmark_execution", "benchmark_algorithm") and not output_dir:
        raise RuntimeError("benchmark_output_dir is required for benchmark run_mode")
    comparison, benchmark_params = _benchmark_parameters(context)
    return {
        **params,
        **benchmark_params,
        "planning_client": _value(context, "ik_plugin"),
        "default_pipeline_id": _value(context, "planning_pipeline_id"),
        "default_planner_id": _value(context, "planner_id"),
        "allowed_planning_time": float(_value(context, "allowed_planning_time")),
        "planner_random_seed": int(_value(context, "planner_random_seed")),
        "benchmark_variant": _value(context, "benchmark_variant"),
        "start_joints": _value(context, "start_joints"),
        "start_id": _value(context, "start_id"),
        "ik_timeout": launch_parameter_value(
            _value(context, "move_group_ready_timeout_sec"),
            _RAW_DEFAULTS["move_group_ready_timeout_sec"],
        ),
        "use_sim_time": _value(context, "use_sim_time").lower() == "true",
        "sim_world": _value(context, "world"),
        "run_mode": run_mode,
        "goal_collection_dir": goal_collection_dir,
        "benchmark_output_dir": output_dir,
        "benchmark_effective_config_json": json.dumps(
            {"comparison": comparison, "scene": _BENCHMARK["scene"],
             "goals": _BENCHMARK["goals"], "random": {"planner_seed": int(_value(context, "planner_random_seed"))}},
            sort_keys=True,
        ),
    }


def _setup(context, *_args, **_kwargs):
    gz_share = get_package_share_directory("myrobot_simulation")
    node_params = _node_parameters(context)
    benchmark_mode = node_params["run_mode"] in ("benchmark_execution", "benchmark_algorithm")
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
            "fairino_ik_task_profile": (
                node_params["ik_mode"]
                if node_params["run_mode"] != "interactive" else "grasp"
            ),
            "benchmark_goal_root_mode": (
                node_params["benchmark_goal_root_mode"]
                if benchmark_mode else ""
            ),
            "benchmark_comparison_json": (
                json.dumps(_benchmark_parameters(context)[0], sort_keys=True)
                if benchmark_mode else "{}"
            ),
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
    actions = []
    benchmark_planners = {
        "rrt*", "informed_rrt*", "birrt*", "aapf_birrt*",
        "mire_biait*", "prm",
    }
    if (benchmark_mode and
            node_params["default_planner_id"] in benchmark_planners):
        stats_name = (
            f".planner_diagnostics_{_benchmark_slug(node_params['default_planner_id'])}"
            f"_seed{node_params['planner_random_seed']}.csv"
        )
        actions.append(SetEnvironmentVariable(
            "FAIRINO_PLANNER_DIAGNOSTICS_PATH",
            os.path.join(node_params["benchmark_output_dir"], stats_name),
        ))
        planner_slug = _benchmark_slug(node_params["default_planner_id"])
        planner_seed = node_params["planner_random_seed"]
        for environment_name, prefix in (
            ("FAIRINO_ANYTIME_TRACE_PATH", "anytime_trace"),
            ("FAIRINO_ROOT_DIAGNOSTICS_PATH", "root_diagnostics"),
            ("FAIRINO_TRAJECTORY_PATHS_PATH", "trajectory_paths"),
        ):
            sidecar_name = ".{}_{}_seed{}.csv".format(
                prefix, planner_slug, planner_seed
            )
            actions.append(SetEnvironmentVariable(
                environment_name, os.path.join(node_params["benchmark_output_dir"], sidecar_name)
            ))
    if benchmark_mode and node_params["default_planner_id"] == "aapf_birrt*":
        stats_name = (
            f".aapf_sampling_stats_{_benchmark_slug(node_params['default_planner_id'])}"
            f"_seed{node_params['planner_random_seed']}.csv"
        )
        actions.append(SetEnvironmentVariable(
            "FAIRINO_AAPF_STATS_PATH", os.path.join(node_params["benchmark_output_dir"], stats_name)
        ))
    if (benchmark_mode and
            node_params["default_planner_id"] == "mire_biait*"):
        stats_name = (
            f".mire_stats_{_benchmark_slug(node_params['default_planner_id'])}"
            f"_seed{node_params['planner_random_seed']}.csv"
        )
        actions.append(SetEnvironmentVariable(
            "FAIRINO_MIRE_STATS_PATH", os.path.join(node_params["benchmark_output_dir"], stats_name)
        ))
    actions.extend([
        simulation,
        TimerAction(period=5.0, actions=[
            LogInfo(msg=["[motion_planning_demo] mode=", node_params["run_mode"]]), node,
        ]),
    ])
    if node_params["run_mode"] != "interactive":
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[
                LogInfo(msg="[motion_planning_demo] non-interactive node exited; shutting down launch."),
                EmitEvent(event=Shutdown(reason="planning non-interactive mode completed")),
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
                choices=(['interactive', 'goal_collection', 'benchmark_execution', 'benchmark_algorithm'] if name == 'run_mode' else None),
                description="业务 launch 参数。",
            )
            for name in _PUBLIC_ARGUMENTS
        ],
        OpaqueFunction(function=_setup),
    ])
