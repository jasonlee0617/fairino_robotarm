"""Simulation launch helpers backed by Gazebo/GZ."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from .controllers import controller_spawner_actions
from .moveit_stack import (
    build_moveit_config,
    move_group_nodes,
    robot_description_with_package_paths,
    rviz_node,
)
from .robot_profiles import RobotProfile


def sim_resource_path(profile: RobotProfile):
    gz_share = get_package_share_directory("myrobot_simulation")
    desc_share = get_package_share_directory(profile.description_package)
    try:
        realsense_share = get_package_share_directory("realsense2_description")
    except Exception:
        realsense_share = None
    paths = [
        os.path.join(gz_share, "worlds"),
        os.path.join(gz_share, "worlds", "models"),
        str(Path(desc_share).parent.resolve()),
    ]
    if realsense_share:
        paths.append(str(Path(realsense_share).resolve()))
    return SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=":".join(paths),
    )


def sim_node(world: str, *, headless: bool = False):
    gz_args = "empty.sdf -r" if world == "empty" else f"{world}.sdf -r"
    if headless:
        gz_args = f"{gz_args} -s"
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )


def clock_bridge_node(use_sim_time: bool):
    return Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="both",
        parameters=[{"use_sim_time": use_sim_time}],
    )


def robot_spawn_node(moveit_config, profile: RobotProfile, spawn_xyz, spawn_rpy, spawn_name: str):
    robot_description = robot_description_with_package_paths(moveit_config, profile)
    return Node(
        package="ros_gz_sim",
        executable="create",
        output="both",
        arguments=[
            "-string",
            robot_description,
            "-x",
            str(spawn_xyz[0]),
            "-y",
            str(spawn_xyz[1]),
            "-z",
            str(spawn_xyz[2]),
            "-R",
            str(spawn_rpy[0]),
            "-P",
            str(spawn_rpy[1]),
            "-Y",
            str(spawn_rpy[2]),
            "-name",
            spawn_name or profile.spawn_name,
            "-allow_renaming",
            "false",
        ],
    )


def robot_state_publisher_node(moveit_config, use_sim_time: bool, publish_frequency: float):
    return Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": use_sim_time},
            {"publish_frequency": publish_frequency},
        ],
    )


def base_simulation_actions(
    profile: RobotProfile,
    *,
    world: str,
    rviz_config: str,
    spawn_xyz: Optional[List[float]] = None,
    spawn_rpy: Optional[List[float]] = None,
    spawn_name: str = "",
    use_sim_time: bool = True,
    enable_rviz: bool = True,
    publish_frequency: float = 100.0,
    enable_camera_model: Optional[bool] = None,
    robot_spawn_delay: float = 5.0,
    controller_spawn_delay: float = 8.0,
    planner_random_seed: int = 0,
    moveit_clients: Tuple[str, ...] = ("fairino", "kdl"),
    fairino_ik_task_profile: str = "grasp",
    benchmark_goal_root_mode: str = "",
    benchmark_comparison: Optional[Dict[str, object]] = None,
    extra_mappings: Optional[Dict[str, str]] = None,
):
    moveit_config = build_moveit_config(
        profile,
        enable_camera_model=enable_camera_model,
        extra_mappings=extra_mappings,
    )
    xyz = spawn_xyz if spawn_xyz is not None else profile.spawn_xyz
    rpy = spawn_rpy if spawn_rpy is not None else profile.spawn_rpy

    robot_spawn = robot_spawn_node(moveit_config, profile, xyz, rpy, spawn_name)
    controller_spawners = controller_spawner_actions(profile)

    actions = [
        sim_resource_path(profile),
        # Benchmark launches disable RViz and must not start Gazebo's Qt GUI.
        # Interactive launches retain the existing GUI behavior.
        sim_node(world, headless=not enable_rviz),
        clock_bridge_node(use_sim_time),
        robot_state_publisher_node(moveit_config, use_sim_time, publish_frequency),
        *move_group_nodes(
            moveit_config,
            profile,
            use_sim_time,
            planner_random_seed,
            moveit_clients,
            fairino_ik_task_profile,
            benchmark_goal_root_mode,
            benchmark_comparison,
        ),
        TimerAction(period=max(0.0, robot_spawn_delay), actions=[robot_spawn]),
        TimerAction(period=max(0.0, controller_spawn_delay), actions=controller_spawners),
    ]
    if enable_rviz:
        actions.append(
            TimerAction(
                period=max(0.0, controller_spawn_delay + 1.0),
                actions=[rviz_node(moveit_config, profile, rviz_config, use_sim_time, moveit_clients[0])],
            )
        )
    return actions, moveit_config
