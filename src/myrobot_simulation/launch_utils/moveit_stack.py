"""MoveIt-related launch helpers for myrobot_simulation."""

from typing import Any, Dict, Optional

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from .robot_profiles import RobotProfile
from myrobot_common.launch_utils.yaml_loader import (
    load_yaml,
    package_file,
    wrap_yaml_as_ros_params_file,
)


def _as_xacro_bool(value: bool) -> str:
    return "true" if value else "false"


def build_moveit_config(
    profile: RobotProfile,
    enable_camera_model: Optional[bool] = None,
    extra_mappings: Optional[Dict[str, str]] = None,
):
    """Create a MoveItConfigs object for the selected robot profile."""
    camera_enabled = profile.has_camera if enable_camera_model is None else enable_camera_model
    builder = (
        MoveItConfigsBuilder(profile.moveit_config_name, package_name=profile.moveit_config_package)
        .robot_description(
            package_file("myrobot_simulation", profile.sim_xacro),
            mappings={
                "enable_camera": _as_xacro_bool(camera_enabled),
                "controllers_file": package_file(
                    profile.moveit_config_package, profile.controllers_file
                ),
                "initial_positions_file": package_file(
                    profile.moveit_config_package, profile.initial_positions_file
                ),
                **(extra_mappings or {}),
            },
        )
        .robot_description_semantic(profile.semantic_file)
        .robot_description_kinematics(
            package_file(profile.moveit_config_package, profile.default_kinematics_file)
        )
        .planning_pipelines(
            pipelines=profile.planning_pipelines,
            default_planning_pipeline=profile.default_planning_pipeline,
        )
    )
    if profile.moveit_controllers_file:
        builder.trajectory_execution(
            file_path=profile.moveit_controllers_file,
            moveit_manage_controllers=False,
        )
    return builder.to_moveit_configs()


def planning_parameter_configs(profile: RobotProfile) -> Dict[str, Any]:
    """Load planning and kinematics YAML as dictionaries for launch injection."""
    return {
        "fairino_planning": load_yaml(profile.moveit_config_package, profile.planning_pipeline_file),
        "kinematics_fairino": load_yaml(profile.moveit_config_package, profile.kinematics_fairino_file),
        "kinematics_kdl": load_yaml(profile.moveit_config_package, profile.kinematics_kdl_file),
        "sensors_3d": wrap_yaml_as_ros_params_file(
            profile.moveit_config_package, "config/sensors_3d.yaml"
        ),
        "planning_core": load_yaml("myrobot_planning_core", "config/common_planning_params.yaml"),
        "aapf_birrt_star_core": load_yaml("myrobot_planning_core", "config/aapf_birrt*_params.yaml"),
        "tube_birrt_star_core": load_yaml("myrobot_planning_core", "config/tube_birrt*_params.yaml"),
        "birrt_star_core": load_yaml("myrobot_planning_core", "config/birrt*_params.yaml"),
        "rrt_star_core": load_yaml("myrobot_planning_core", "config/rrt*_params.yaml"),
        "ik_core": load_yaml("myrobot_planning_core", "config/ik_params.yaml"),
        "cartesian_path_planner": load_yaml(
            "myrobot_planning_core", "config/cartesian_path_planner_params.yaml"
        ),
    }


def robot_description_with_package_paths(moveit_config, profile: RobotProfile) -> str:
    """Expand ALL package:// references to filesystem paths for ros_gz_sim."""
    import re
    description = moveit_config.robot_description["robot_description"]

    def resolve(m):
        pkg = m.group(1)
        try:
            return "file://" + get_package_share_directory(pkg)
        except Exception:
            return m.group(0)

    return re.sub(r'package://([^/]+)', resolve, description)


def rviz_node(moveit_config, profile: RobotProfile, rviz_config: str, use_sim_time: bool, move_group_client: str = "fairino"):
    params = planning_parameter_configs(profile)
    # RViz MotionPlanning display defaults to root namespace; remap it to the
    # active MoveIt client so GUI planning/execution stays aligned with simulation.
    remappings = [
        ("get_planning_scene", f"/move_group_{move_group_client}/get_planning_scene"),
        ("plan_kinematic_path", f"/move_group_{move_group_client}/plan_kinematic_path"),
        ("query_planner_interfaces", f"/move_group_{move_group_client}/query_planner_interfaces"),
        ("compute_cartesian_path", f"/move_group_{move_group_client}/compute_cartesian_path"),
        ("execute_trajectory", f"/move_group_{move_group_client}/execute_trajectory"),
        ("move_action", f"/move_group_{move_group_client}/move_action"),
        ("monitored_planning_scene", f"/move_group_{move_group_client}/monitored_planning_scene"),
    ]
    return Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        remappings=remappings,
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            params["fairino_planning"],
            {"use_sim_time": use_sim_time},
        ],
    )


def move_group_nodes(moveit_config, profile: RobotProfile, use_sim_time: bool, planner_random_seed: int = 0, clients=("fairino", "kdl")):
    """Build only the requested MoveIt client nodes for the selected profile."""
    clients = tuple(clients)
    if not clients or any(client not in ("fairino", "kdl") for client in clients):
        raise ValueError("MoveIt clients must be a non-empty subset of fairino,kdl")
    params = planning_parameter_configs(profile)
    remappings = [
        ("joint_states", "/joint_states"),
        # All manipulation clients publish the standard MoveIt stop event on
        # the root topic. Without this remap, namespaced move_group instances
        # listen under their namespace and cannot stop an active trajectory.
        ("trajectory_execution_event", "/trajectory_execution_event"),
        # Obstacle publishers in demos use root topics. Keep both move_group
        # instances subscribed there so PlanningScene collision objects are not
        # lost under /move_group_fairino or /move_group_kdl namespaces.
        ("planning_scene", "/planning_scene"),
        ("collision_object", "/collision_object"),
        ("attached_collision_object", "/attached_collision_object"),
        (
            f"{profile.arm_controller.lstrip('/')}/follow_joint_trajectory",
            f"{profile.arm_controller}/follow_joint_trajectory",
        ),
    ]
    if profile.has_gripper and profile.hand_controller:
        remappings.append(
            (
                f"{profile.hand_controller.lstrip('/')}/follow_joint_trajectory",
                f"{profile.hand_controller}/follow_joint_trajectory",
            )
        )
    fairino_ik_grasp_profile = {"fairino": {"ik": {"task_profile": "grasp"}}}
    fairino_ik_continuous_profile = {"fairino": {"ik": {"task_profile": "continuous"}}}
    planner_seed_param = {"planner": {"random_seed": int(planner_random_seed)}}

    fairino_parameters = [
        moveit_config.to_dict(),
        params["kinematics_fairino"],
        params["sensors_3d"],
        params["fairino_planning"],
        params["planning_core"],
        params["aapf_birrt_star_core"],
        params["tube_birrt_star_core"],
        params["birrt_star_core"],
        params["rrt_star_core"],
        params["ik_core"],
        fairino_ik_grasp_profile,
        planner_seed_param,
        {"use_sim_time": use_sim_time},
    ]
    fairino_cartesian_parameters = [
        *fairino_parameters,
        params["cartesian_path_planner"],
        fairino_ik_continuous_profile,
    ]

    kdl_parameters = [
        moveit_config.to_dict(),
        params["kinematics_kdl"],
        params["sensors_3d"],
        params["fairino_planning"],
        params["planning_core"],
        params["aapf_birrt_star_core"],
        params["tube_birrt_star_core"],
        params["birrt_star_core"],
        params["rrt_star_core"],
        params["ik_core"],
        planner_seed_param,
        {"use_sim_time": use_sim_time},
    ]

    nodes = []
    if "fairino" in clients:
        nodes.append(Node(
            package="moveit_ros_move_group",
            executable="move_group",
            namespace="move_group_fairino",
            name="move_group",
            output="screen",
            remappings=remappings,
            parameters=fairino_parameters,
        ))
    if "kdl" in clients:
        nodes.append(Node(
            package="moveit_ros_move_group",
            executable="move_group",
            namespace="move_group_kdl",
            name="move_group",
            output="screen",
            remappings=remappings,
            parameters=kdl_parameters,
        ))
    nodes.append(Node(
        package="myrobot_planning_ros",
        executable="fairino_cartesian_path_server",
        name="fairino_cartesian_path_server",
        output="screen",
        parameters=fairino_cartesian_parameters,
    ))
    return nodes
