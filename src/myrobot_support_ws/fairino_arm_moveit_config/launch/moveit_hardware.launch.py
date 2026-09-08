"""Real Fairino hardware: one control stack and one safe active MoveIt executor."""

from copy import deepcopy
from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg


_EXECUTION_CAPABILITIES = " ".join((
    "move_group/MoveGroupExecuteTrajectoryAction",
    "move_group/MoveGroupExecuteService",
))
_PLANNING_ONLY_CONTROLLER = "__planning_only_controller__"
_PLANNING_ONLY_JOINT = "__planning_only_joint__"


def _planning_config(filename: str) -> dict:
    """Load the plain planning YAML mappings used by the Cartesian server."""
    path = Path(get_package_share_directory("myrobot_planning_core")) / "config" / filename
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _condition(executor: str) -> IfCondition:
    return IfCondition(PythonExpression([
        "'", LaunchConfiguration("execution_ik"), f"' == '{executor}'",
    ]))


def _inactive_condition(executor: str) -> IfCondition:
    return IfCondition(PythonExpression([
        "'", LaunchConfiguration("execution_ik"), f"' != '{executor}'",
    ]))


def _planning_only_parameters(common: dict) -> dict:
    """Keep planning available without exposing a real controller endpoint."""
    parameters = deepcopy(common)
    for name in (
        "moveit_simple_controller_manager",
        "moveit_manage_controllers",
        "trajectory_execution",
    ):
        parameters.pop(name, None)
    parameters.update({
        "allow_trajectory_execution": False,
        "disable_capabilities": _EXECUTION_CAPABILITIES,
        # Humble constructs TrajectoryExecutionManager even with execution disabled.
        # ROS 2 cannot represent an empty controller_names array, so provide a
        # local, non-executable placeholder instead of any real controller.
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "moveit_simple_controller_manager": {
            "controller_names": [_PLANNING_ONLY_CONTROLLER],
            _PLANNING_ONLY_CONTROLLER: {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "joints": [_PLANNING_ONLY_JOINT],
                "default": False,
            },
        },
    })
    return parameters


def _moveit_parameters(kinematics_file: str, default_pipeline: str) -> dict:
    return (
        MoveItConfigsBuilder(
            "fairino_arm_moveit_descriptions",
            package_name="fairino_arm_moveit_config",
        )
        .robot_description_kinematics(file_path=f"config/{kinematics_file}")
        .planning_pipelines(default_planning_pipeline=default_pipeline)
        .trajectory_execution(
            file_path="config/moveit_controllers_real.yaml",
            moveit_manage_controllers=False,
        )
        .to_moveit_configs()
        .to_dict()
    )


def _move_group_configuration(*, active: bool) -> dict:
    should_publish = ParameterValue(
        LaunchConfiguration("publish_monitored_planning_scene"), value_type=bool
    )
    return {
        "publish_robot_description_semantic": True,
        "default_planning_pipeline": ParameterValue(
            LaunchConfiguration("execution_pipeline"), value_type=str
        ),
        "allow_trajectory_execution": (
            ParameterValue(LaunchConfiguration("allow_trajectory_execution"), value_type=bool)
            if active
            else False
        ),
        "capabilities": ParameterValue(LaunchConfiguration("capabilities"), value_type=str),
        "disable_capabilities": (
            ParameterValue(LaunchConfiguration("disable_capabilities"), value_type=str)
            if active
            else ParameterValue(
                [
                    LaunchConfiguration("disable_capabilities"),
                    " ",
                    _EXECUTION_CAPABILITIES,
                ],
                value_type=str,
            )
        ),
        "publish_planning_scene": should_publish,
        "publish_geometry_updates": should_publish,
        "publish_state_updates": should_publish,
        "publish_transforms_updates": should_publish,
        "monitor_dynamics": ParameterValue(
            LaunchConfiguration("monitor_dynamics"), value_type=bool
        ),
    }


def _move_group(namespace: str, parameters: list[dict], condition: IfCondition) -> GroupAction:
    common = {
        "package": "moveit_ros_move_group",
        "executable": "move_group",
        "namespace": namespace,
        "name": f"{namespace}_server",
        "output": "screen",
        "remappings": [
            ("joint_states", "/joint_states"),
            ("planning_scene", "/planning_scene"),
            ("collision_object", "/collision_object"),
            ("attached_collision_object", "/attached_collision_object"),
            ("trajectory_execution_event", "/trajectory_execution_event"),
        ],
        "parameters": parameters,
    }
    commands_file = (
        Path(get_package_share_directory("fairino_arm_moveit_config"))
        / "launch"
        / "gdb_settings.gdb"
    )
    return GroupAction(
        condition=condition,
        actions=[
            Node(
                condition=UnlessCondition(LaunchConfiguration("debug")),
                **common,
            ),
            Node(
                condition=IfCondition(LaunchConfiguration("debug")),
                prefix=[f"gdb -x {commands_file} --ex run --args"],
                arguments=["--debug"],
                **common,
            ),
        ],
    )


def _planning_pipeline_parameters(moveit_parameters: dict) -> dict:
    names = moveit_parameters["planning_pipelines"]
    return {
        "planning_pipelines": names,
        "default_planning_pipeline": moveit_parameters["default_planning_pipeline"],
        **{name: moveit_parameters[name] for name in names},
    }


def _rviz(config, moveit_parameters: dict) -> Node:
    """Run RViz with the caller's static, namespaced configuration file."""
    return Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        remappings=[("joint_states", "/joint_states")],
        prefix=PythonExpression([
            "'gdb --ex run --args' if '",
            LaunchConfiguration("debug"),
            "'.lower() == 'true' else ''",
        ]),
        parameters=[
            config.robot_description,
            config.robot_description_semantic,
            config.joint_limits,
            {"robot_description_kinematics": moveit_parameters["robot_description_kinematics"]},
            _planning_pipeline_parameters(moveit_parameters),
        ],
    )


def generate_launch_description():
    package = Path(get_package_share_directory("fairino_arm_moveit_config"))
    rviz_config = MoveItConfigsBuilder(
        "fairino_arm_moveit_descriptions",
        package_name="fairino_arm_moveit_config",
    ).trajectory_execution(
        file_path="config/moveit_controllers_real.yaml",
        moveit_manage_controllers=False,
    ).to_moveit_configs()
    fairino_base_parameters = _moveit_parameters("kinematics_fairino.yaml", "fairino")
    kdl_base_parameters = _moveit_parameters("kinematics_kdl.yaml", "ompl")
    core_planning_parameters = [
        _planning_config(filename)
        for filename in (
            "common_planning_params.yaml",
            "mire_biait*_params.yaml",
            "aapf_birrt*_params.yaml",
            "birrt*_params.yaml",
            "rrt_params.yaml",
            "rrt*_params.yaml",
            "prm_params.yaml",
            "ik_params.yaml",
        )
    ]
    fairino_parameters = [fairino_base_parameters, *core_planning_parameters]
    kdl_parameters = [kdl_base_parameters, *core_planning_parameters]
    fairino_cartesian_parameters = [
        *fairino_parameters,
        _planning_config("cartesian_path_planner_params.yaml"),
        {"fairino": {"ik": {"task_profile": "continuous"}}},
    ]
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    robot_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["robot_arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    hand_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["hand_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    moveit_actions = [
        _move_group(
            "move_group_fairino",
            [*fairino_parameters, _move_group_configuration(active=True)],
            _condition("fairino"),
        ),
        _move_group(
            "move_group_fairino",
            [
                _planning_only_parameters(fairino_base_parameters),
                *core_planning_parameters,
                _move_group_configuration(active=False),
            ],
            _inactive_condition("fairino"),
        ),
        _move_group(
            "move_group_kdl",
            [*kdl_parameters, _move_group_configuration(active=True)],
            _condition("kdl"),
        ),
        _move_group(
            "move_group_kdl",
            [
                _planning_only_parameters(kdl_base_parameters),
                *core_planning_parameters,
                _move_group_configuration(active=False),
            ],
            _inactive_condition("kdl"),
        ),
        Node(
            package="myrobot_planning_ros",
            executable="fairino_cartesian_path_server",
            name="fairino_cartesian_path_server",
            output="screen",
            parameters=fairino_cartesian_parameters,
        ),
        GroupAction(
            condition=_condition("fairino"),
            actions=[_rviz(
                rviz_config,
                fairino_base_parameters,
            )],
        ),
        GroupAction(
            condition=_condition("kdl"),
            actions=[_rviz(
                rviz_config,
                kdl_base_parameters,
            )],
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "execution_ik",
            default_value="fairino",
            choices=["fairino", "kdl"],
            description="Internal selected IK client; only this MoveIt server may execute real trajectories.",
        ),
        DeclareLaunchArgument(
            "execution_pipeline",
            default_value="fairino",
            choices=["fairino", "ompl"],
            description="Internal planning pipeline selected by the task YAML.",
        ),
        DeclareBooleanLaunchArg("debug", default_value=False),
        DeclareBooleanLaunchArg("use_rviz", default_value=True),
        DeclareBooleanLaunchArg("allow_trajectory_execution", default_value=True),
        DeclareBooleanLaunchArg(
            "publish_monitored_planning_scene", default_value=True
        ),
        DeclareBooleanLaunchArg("monitor_dynamics", default_value=False),
        DeclareLaunchArgument(
            "capabilities",
            default_value=rviz_config.move_group_capabilities["capabilities"],
        ),
        DeclareLaunchArgument(
            "disable_capabilities",
            default_value=rviz_config.move_group_capabilities["disable_capabilities"],
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(package / "config" / "moveit.rviz"),
        ),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(package / "launch" / "static_virtual_joint_tfs.launch.py")
        )),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(package / "launch" / "rsp.launch.py")
        )),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[str(package / "config" / "ros2_controllers_real.yaml")],
            remappings=[("~/robot_description", "/robot_description")],
        ),
        joint_state_broadcaster,
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[robot_arm_controller],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=robot_arm_controller,
            on_exit=[hand_controller],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=hand_controller,
            on_exit=moveit_actions,
        )),
    ])
