import ast
import importlib.util
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace

from launch import LaunchContext
import yaml

LAUNCH_ROOT = Path(__file__).resolve().parents[1] / "launch"
if str(LAUNCH_ROOT) not in sys.path:
    sys.path.insert(0, str(LAUNCH_ROOT))

from handeye_launch_utils import (
    default_from_settings,
    load_handeye_profile,
)
from hand_eye_calibration.config import flatten_ros_parameters
MOVEIT_DEMO = (
    Path(__file__).resolve().parents[3]
    / "myrobot_support_ws" / "fairino_arm_moveit_config" / "launch" / "demo.launch.py"
)
MOVEIT_HARDWARE = MOVEIT_DEMO.with_name("moveit_hardware.launch.py")
REMOVED_CONFIG = "handeye" + "_bringup_params.yaml"


def _source(name):
    return (LAUNCH_ROOT / name).read_text(encoding="utf-8")


def _load_moveit_hardware():
    spec = importlib.util.spec_from_file_location("fairino_hardware_launch", MOVEIT_HARDWARE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_launch_module(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".", "_"), LAUNCH_ROOT / name
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_calibrate_only_starts_environment_and_vision():
    source = _source("calibrate.launch.py")

    assert "ros2_aruco" in source
    assert "visualize_aruco_marker.py" in source
    assert '"moveit_hardware.launch.py"' in source
    assert "start_demo_moveit" not in source
    assert '"rgb_camera.color_profile"' in source
    assert '"depth_module.depth_profile"' in source
    assert 'profile["rviz_config"]' in source
    assert '"active_executor"' not in source
    assert '"execution_ik"' in source
    assert '"execution_pipeline"' in source
    for argument in (
        "debug",
        "allow_trajectory_execution",
        "publish_monitored_planning_scene",
        "monitor_dynamics",
        "capabilities",
        "disable_capabilities",
    ):
        assert f'"{argument}"' in source
    assert "easy_handeye2" not in source
    assert "manual_calibration_assistant.py" not in source
    assert "auto_calibration_collector.py" not in source


def test_real_hardware_launch_has_no_warehouse_database_path():
    hardware_source = MOVEIT_HARDWARE.read_text(encoding="utf-8")
    calibrate_source = _source("calibrate.launch.py")

    for name in (
        "warehouse_db.launch.py",
        '"db"',
        "moveit_warehouse_database_path",
        "moveit_warehouse_port",
        "moveit_warehouse_host",
        '"reset"',
    ):
        assert name not in hardware_source
        assert name not in calibrate_source


def test_assisted_launch_only_starts_easy_and_manual_assistant():
    source = _source("assisted_calibration.launch.py")
    parameters = flatten_ros_parameters(yaml.safe_load(
        (LAUNCH_ROOT.parent / "config" / "manual_calibration_assistant_params.yaml").read_text(
            encoding="utf-8"
        )
    )["manual_calibration_assistant"]["ros__parameters"])

    assert "easy_handeye2" in source
    assert "manual_calibration_assistant.py" in source
    assert "auto_calibration.launch.py" not in source
    assert 'package="ros2_aruco"' not in source
    assert 'package="realsense2_camera"' not in source
    assert "manual_calibration_assistant_params.yaml" in source
    assert "auto_calibration_collector_params.yaml" not in source
    assert "_LAUNCH_DEFAULTS" in source
    assert "_LAUNCH_CONFIGURATIONS" in source
    assert "_TOPOLOGY_PARAMETER_NAMES" in source
    assert "if name not in _TOPOLOGY_PARAMETER_NAMES" in source
    assert "_ASSISTANT_NODE_DEFAULTS," in source
    assert "_ASSISTANT_YAML_PARAMETERS," in source
    assert parameters["calibration_type"] == "eye_on_base"
    assert parameters["use_sim_time"] is True
    assert "calibration_output_directory" in parameters
    assert "ground_truth_check_enabled" not in source


def test_handeye_launches_keep_profiles_in_the_shared_helper():
    for name in (
        "calibrate.launch.py",
        "assisted_calibration.launch.py",
        "evaluate.launch.py",
        "follow_aruco_move.launch.py",
    ):
        source = _source(name)
        if name == "follow_aruco_move.launch.py":
            assert "_LAUNCH_ARGUMENT_SPECS" in source
            assert "follow_aruco_move_params.yaml" in source
        else:
            assert "_DEFAULTS" in source or "_LAUNCH_DEFAULTS" in source
        assert "LaunchConfiguration" in source

    assisted_source = _source("assisted_calibration.launch.py")
    assert '"calibration_type": calibration_type' in assisted_source
    assert '"storage_directory": storage_directory' in assisted_source
    assert '"calibration_output_directory": storage_directory' in assisted_source


def test_follow_aruco_move_uses_hardware_moveit_and_shared_global_motion():
    launch = _source("follow_aruco_move.launch.py")
    follower = (LAUNCH_ROOT.parent / "scripts" / "follow_aruco_marker.py").read_text(
        encoding="utf-8"
    )
    config = yaml.safe_load(
        (LAUNCH_ROOT.parent / "config" / "follow_aruco_move_params.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert '"moveit_hardware.launch.py"' in launch
    assert '"demo.launch.py"' not in launch
    assert '"rgb_camera.color_profile"' in launch
    assert '"depth_module.depth_profile"' in launch
    assert '"calibration_aruco_publisher.py"' not in launch
    assert '"follow_aruco_move.rviz"' in launch
    assert "MoveItMotion" in follower
    assert "self.motion.move_to_pose" in follower
    assert "self.moveit2.move_to_pose" not in follower
    assert "_latest_target" in follower
    assert "marker_pose_timeout_sec" in follower
    assert "environments" not in config
    assert config["launch"]["color_profile"] == "1280x720x30"
    assert config["launch"]["depth_profile"] == "848x480x30"
    follow = config["nodes"]["aruco_marker_follower"]["ros__parameters"]
    assert follow["above_offset"] == 0.20
    assert follow["target_rpy_deg"] == [0.0, -180.0, 100.0]
    assert "from myrobot_common.utils.params import param" in follower
    assert "self.arm_group_name = self._string" in follower


def test_real_profiles_are_builtin_and_eye_on_base_uses_tool0():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "config" / REMOVED_CONFIG).exists()
    assert default_from_settings("calibration_type", "invalid") == "eye_in_hand"
    assert default_from_settings("camera_type", "invalid") == "realsense"
    for calibration_type in ("eye_in_hand", "eye_on_base"):
        profile = load_handeye_profile(calibration_type)
        assert profile["calibration_type"] == calibration_type
        assert profile["robot_effector_frame"] == "tool0"


def test_handeye_launches_use_the_independent_aruco_yaml():
    root = Path(__file__).resolve().parents[1]
    aruco = root / "config" / "aruco_parameters.yaml"
    assert aruco.exists()
    assert yaml.safe_load(aruco.read_text(encoding="utf-8"))["/aruco_node"]["ros__parameters"]["marker_size"] == 0.07
    for name in (
        "calibrate.launch.py",
        "evaluate.launch.py",
        "follow_aruco_move.launch.py",
    ):
        source = _source(name)
        assert "aruco_parameters.yaml" in source


def test_camera_launch_is_shared_outside_handeye_utilities():
    helper = _source("handeye_launch_utils.py")
    assert "def camera_launch" not in helper
    for name in (
        "calibrate.launch.py",
        "evaluate.launch.py",
        "follow_aruco_move.launch.py",
    ):
        assert "from myrobot_common.camera.launch import camera_launch" in _source(name)


def test_auto_collector_launch_is_isolated_and_exposes_motion_overrides():
    source = _source("auto_calibration_collector_launch.py")
    collector_root = LAUNCH_ROOT.parent / "hand_eye_calibration"
    collector_source = "\n".join(
        path.read_text(encoding="utf-8") for path in collector_root.rglob("*.py")
    )
    motion_executor_source = (
        LAUNCH_ROOT.parents[2] / "myrobot_common_ws" / "myrobot_common"
        / "myrobot_common" / "planning" / "motion_executor.py"
    ).read_text(encoding="utf-8")

    assert 'executable="auto_calibration_collector.py"' in source
    assert 'name="auto_calibration_collector"' in source
    assert "emulate_tty=True" in source
    assert 'output="screen"' in source
    assert "bash -c" not in source
    assert "exec </dev/tty" not in source
    for message in ("Fairino collector configured", "Time source from YAML", "Standby. Press Enter/s"):
        assert message in collector_source
    assert "Motion planning policy" in motion_executor_source
    assert "auto_calibration_collector_params.yaml" in source
    for name in (
        "calibration_type", "ik_plugin", "planning_pipeline_id", "planner_id",
        "max_velocity", "max_acceleration", "allowed_planning_time", "max_step_size",
        "position_tolerance", "orientation_tolerance", "allowed_start_tolerance",
        "moveit_ready_timeout", "moveit_ready_poll_interval",
    ):
        assert f'"{name}"' in source
    assert '"fairino": "/move_group_fairino"' in source
    assert '"kdl": "/move_group_kdl"' in source
    assert "launch_parameter_value" in source
    for forbidden in (
        "easy_handeye2", "ros2_aruco", "realsense2_camera",
        "moveit_hardware.launch.py", "camera_launch", "IncludeLaunchDescription",
    ):
        assert forbidden not in source


def test_auto_collector_motion_overrides_are_typed_and_select_kdl(monkeypatch):
    module = _load_launch_module("auto_calibration_collector_launch.py")
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)
    context = LaunchContext()
    context.launch_configurations.update(module._LAUNCH_DEFAULTS)
    context.launch_configurations.update({
        "calibration_type": "eye_on_base",
        "ik_plugin": "kdl",
        "max_velocity": "0.35",
        "max_acceleration": "0.4",
        "allowed_planning_time": "7.5",
    })

    node = module._overrides(context)[0]
    overrides = node["parameters"][-1]
    assert overrides["calibration_type"] == "eye_on_base"
    assert overrides["max_velocity"] == 0.35
    assert overrides["max_acceleration"] == 0.4
    assert overrides["allowed_planning_time"] == 7.5
    assert overrides["move_group_ns_fairino"] == "/move_group_kdl"


def test_handeye_launch_sources_do_not_reference_removed_bringup_yaml():
    root = Path(__file__).resolve().parents[1]
    for path in (*LAUNCH_ROOT.glob("*.py"), *root.rglob("*.yaml")):
        assert REMOVED_CONFIG not in path.read_text(encoding="utf-8")


def test_native_demo_is_not_the_real_hardware_entrypoint():
    source = MOVEIT_DEMO.read_text(encoding="utf-8")

    assert "generate_demo_launch" in source
    assert "move_group_fairino" not in source
    assert "ros2_control_node" not in source


def test_real_moveit_hardware_has_two_isolated_servers_and_one_controller_stack():
    source = MOVEIT_HARDWARE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    move_group_namespaces = [
        ast.literal_eval(call.args[0])
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_move_group"
        and call.args
    ]
    executables = [
        ast.literal_eval(keyword.value)
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "Node"
        for keyword in call.keywords
        if keyword.arg == "executable" and isinstance(keyword.value, ast.Constant)
    ]

    assert move_group_namespaces.count("move_group_fairino") == 2
    assert move_group_namespaces.count("move_group_kdl") == 2
    assert set(move_group_namespaces) == {"move_group_fairino", "move_group_kdl"}
    assert '"active_executor"' not in source
    assert executables.count("ros2_control_node") == 1
    assert executables.count("spawner") == 3
    assert '"allow_trajectory_execution": False' in source
    assert "moveit_controllers_real.yaml" in source
    assert "MoveGroupExecuteTrajectoryAction" in source
    assert "GroupAction" in source
    assert "OnProcessExit" in source
    assert 'remappings=[("joint_states", "/joint_states")]' in source
    assert "rviz_remappings" not in source


def test_inactive_moveit_parameters_bind_no_real_controller():
    module = _load_moveit_hardware()
    common = {
        "moveit_controller_manager": "manager",
        "moveit_simple_controller_manager": {"controller_names": ["robot_arm_controller"]},
        "moveit_manage_controllers": True,
        "trajectory_execution": {"allowed_goal_duration_margin": 1.0},
        "robot_description": "robot",
    }
    result = module._planning_only_parameters(common)
    controller = result["moveit_simple_controller_manager"]
    assert result["robot_description"] == "robot"
    assert result["allow_trajectory_execution"] is False
    assert "moveit_manage_controllers" not in result
    assert "trajectory_execution" not in result
    assert result["disable_capabilities"] == (
        "move_group/MoveGroupExecuteTrajectoryAction "
        "move_group/MoveGroupExecuteService"
    )
    assert result["moveit_controller_manager"] == (
        "moveit_simple_controller_manager/MoveItSimpleControllerManager"
    )
    assert controller["controller_names"] == ["__planning_only_controller__"]
    assert "/robot_arm_controller" not in controller
    assert "/hand_controller" not in controller
    assert controller["__planning_only_controller__"] == {
        "type": "FollowJointTrajectory",
        "action_ns": "follow_joint_trajectory",
        "joints": ["__planning_only_joint__"],
        "default": False,
    }

    # launch_ros converts non-empty arrays to typed tuples.  Empty arrays become
    # an untyped (), which rclpy rejects while constructing the node.
    assert controller["controller_names"]
    assert controller["__planning_only_controller__"]["joints"]
    assert common["moveit_simple_controller_manager"]["controller_names"] == [
        "robot_arm_controller"
    ]


def test_rviz_config_uses_selected_absolute_move_group_namespace(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))
    module = _load_moveit_hardware()
    module.Node = lambda **kwargs: kwargs
    config = SimpleNamespace(
        robot_description={},
        robot_description_semantic={},
        joint_limits={},
    )
    parameters = {
        "robot_description_kinematics": {},
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": {},
    }
    rviz = module._rviz(config, parameters)
    assert rviz["arguments"][0] == "-d"
    assert rviz["remappings"] == [("joint_states", "/joint_states")]


def test_real_controller_configuration_uses_root_action_endpoints():
    source = (
        MOVEIT_HARDWARE.parents[1] / "config" / "moveit_controllers_real.yaml"
    ).read_text(encoding="utf-8")

    assert "- /robot_arm_controller" in source
    assert "- /hand_controller" in source
    assert "/robot_arm_controller:" in source
    assert "/hand_controller:" in source
