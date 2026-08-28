from pathlib import Path
import re
from xml.etree import ElementTree

import yaml
import pytest

from myrobot_common.launch_utils.yaml_loader import (
    load_launch_parameters_yaml,
    load_moveit_parameters_yaml,
    load_node_parameters_yaml,
)
from visual_servo_bringup.position_servo_config import (
    aruco_detector_parameters,
    aruco_parameters,
    aruco_pose_source_parameters,
    sim_target_motion_parameters,
    visual_servo_parameters,
)


PACKAGE = Path(__file__).resolve().parents[1]
POSITION_SERVO_NODE = (
    PACKAGE / "visual_servo_bringup" / "nodes" / "visual_position_servo_node.py"
)
HARDWARE_LAUNCH = PACKAGE / "launch" / "visual_position_servo.launch.py"
IMAGE_HARDWARE_LAUNCH = PACKAGE / "launch" / "visual_image_servo.launch.py"
SIM_LAUNCH = PACKAGE.parent / "myrobot_simulation" / "launch" / "visual_position_servo_sim.launch.py"
ROBOTARM_WORLD = PACKAGE.parent / "myrobot_simulation" / "worlds" / "robotarm_world.sdf"
ARUCO_MODEL = (
    PACKAGE.parent / "myrobot_simulation" / "worlds" / "models" / "aruco_5x5_250_id1_motion" / "model.sdf"
)
CALIBRATION_ARUCO_MODEL = (
    PACKAGE.parent / "myrobot_simulation" / "worlds" / "models" / "aruco_5x5_250_id1" / "model.sdf"
)
ARUCO_NODE = (
    PACKAGE.parent / "calibration_ws" / "ros2_aruco" / "ros2_aruco" / "ros2_aruco" / "aruco_node.py"
)
SERVO_PARAMETERS = (
    PACKAGE.parent / "myrobot_support_ws" / "fairino_arm_moveit_config" / "config" / "servo_parameters_real.yaml"
)
REAL_CONTROLLERS = (
    PACKAGE.parent
    / "myrobot_support_ws"
    / "fairino_arm_moveit_config"
    / "config"
    / "ros2_controllers_real.yaml"
)
REAL_NO_GRIPPER_CONTROLLERS = (
    PACKAGE.parent
    / "myrobot_support_ws"
    / "fairino3_v6_moveit2_config"
    / "config"
    / "ros2_controllers_real.yaml"
)
SIM_CONTROLLERS = (
    PACKAGE.parent
    / "myrobot_support_ws"
    / "fairino_arm_moveit_config"
    / "config"
    / "ros2_controllers_sim.yaml"
)
SIM_NO_GRIPPER_CONTROLLERS = (
    PACKAGE.parent
    / "myrobot_support_ws"
    / "fairino3_v6_moveit2_config"
    / "config"
    / "ros2_controllers_sim.yaml"
)
HARDWARE_MOVEIT_LAUNCH = (
    PACKAGE.parent / "myrobot_support_ws" / "fairino_arm_moveit_config" / "launch"
    / "moveit_hardware.launch.py"
)
SHARED_RVIZ = PACKAGE / "rviz" / "visual_position_servo.rviz"
IMAGE_RVIZ = PACKAGE / "rviz" / "visual_image_servo.rviz"
REMOVED_SIM_RVIZ = PACKAGE.parent / "myrobot_simulation" / "rviz" / "visual_position_servo_sim.rviz"


def test_business_yaml_moveit_layers_share_the_same_client_contract():
    configs = (
        (PACKAGE.parent / "llm_arm_control" / "config" / "llm_robot_control_params.yaml", "llm_control_task_server", True),
        (PACKAGE.parent / "graspnet_ws" / "graspnet_bringup" / "config" / "graspnet_grasping_params.yaml", "graspnet_visual_grasping", True),
        (PACKAGE.parent / "visual_grasping_bringup" / "config" / "visual_grasping_params.yaml", "visual_grasping", True),
        (PACKAGE / "config" / "visual_position_servo_params.yaml", "visual_position_servo", True),
        (PACKAGE / "config" / "visual_image_servo_params.yaml", "visual_image_servo", False),
    )
    for path, node, needs_discrete_planner in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        parameters = config["common"]["nodes"][node].get(
            "ros__parameters", config["common"]["nodes"][node]
        )
        moveit = parameters["moveit"]
        assert set(moveit["move_groups"]) == {"fairino", "kdl"}
        assert moveit["ik"]["default"] in {"fairino", "kdl"}
        if needs_discrete_planner:
            assert {"pipeline", "planner", "readiness"} <= set(moveit)
        else:
            assert "pipeline" not in moveit and "planner" not in moveit


def test_business_yaml_environment_loader_merges_common_and_selected_launch_defaults():
    configs = (
        ("llm_arm_control", "config/llm_robot_control_params.yaml", "llm_control_task_server"),
        ("graspnet_bringup", "config/graspnet_grasping_params.yaml", "graspnet_visual_grasping"),
        ("visual_grasping_bringup", "config/visual_grasping_params.yaml", "visual_grasping"),
    )
    for package, relative_path, node in configs:
        for environment in ("real", "sim"):
            assert load_launch_parameters_yaml(package, relative_path, environment)
            assert load_node_parameters_yaml(package, relative_path, node, environment)
            assert load_moveit_parameters_yaml(package, relative_path, node, environment)


def test_hardware_moveit_executes_only_the_yaml_selected_ik_client():
    source = HARDWARE_MOVEIT_LAUNCH.read_text(encoding="utf-8")

    assert 'LaunchConfiguration("execution_ik")' in source
    assert 'LaunchConfiguration("execution_pipeline")' in source
    assert 'DeclareLaunchArgument(\n            "execution_ik"' in source
    assert 'DeclareLaunchArgument(\n            "execution_pipeline"' in source
    assert "_planning_only_parameters" in source


def test_hardware_position_servo_uses_the_visual_grasping_hardware_contract():
    source = HARDWARE_LAUNCH.read_text(encoding="utf-8")

    for text in (
        'camera_launch(',
        '"realsense",',
        '"color_profile": "640x480x60"',
        '"depth_profile": "640x480x60"',
        '"rgb_camera.color_profile": value(context, "color_profile")',
        '"depth_module.depth_profile": value(context, "depth_profile")',
        '"align_depth.enable": "true"',
        '"moveit_hardware.launch.py"',
        '"handeye_publisher.py"',
        '"calibration_name": "robot_calibration"',
        '"yolo_kalman_detector_obb.py"',
        'executable="visual_position_servo"',
    ):
        assert text in source
    assert '"demo.launch.py"' not in source
    assert "dual_move_group_nodes" not in source


def test_hardware_position_servo_centralizes_runtime_launch_arguments():
    source = HARDWARE_LAUNCH.read_text(encoding="utf-8")
    defaults = source.split("DEFAULTS = {", 1)[1].split("DESCRIPTIONS = {", 1)[0]

    for name in (
        "use_sim_time",
        "camera_serial_no",
        "color_profile",
        "depth_profile",
        "pointcloud_enable",
        "use_rviz",
        "rviz_config",
        "debug",
        "allow_trajectory_execution",
        "publish_monitored_planning_scene",
        "monitor_dynamics",
        "capabilities",
        "disable_capabilities",
        "publish_frequency",
    ):
        assert f'"{name}":' in defaults

    assert "camera_type" not in defaults
    assert "active_executor" not in defaults
    assert 'moveit_args["execution_ik"] = visual_servo_params["ik_plugin"]' in source
    assert "*(_argument(name, default) for name, default in DEFAULTS.items())" in source
    assert "OpaqueFunction(function=_launch_setup)" in source
    assert "name: value(context, name)" in source
    assert '"open_gripper_after_home"' in source
    assert "visual_servo_params.update(_public_node_parameters(context))" in source


def test_position_servo_launches_import_aruco_parameters_from_yaml_only():
    for launch in (HARDWARE_LAUNCH, SIM_LAUNCH):
        source = launch.read_text(encoding="utf-8")
        assert "aruco_parameters" in source
        assert "aruco_detector_parameters" in source
        assert "aruco_pose_source_parameters" in source
        assert "_ARUCO_LAUNCH_FALLBACKS" not in source
        assert "_aruco_launch_parameters" not in source

    source = ARUCO_NODE.read_text(encoding="utf-8")
    for parameter in (
        "adaptive_thresh_win_size_min",
        "min_marker_perimeter_rate",
        "corner_refinement_method",
        "corner_refinement_min_accuracy",
    ):
        assert f'"{parameter}"' in source
    assert "self._configure_detector_parameters()" in source

    aruco = aruco_parameters()
    assert aruco_detector_parameters(aruco)["marker_size"] == aruco["aruco_marker_size_m"]
    assert aruco_pose_source_parameters(aruco)["marker_id"] == aruco["aruco_marker_id"]


def test_position_servo_open_gripper_is_shared_and_launch_overridable():
    source = SIM_LAUNCH.read_text(encoding="utf-8")
    config = yaml.safe_load((PACKAGE / "config" / "visual_position_servo_params.yaml").read_text(encoding="utf-8"))

    assert "open_gripper_after_home" not in config["environments"]["sim"]["launch"]
    assert config["common"]["nodes"]["visual_position_servo"]["task"]["open_gripper_after_home"] is False
    assert '"open_gripper_after_home"' in source
    assert "sim_open_gripper_after_home" not in source
    assert "_SIM_OPEN_GRIPPER_AFTER_HOME" not in source
    assert '"open_gripper_after_home"' in source
    assert "_PUBLIC_NODE_PARAMETER_NAMES" in source
    assert '"robot_profile", "fairino_arm_gripper_onbase"' in source
    assert '"fairino_arm_gripper_inhand"' in source
    assert '"fairino3_v6"' in source
    assert 'target_motion_controller_node.py' in source
    assert 'aruco_marker_pose_publisher.py' in source


def test_sim_aruco_target_is_predefined_in_the_world_not_spawned_by_launch():
    launch_source = SIM_LAUNCH.read_text(encoding="utf-8")
    world_source = ROBOTARM_WORLD.read_text(encoding="utf-8")
    model_source = ARUCO_MODEL.read_text(encoding="utf-8")

    assert 'aruco_5x5_250_id1_dynamic' not in launch_source
    assert "ros_gz_sim', executable='create'" not in launch_source
    assert '<uri>model://aruco_5x5_250_id1_motion</uri>' in world_source
    assert '<name>aruco_marker_model</name>' in world_source
    assert re.search(
        r"<pose>0\.2 0\.45 [-+]?\d+(?:\.\d+)? 0\.0 0\.0 0\.0</pose>",
        world_source,
    )
    assert '<static>false</static>' in model_source
    assert '<self_collide>false</self_collide>' in model_source
    assert '<pose>0 0 0 1.5708 0 0</pose>' in model_source
    assert '<gravity>false</gravity>' in model_source
    assert '<allow_auto_disable>false</allow_auto_disable>' in model_source
    assert '<mu>0.08</mu>' in model_source
    assert 'ignition::gazebo::systems::VelocityControl' in model_source
    assert '<topic>/model/aruco_marker_model/cmd_vel</topic>' in model_source

    calibration_model_source = CALIBRATION_ARUCO_MODEL.read_text(encoding="utf-8")
    assert '<pose>0 0 0 0 0 0</pose>' in calibration_model_source


def test_position_servo_yaml_has_exclusive_aruco_source_and_target_rpy():
    config = yaml.safe_load((PACKAGE / "config" / "visual_position_servo_params.yaml").read_text(encoding="utf-8"))
    assert list(config)[:2] == ["environments", "common"]
    node = config["common"]["nodes"]["visual_position_servo"]
    yolo = node["perception"]["yolo_kalman"]
    aruco = node["perception"]["aruco"]

    assert node["perception"]["active_source"] in {"yolo_kalman", "aruco"}
    assert yolo["backend"] == "tensorrt"
    assert yolo["engine_path"] == "yolo-obb-640.engine"
    assert aruco["detection"]["marker_size_m"] == 0.07
    assert aruco["detection"]["dictionary"] == "DICT_5X5_250"
    assert aruco["detection"]["detector"]["corner_refinement_method"] == "none"
    assert aruco["position_source"] == {
        "marker_id": 1,
        "input_topic": "/aruco_markers",
        "output_topic": "/aruco_marker/pose",
    }
    assert aruco["visualization"]["image_topic"] == "/aruco_marker/visualization"
    assert aruco["visualization"]["marker_id"] == 1
    prediction_hold_sec = aruco["tracking"]["prediction_hold_sec"]
    for environment in ("sim", "real"):
        parameters = visual_servo_parameters(environment)
        assert parameters["aruco_marker_id"] == 1
        assert parameters["aruco_prediction_hold_sec"] == prediction_hold_sec
        assert parameters["aruco_corner_refinement_method"] == "none"
        assert not any(isinstance(value, dict) for value in parameters.values())
    assert config["environments"]["sim"]["nodes"]["visual_position_servo"]["planning"]["target_above_rpy_deg"] == [-45.0, -180.0, 0.0]
    assert config["environments"]["real"]["nodes"]["visual_position_servo"]["planning"]["target_above_rpy_deg"] == [0.0, -180.0, 100.0]


def test_position_servo_environment_parameters_are_explicit_and_flattened():
    sim = visual_servo_parameters("sim")
    real = visual_servo_parameters("real")

    assert sim["position_servo_rate_hz"] == 250.0
    assert real["position_servo_rate_hz"] == 250.0
    assert sim["above_offset"] == 0.22
    assert real["above_offset"] == 0.20
    assert sim["home_pose.xyz"] == [-0.2, 0.25, 0.25]
    assert real["home_pose.xyz"] == [0.1, 0.3, 0.25]
    assert "common" not in sim and "environments" not in sim
    with pytest.raises(ValueError):
        visual_servo_parameters("invalid")


def test_sim_target_motion_comes_from_one_complete_yaml_section():
    required = {
        "model_name", "cmd_topic", "cmd_internal_topic", "auto_start_topic",
        "trajectory_type", "linear_x", "linear_y", "angular_z",
        "rect_length_x", "rect_length_y", "rect_speed", "max_motion_time",
        "auto_start", "startup_hold_sec", "publish_rate",
    }
    cube = sim_target_motion_parameters("yolo_kalman")
    aruco = sim_target_motion_parameters("aruco")
    assert required <= set(cube) == set(aruco)
    assert (cube["model_name"], cube["linear_x"], cube["angular_z"]) == ("cube_model", 0.05, 0.5)
    assert (aruco["model_name"], aruco["linear_x"], aruco["angular_z"]) == ("aruco_marker_model", 0.05, 0.5)
    controller_source = (PACKAGE.parent / "myrobot_simulation" / "scripts" / "target_motion_controller_node.py").read_text(encoding="utf-8")
    assert "cmd_msg.linear.z = 0.0" in controller_source
    assert "cmd_msg.angular.x = 0.0" in controller_source
    assert "cmd_msg.angular.y = 0.0" in controller_source
    with pytest.raises(ValueError):
        sim_target_motion_parameters("invalid")


def test_real_arm_controller_accepts_continuous_moveit_servo_trajectories():
    controllers = yaml.safe_load(REAL_CONTROLLERS.read_text(encoding="utf-8"))

    assert controllers["robot_arm_controller"]["ros__parameters"][
        "allow_nonzero_velocity_at_trajectory_end"
    ] is True
    assert controllers["hand_controller"]["ros__parameters"][
        "allow_nonzero_velocity_at_trajectory_end"
    ] is False


def test_real_jtc_samples_each_moveit_servo_window_before_replacement():
    for path in (REAL_CONTROLLERS, REAL_NO_GRIPPER_CONTROLLERS):
        controllers = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert controllers["controller_manager"]["ros__parameters"]["update_rate"] == 500

    for path in (SIM_CONTROLLERS, SIM_NO_GRIPPER_CONTROLLERS):
        controllers = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert controllers["controller_manager"]["ros__parameters"]["update_rate"] == 500

    servo = yaml.safe_load(SERVO_PARAMETERS.read_text(encoding="utf-8"))
    assert servo["publish_period"] == pytest.approx(0.004)


def test_real_arm_controller_and_xacro_expose_actual_joint_velocity():
    xacro_paths = (
        PACKAGE.parent
        / "myrobot_support_ws"
        / "fairino_arm_moveit_config"
        / "config"
        / "fairino_arm_moveit_descriptions.ros2_control_real.xacro",
        PACKAGE.parent
        / "myrobot_support_ws"
        / "fairino3_v6_moveit2_config"
        / "config"
        / "fairino3_v6_robot.ros2_control_real.xacro",
    )
    for path in xacro_paths:
        source = path.read_text(encoding="utf-8")
        assert source.count('\n                <state_interface name="velocity"/>') == 6

    controllers = yaml.safe_load(REAL_CONTROLLERS.read_text(encoding="utf-8"))
    assert controllers["robot_arm_controller"]["ros__parameters"]["state_interfaces"] == [
        "position",
        "velocity",
    ]


def test_aruco_overlay_reuses_the_detection_and_rviz_displays_it():
    source = ARUCO_NODE.read_text(encoding="utf-8")
    rviz = SHARED_RVIZ.read_text(encoding="utf-8")

    assert source.count("cv2.aruco.detectMarkers(") == 1
    assert 'value="/aruco_marker/visualization"' in source
    assert 'name="visualization_marker_id"' in source
    assert "self.visualization_pub.get_subscription_count() == 0" in source
    assert "cv2.circle(image, center, 4, (0, 255, 0), -1)" in source
    assert "output.header = img_msg.header" in source
    assert "Name: Aruco Center Overlay" in rviz
    assert "Value: /aruco_marker/visualization" in rviz
    assert "Reliability Policy: Best Effort" in rviz


def test_position_servo_uses_the_visual_position_node_instance_name():
    for path in (POSITION_SERVO_NODE, HARDWARE_LAUNCH, SIM_LAUNCH):
        source = path.read_text(encoding="utf-8")
        assert "visual_position_servo_node" in source
    deprecated_name = "visual_servo" + "_grasping"
    assert deprecated_name not in source


def test_position_servo_entries_share_one_rviz_file():
    source = SIM_LAUNCH.read_text(encoding="utf-8")

    assert SHARED_RVIZ.is_file()
    assert not REMOVED_SIM_RVIZ.exists()
    assert 'get_package_share_directory("visual_servo_bringup")' in source
    assert '"visual_position_servo.rviz"' in source
    assert "vision_velocity" + "_evaluator" not in source


def test_position_servo_uses_environment_specific_6d_home_pose():
    source = POSITION_SERVO_NODE.read_text(encoding="utf-8")
    config = yaml.safe_load((PACKAGE / "config" / "visual_position_servo_params.yaml").read_text(encoding="utf-8"))
    environments = config["environments"]

    assert environments["sim"]["nodes"]["visual_position_servo"]["task"]["home_pose"] == {
        "xyz": [-0.2, 0.25, 0.25],
        "rpy_deg": [0.0, -180.0, 0.0],
    }
    assert environments["real"]["nodes"]["visual_position_servo"]["task"]["home_pose"] == {
        "xyz": [0.1, 0.3, 0.25],
        "rpy_deg": [0.0, -180.0, 100.0],
    }
    assert "home_joints" not in environments["sim"]["nodes"]["visual_position_servo"]["task"]
    assert "home_joints" not in environments["real"]["nodes"]["visual_position_servo"]["task"]
    assert 'pose_values("home_pose.xyz", [0.0, 0.0, 0.0])' in source
    assert 'pose_values("home_pose.rpy_deg", [0.0, 0.0, 0.0])' in source
    assert "self.home_pose = self.pose_tools.make_pose(*home_xyz, *home_rpy_deg)" in source
    assert "self.motion.move_to_pose(\n                self.home_pose," in source
    assert "self.motion.move_to_joints(\n                self.home_joints," not in source
    assert 'param_b(self, "open_gripper_after_home", False)' in source


def test_robotarm_world_is_valid_xml_with_manual_target_switches_available():
    root = ElementTree.parse(ROBOTARM_WORLD).getroot()
    active_models = {
        include.findtext("uri")
        for include in root.findall(".//include")
    }
    world = ROBOTARM_WORLD.read_text(encoding="utf-8")
    assert "model://cube" not in active_models
    assert "model://aruco_5x5_250_id1_motion" in active_models
    assert "model://aruco_5x5_250_id1_motion" in world


def test_image_servo_sim_entry_is_removed():
    assert not (
        PACKAGE.parent
        / "myrobot_simulation"
        / "launch"
        / ("visual_image_servo" + "_sim.launch.py")
    ).exists()


def test_image_servo_hardware_launch_uses_realsense_handeye_and_moveit_servo():
    source = IMAGE_HARDWARE_LAUNCH.read_text(encoding="utf-8")

    for text in (
        'camera_launch(',
        '"color_profile": "640x480x60"',
        '"depth_profile": "640x480x60"',
        '"align_depth.enable": "true"',
        '"moveit_hardware.launch.py"',
        'executable="servo_node_main"',
        'executable="handeye_publisher.py"',
        'executable="visual_image_servo"',
        '"use_gazebo": False',
        '"visual_image_servo.rviz"',
        'def _node_parameter(context, name):',
        'image_params = {name: _node_parameter(context, name) for name in _IMAGE_SERVO_DEFAULTS}',
        '*(_argument(name, default) for name, default in DEFAULTS.items())',
    ):
        assert text in source
    assert "follow_aruco_marker" not in source
    assert "enable_validation_follower" not in source
    assert '"config/servo_parameters_real.yaml"' in source
    config = (PACKAGE / "config" / "visual_image_servo_params.yaml").read_text(encoding="utf-8")
    assert "      auto_start: true" in config
    assert "profiles:" not in config


def test_position_servo_launches_keep_literal_fallbacks_before_yaml_defaults():
    for source_path in (HARDWARE_LAUNCH, SIM_LAUNCH):
        source = source_path.read_text(encoding="utf-8")
        assert "DEFAULTS = {" in source or "_LAUNCH_ARGUMENT_SPECS" in source
        assert "load_launch_parameters_yaml" in source
        assert "LaunchConfiguration" in source


def test_image_servo_launch_keeps_literal_node_fallbacks_before_yaml_defaults():
    source = IMAGE_HARDWARE_LAUNCH.read_text(encoding="utf-8")
    assert "_NODE_PARAMETER_FALLBACKS" in source
    assert '"lambda_gain": 0.9' in source
    assert "**image_servo_parameters()" in source


def test_image_servo_rviz_is_a_copy_with_visible_interactive_markers():
    position_config = SHARED_RVIZ.read_text(encoding="utf-8")
    image_config = IMAGE_RVIZ.read_text(encoding="utf-8")

    assert IMAGE_RVIZ.is_file()
    assert "Interactive Marker Size: 0.25" in position_config
    assert "Interactive Marker Size: 0.25" in image_config
    assert "Move Group Namespace: /move_group_fairino" in position_config
    assert "Move Group Namespace: /move_group_fairino" in image_config
    assert "Planning Scene Topic: /move_group_fairino/monitored_planning_scene" in position_config
    assert "Planning Scene Topic: /move_group_fairino/monitored_planning_scene" in image_config
