from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "myrobot_simulation" / "launch" / "llm_robot_control_sim.launch.py"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "llm_robot_control_params.yaml"
GRASPNET_CONFIG = ROOT / "graspnet_ws" / "graspnet_bringup" / "config" / "graspnet_grasping_params.yaml"
YOLO_CONFIG = ROOT / "visual_grasping_bringup" / "config" / "visual_grasping_params.yaml"
HARDWARE_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "llm_robot_control.launch.py"


def _effective_params(config, node_name):
    nodes = config.get("common", {}).get("nodes", config)
    shared = nodes.get("/**", {}).get("ros__parameters", {})
    node = nodes[node_name]["ros__parameters"]
    return {**shared, **node}


def test_llm_robot_launch_declares_and_forwards_only_its_robot_profile():
    source = LAUNCH.read_text(encoding="utf-8")

    assert '"robot_profile", "fairino_arm_gripper_inhand"' in source
    assert "_YAML_LAUNCH_DEFAULTS" in source
    assert "_declare_launch_arguments" in source
    assert '"robot_profile": _LAUNCH_CONFIGURATIONS["robot_profile"]' in source
    assert "fairino_arm_gripper_calibration_onbase" in source
    assert "fairino_arm_gripper_inhand" in source


def test_shared_llm_robot_config_has_pregrasp_and_no_fixed_sim_time():
    source = CONFIG.read_text(encoding="utf-8")

    config = yaml.safe_load(source)
    assert "use_sim_time" not in config["common"]["nodes"]["llm_control_task_server"]["ros__parameters"]
    for line in (
        "pregrasp_pose.x: 0.1",
        "pregrasp_pose.y: 0.35",
        "pregrasp_pose.z: 0.3",
        "pregrasp_pose.roll: 0.0",
        "pregrasp_pose.pitch: -180.0",
        "pregrasp_pose.yaw: 100.0",
    ):
        assert line in source
    assert not (CONFIG.parent / "llm_yolo_task_sim.yaml").exists()
    assert not (CONFIG.parent / "llm_yolo_task_hardware.yaml").exists()


def test_llm_launch_starts_cli_and_new_perception_without_motion_control_node():
    source = LAUNCH.read_text(encoding="utf-8")

    assert '"llm_visual_perception.launch.py"' in source
    assert "ExecuteProcess" in source
    assert '"gnome-terminal"' in source
    assert '"llm_control_cli"' in source
    assert '["use_sim_time:=", _LAUNCH_CONFIGURATIONS["use_sim_time"]]' in source
    assert '["command_burst_count:=", str(_YAML_LAUNCH_DEFAULTS["command_burst_count"])]' in source
    assert 'emulate_tty=True' not in source
    assert 'executable="llm_control_cli"' not in source
    assert 'executable="robot_pose_monitor_node"' in source
    assert 'executable="llm_control_task_server"' in source
    old_monitor = "fairino" + "_pose_monitor"
    old_server = "llm" + "_yolo_task_server"
    assert f'executable="{old_monitor}"' not in source
    assert f'executable="{old_server}"' not in source
    assert 'package="myrobot_common"' not in source
    assert 'executable="motion_control"' not in source


def test_sim_llm_launch_serializes_internal_yolo_boolean_for_its_child_launch():
    source = LAUNCH.read_text(encoding="utf-8")

    assert '"use_continuous_yolo": launch_defaults_as_strings(' in source
    assert '_PERCEPTION_PARAMETERS\n                    )["use_continuous_yolo"]' in source


def test_hardware_llm_launch_delays_yolo_task_server_and_cli():
    source = HARDWARE_LAUNCH.read_text(encoding="utf-8")

    assert source.count("period=8.0") == 4
    assert "yolo_obb = TimerAction(" in source
    assert "monitor = TimerAction(" in source
    assert "task = TimerAction(" in source
    assert "cli = TimerAction(" in source
    assert "task_server = TimerAction(" not in source
    assert "cli_terminal = TimerAction(" not in source
    assert 'llm_visual_perception.launch.py' in source
    assert '"use_continuous_yolo": launch_defaults_as_strings(' in source
    assert '_LLM_PERCEPTION_PARAMETERS\n                    )["use_continuous_yolo"]' in source
    assert 'executable="llm_control_task_server"' in source
    assert '"llm_control_cli"' in source


def test_llm_launch_uses_the_rviz_file_installed_by_llm_arm_control():
    source = LAUNCH.read_text(encoding="utf-8")

    assert 'llm_arm_share = get_package_share_directory("llm_arm_control")' in source
    assert 'os.path.join(llm_arm_share, "rviz", "llm_robot_control.rviz")' in source


def test_shared_config_uses_new_yolo_topics_and_grasp_profile():
    source = CONFIG.read_text(encoding="utf-8")

    assert "yolo_topic: /yolo/detected_result" in source
    assert "depth_topic: /yolo/detected_result/depth" in source
    assert "descend_to_box: 0.04" in source
    assert "grasp.stone.yaw_offset: -45.0" in source
    assert "arm_max_velocity: 0.2" in source
    assert "arm_max_acceleration: 0.2" in source
    assert "common:\n  launch: {}\n  nodes:\n    llm_control_task_server:" in source
    assert "    llm_visual_perception:" in source
    assert "graspnet_visual_grasping:" not in source
    assert "graspnet_inference:" in source


def test_llm_graspnet_inference_filters_match_standalone_config():
    llm = _effective_params(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8")), "graspnet_inference"
    )
    standalone = _effective_params(
        yaml.safe_load(GRASPNET_CONFIG.read_text(encoding="utf-8")),
        "graspnet_inference",
    )
    common = (
        "rgb_topic",
        "depth_topic",
        "camera_info_topic",
        "poses_topic",
        "scores_topic",
        "metadata_topic",
        "preview_best_pose_topic",
        "preview_best_score_topic",
        "num_point",
        "top_k_publish",
        "min_valid_points",
        "base_frame",
        "depth_min_m",
        "depth_max_m",
        "sync_queue_size",
        "sync_slop_s",
        "random_seed",
        "min_grasp_width_m",
        "max_grasp_width_m",
        "support_plane_distance_m",
        "support_plane_min_inlier_ratio",
        "support_plane_max_tilt_deg",
        "workspace_x_min_m",
        "workspace_x_max_m",
        "workspace_y_min_m",
        "workspace_y_max_m",
        "object_min_height_m",
        "object_max_height_m",
        "collision_detection_enabled",
        "collision_voxel_size_m",
        "collision_approach_distance_m",
        "collision_threshold",
        "confirm_before_publish",
        "confirm_visual_top_k",
        "confirm_window_name",
    )

    assert {key: llm[key] for key in common} == {key: standalone[key] for key in common}
    assert llm["object_min_height_m"] == 0.0
    assert llm["min_grasp_width_m"] == 0.0


def test_llm_task_keeps_graspnet_and_yolo_parameters_separate():
    task = _effective_params(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8")), "llm_control_task_server"
    )
    standalone = _effective_params(
        yaml.safe_load(GRASPNET_CONFIG.read_text(encoding="utf-8")),
        "graspnet_visual_grasping",
    )

    assert all(key.startswith("graspnet_") for key in task if "graspnet" in key)
    assert task["graspnet_max_candidates"] == standalone["max_grasp_candidates"]
    assert task["graspnet_approach_distance_m"] == standalone["approach_distance_m"]
    assert task["graspnet_grasp_offset_m"] == standalone["grasp_offset_m"]
    assert task["graspnet_lift_distance"] == standalone["lift_distance"]
    assert task["graspnet_to_ee_rpy_deg"] == standalone["graspnet_to_ee_rpy_deg"]
    assert task["graspnet_use_width"] == standalone["use_graspnet_width"]
    assert task["grasp_above"] == 0.04
    assert task["grasp_offset"] == 0.013
    assert task["place_offset"] == 0.08
    assert task["descend_to_box"] == 0.04
    assert "min_grasp_width_m" not in task
    assert "object_min_height_m" not in task


def test_layered_visual_configs_and_node_boundaries():
    graspnet = yaml.safe_load(GRASPNET_CONFIG.read_text(encoding="utf-8"))
    yolo = yaml.safe_load(YOLO_CONFIG.read_text(encoding="utf-8"))

    graspnet_nodes = graspnet["common"]["nodes"]
    yolo_nodes = yolo["common"]["nodes"]
    assert {"/**", "graspnet_visual_grasping", "graspnet_inference"} <= set(graspnet_nodes)
    assert {"visual_grasping", "yolo_detector_obb"} <= set(yolo_nodes)

    visual = yolo_nodes["visual_grasping"]["ros__parameters"]
    detector = yolo_nodes["yolo_detector_obb"]["ros__parameters"]
    assert "grasp_above" in visual
    assert "depth_inlier_m" in detector
    assert "grasp_above" not in detector
    assert "depth_inlier_m" not in visual


def test_graspnet_launches_load_static_inference_from_yaml_only():
    launches = (
        ROOT / "graspnet_ws" / "graspnet_bringup" / "launch" / "graspnet_grasping.launch.py",
        ROOT / "myrobot_simulation" / "launch" / "graspnet_grasping_sim.launch.py",
    )
    forbidden = (
        "-p rgb_topic:=",
        "-p depth_topic:=",
        "-p camera_info_topic:=",
        "-p num_point:=",
        "-p top_k_publish:=",
        "-p min_valid_points:=",
        "-p confirm_before_publish:=",
        "-p confirm_visual_top_k:=",
    )
    for launch in launches:
        source = launch.read_text(encoding="utf-8")
        assert "graspnet_grasping_params.yaml" in source
        assert "baseline_dir" in source
        assert "checkpoint_path" in source
        assert all(item not in source for item in forbidden)


def test_yolo_launches_load_layered_config_for_visual_and_detector():
    launches = (
        ROOT / "visual_grasping_bringup" / "launch" / "visual_grasping.launch.py",
        ROOT / "myrobot_simulation" / "launch" / "visual_grasping_sim.launch.py",
    )
    for launch in launches:
        source = launch.read_text(encoding="utf-8")
        assert source.count("visual_grasping_params.yaml") >= 2
        assert 'name="visual_grasping"' in source
        assert 'name="yolo_detector_obb"' in source


def test_business_launches_expose_only_environment_and_planning_overrides():
    launches = (
        LAUNCH,
        HARDWARE_LAUNCH,
        ROOT / "graspnet_ws" / "graspnet_bringup" / "launch" / "graspnet_grasping.launch.py",
        ROOT / "myrobot_simulation" / "launch" / "graspnet_grasping_sim.launch.py",
        ROOT / "visual_grasping_bringup" / "launch" / "visual_grasping.launch.py",
        ROOT / "myrobot_simulation" / "launch" / "visual_grasping_sim.launch.py",
    )
    retired_public_names = (
        '"graspnet_model_profile",', '"model_profile",', '"command_burst_count",',
        '"use_continuous_yolo",', '"imgsz",', '"conf",', '"device",',
    )
    for launch in launches:
        source = launch.read_text(encoding="utf-8")
        declarations = source.split("def _declare_launch_arguments", 1)[0]
        if "def _declare_launch_arguments" not in source:
            declarations = source.split("def _argument", 1)[0]
        assert all(name not in declarations for name in retired_public_names)
        assert "_PUBLIC_TASK_PARAMETER_NAMES" in source


def test_dual_mode_launches_share_the_llm_config_without_motion_control_node():
    for launch in (LAUNCH, HARDWARE_LAUNCH):
        source = launch.read_text(encoding="utf-8")
        assert "graspnet_visual_grasping" not in source
        assert "graspnet_inference" in source
        assert "llm_control_cli" in source
        assert "motion_control" not in source


def test_control_nodes_use_the_new_ros_interface_prefix():
    node_root = ROOT / "llm_arm_control" / "llm_arm_control_nodes"
    for name in ("robot_pose_control_server.py", "robot_pose_monitor_node.py", "llm_control_cli.py", "llm_control_task_server.py"):
        assert (node_root / name).exists()
    assert not (node_root / "entry_point.py").exists()
    assert "/llm_control/" in (node_root / "robot_pose_control_server.py").read_text(encoding="utf-8")
    assert "/llm_control/" in (node_root / "robot_pose_monitor_node.py").read_text(encoding="utf-8")
    assert "/llm_control/" in (node_root / "llm_control_cli.py").read_text(encoding="utf-8")
    assert "/llm_control/" in (node_root / "llm_control_task_server.py").read_text(encoding="utf-8")


def test_sim_llm_launch_and_resources_are_present():
    root = ROOT
    old_config = "llm_yolo" + "_task.yaml"
    old_rviz = "llm_yolo" + ".rviz"
    assert (root / "myrobot_simulation" / "launch" / "llm_robot_control_sim.launch.py").exists()
    assert not (CONFIG.parent / old_config).exists()
    assert not (CONFIG.parent.parent / "rviz" / old_rviz).exists()
