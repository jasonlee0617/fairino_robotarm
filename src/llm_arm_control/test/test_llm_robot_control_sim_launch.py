from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "myrobot_simulation" / "launch" / "llm_robot_control_sim.launch.py"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "llm_robot_control_params.yaml"
YOLO_CONFIG = ROOT / "visual_grasping_bringup" / "config" / "visual_grasping_params.yaml"
HARDWARE_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "llm_robot_control.launch.py"


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
    task_params = config["common"]["nodes"]["llm_control_task_server"][
        "ros__parameters"
    ]
    assert "use_sim_time" not in task_params
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


def test_llm_launch_starts_voice_perception_and_one_motion_control_node():
    source = LAUNCH.read_text(encoding="utf-8")
    voice_launch = (
        ROOT / "llm_arm_control" / "llm_arm_control_nodes" / "voice_launch.py"
    ).read_text(encoding="utf-8")
    cmake = (ROOT / "llm_arm_control" / "CMakeLists.txt").read_text(encoding="utf-8")

    assert '"llm_visual_perception.launch.py"' in source
    assert "ExecuteProcess" not in source
    assert '"gnome-terminal"' not in source
    assert '"llm_control_cli"' not in source
    for executable in ("voice_wake_node", "voice_realtime_node"):
        assert f'executable="{executable}"' in voice_launch
        assert f"RENAME {executable}" in cmake
    assert 'executable="robot_pose_monitor_node"' not in source
    assert 'executable="llm_control_task_server"' in source
    assert '"audio_input_device", "auto"' in source
    assert '"audio_input_volume_percent", "100"' in source
    assert "audio_output_device" not in source
    old_monitor = "fairino" + "_pose_monitor"
    old_server = "llm" + "_yolo_task_server"
    assert f'executable="{old_monitor}"' not in source
    assert f'executable="{old_server}"' not in source
    assert source.count('executable="motion_control"') == 1
    assert '"enable_grasp_trigger_key": False' in source
    assert '"enable_servo"' not in source


def test_sim_llm_launch_forwards_continuous_visual_health_parameters():
    source = LAUNCH.read_text(encoding="utf-8")

    assert "TimerAction" not in source
    assert "yolo_obb = IncludeLaunchDescription(" in source
    assert "use_continuous_yolo" not in source
    for key in ("sync_slop", "sync_watchdog_sec", "expected_camera_rate_hz"):
        assert f')["{key}"]' in source


def test_hardware_llm_launch_removes_cli_and_keeps_local_motion_control():
    source = HARDWARE_LAUNCH.read_text(encoding="utf-8")
    retired_model_override = "voice_model" + "_root"

    assert "TimerAction" not in source
    assert "yolo_obb = IncludeLaunchDescription(" in source
    assert "task = Node(" in source
    assert "motion_control = Node(" in source
    assert "task_server = TimerAction(" not in source
    assert 'llm_visual_perception.launch.py' in source
    assert "use_continuous_yolo" not in source
    for key in ("sync_slop", "sync_watchdog_sec", "expected_camera_rate_hz"):
        assert f')["{key}"]' in source
    assert 'executable="llm_control_task_server"' in source
    assert '"llm_control_cli"' not in source
    assert 'executable="motion_control"' in source
    assert 'environment="real"' in source
    assert 'start_delay_sec=10.0' not in source
    assert '"enable_voice": "true"' in source
    assert '"audio_input_device": "auto"' in source
    assert '"audio_input_volume_percent": "100"' in source
    assert "audio_output_device" not in source
    assert retired_model_override not in source


def test_llm_launch_uses_the_rviz_file_installed_by_llm_arm_control():
    source = LAUNCH.read_text(encoding="utf-8")

    assert 'llm_arm_share = get_package_share_directory("llm_arm_control")' in source
    assert 'os.path.join(llm_arm_share, "rviz", "llm_robot_control.rviz")' in source


def test_shared_config_is_yolo_only():
    source = CONFIG.read_text(encoding="utf-8")
    config = yaml.safe_load(source)

    assert "yolo_topic: /yolo/detected_result" in source
    assert "depth_topic: /yolo/detected_result/depth" in source
    assert "descend_to_box: 0.04" in source
    assert "grasp.stone.yaw_offset: -45.0" in source
    assert "arm_max_velocity: 0.2" in source
    assert "arm_max_acceleration: 0.2" in source
    assert "vision_wait_timeout_sec: 3.5" in source
    shared_launch = config["common"]["launch"]
    assert shared_launch == {
        "enable_voice": True,
        "audio_input_device": "auto",
        "audio_input_volume_percent": 100,
    }
    for environment in ("real", "sim"):
        launch = config["environments"][environment]["launch"]
        assert not (set(shared_launch) & set(launch))
    assert "    llm_visual_perception:" in source
    assert "require_cuda_for_visual_perception: true" in source
    assert "sync_slop: 0.05" in source
    assert "vad_silence_ms: 700" in source
    assert "realtime_provider: qwen" in source
    assert "response_transition_timeout_sec: 5.0" in source
    assert "graspnet" not in source.lower()


def test_layered_visual_configs_and_node_boundaries():
    yolo = yaml.safe_load(YOLO_CONFIG.read_text(encoding="utf-8"))

    yolo_nodes = yolo["common"]["nodes"]
    assert {"visual_grasping", "yolo_detector_obb"} <= set(yolo_nodes)

    visual = yolo_nodes["visual_grasping"]["ros__parameters"]
    detector = yolo_nodes["yolo_detector_obb"]["ros__parameters"]
    assert "grasp_above" in visual
    assert "depth_inlier_m" in detector
    assert "grasp_above" not in detector
    assert "depth_inlier_m" not in visual


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
    )
    retired_public_names = (
        '"model_profile",', '"command_burst_count",',
        '"use_continuous_yolo",', '"imgsz",', '"conf",', '"device",',
    )
    for launch in launches:
        source = launch.read_text(encoding="utf-8")
        declarations = source.split("def _declare_launch_arguments", 1)[0]
        if "def _declare_launch_arguments" not in source:
            declarations = source.split("def _argument", 1)[0]
        assert all(name not in declarations for name in retired_public_names)
        assert "_PUBLIC_TASK_PARAMETER_NAMES" in source


def test_llm_launches_are_yolo_only_and_do_not_reference_cli():
    for launch in (LAUNCH, HARDWARE_LAUNCH):
        source = launch.read_text(encoding="utf-8")
        assert "graspnet" not in source.lower()
        assert "llm_control_cli" not in source
        assert "motion_control" in source


def test_llm_launches_share_fail_fast_voice_builder():
    helper = (
        ROOT / "llm_arm_control" / "llm_arm_control_nodes" / "voice_launch.py"
    ).read_text(encoding="utf-8")
    for launch in (LAUNCH, HARDWARE_LAUNCH):
        assert "build_voice_launch_actions" in launch.read_text(encoding="utf-8")
    assert "validate_voice_runtime" in helper
    assert "OnProcessExit" in helper
    assert "Shutdown" in helper


def test_voice_model_is_installed_and_not_configurable_by_launch():
    package = ROOT / "llm_arm_control"
    model = package / "model" / "kws"
    for name in (
        "tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx",
        "keywords.txt", "keywords_raw.txt", "en.phone",
    ):
        assert (model / name).is_file()
        assert not (model / name).is_symlink()
    cmake = (package / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "DIRECTORY audio config model rviz launch" in cmake
    retired_model_override = "voice_model" + "_root"
    assert retired_model_override not in LAUNCH.read_text(encoding="utf-8")
    assert retired_model_override not in HARDWARE_LAUNCH.read_text(encoding="utf-8")


def test_control_nodes_use_the_new_ros_interface_prefix():
    node_root = ROOT / "llm_arm_control" / "llm_arm_control_nodes"
    for name in (
        "robot_motion_base.py",
        "llm_control_task_server.py", "voice_wake_node.py", "voice_realtime_node.py",
    ):
        assert (node_root / name).exists()
    assert not (node_root / "llm_control_cli.py").exists()
    assert not (node_root / "entry_point.py").exists()
    motion_base = (node_root / "robot_motion_base.py").read_text(encoding="utf-8")
    assert "ControlPose" not in motion_base
    assert "/llm_control/control_pose" not in motion_base
    assert not (node_root / "robot_pose_monitor_node.py").exists()
    task_server = (node_root / "llm_control_task_server.py").read_text(
        encoding="utf-8"
    )
    assert "/llm_control/" in task_server


def test_sim_llm_launch_and_resources_are_present():
    root = ROOT
    old_config = "llm_yolo" + "_task.yaml"
    old_rviz = "llm_yolo" + ".rviz"
    assert (root / "myrobot_simulation" / "launch" / "llm_robot_control_sim.launch.py").exists()
    assert not (CONFIG.parent / old_config).exists()
    assert not (CONFIG.parent.parent / "rviz" / old_rviz).exists()


def test_llm_sim_disables_only_the_unused_native_depth_stream_by_default():
    llm_launch = LAUNCH.read_text(encoding="utf-8")
    gazebo_launch = (
        ROOT / "myrobot_simulation" / "launch" / "gazebo.launch.py"
    ).read_text(encoding="utf-8")

    assert '("native_depth_enabled", "false"' in llm_launch
    assert '"native_depth_enabled": _LAUNCH_CONFIGURATIONS["native_depth_enabled"]' in llm_launch
    assert '("native_depth_enabled", "true"' in gazebo_launch


def test_retime_server_loads_moveit_joint_limits():
    launch = (
        ROOT / "myrobot_common_ws" / "trajectory_retime_server" / "launch"
        / "retime_server.launch.py"
    ).read_text(encoding="utf-8")
    assert '"joint_limits.yaml"' in launch
    assert '"robot_description_planning": robot_description_planning' in launch
