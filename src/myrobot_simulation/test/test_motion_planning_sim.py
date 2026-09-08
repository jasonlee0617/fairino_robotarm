#!/usr/bin/env python3
"""Static contract checks for the merged planning/IK simulation entrypoint."""

import ast
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
NODE_FILE = PACKAGE_DIR / "scripts" / "motion_planning_node_sim.py"
LAUNCH_FILE = PACKAGE_DIR / "launch" / "motion_planning_demo_sim.launch.py"
MOVEIT_STACK_FILE = PACKAGE_DIR / "launch_utils" / "moveit_stack.py"
SIM_STACK_FILE = PACKAGE_DIR / "launch_utils" / "sim_stack.py"
CORE_CONFIG_DIR = PACKAGE_DIR.parent / "myrobot_planning_core" / "config"


def _method_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        item.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MotionPlanningNodeSim"
        for item in node.body
        if isinstance(item, ast.FunctionDef)
    }


def test_merged_node_keeps_planning_and_raw_ik_modes():
    methods = _method_names(NODE_FILE)
    assert {
        "setup_scene",
        "select_mode",
        "run_planning_mode",
        "run_ik_comparison_mode",
        "compare_ik",
        "report_tf_position_error",
    } <= methods
    source = NODE_FILE.read_text(encoding="utf-8")
    assert "request.ik_request.avoid_collisions = False" in source


def test_launch_uses_only_the_merged_executable():
    source = LAUNCH_FILE.read_text(encoding="utf-8")
    assert 'executable="motion_planning_node_sim.py"' in source
    assert "trajectory_plan" + "_node.py" not in source
    assert "run_mode" in source
    assert "goal_collection_dir" in source
    assert "benchmark_output_dir" in source
    assert "benchmark_scene_name" in source
    assert "benchmark_goal_seed" in source
    assert "benchmark_goal_root_mode" in source
    assert '"start_joints"' in source
    assert '"start_id"' in source
    assert "OnProcessExit" in source
    assert "benchmark_execution" in source
    assert "benchmark_algorithm" in source
    assert "goal_collection" in source
    assert '"benchmark_effective_config_json"' in source


def test_planning_launches_keep_cases_in_yaml_and_map_public_moveit_names():
    motion = LAUNCH_FILE.read_text(encoding="utf-8")
    assert "motion_planning_demo_params.yaml" in motion
    assert "planning_client" in motion
    assert '"scene_name"' not in motion.split("_PUBLIC_ARGUMENTS", 1)[1].split("_DEFAULTS", 1)[0]
    assert "benchmark_goal_seed" in motion.split("_PUBLIC_ARGUMENTS", 1)[1].split("_DEFAULTS", 1)[0]
    assert '"moveit_clients"' in motion


def test_legacy_benchmark_entrypoints_are_removed():
    assert not (PACKAGE_DIR / "launch" / ("trajectory_plan" + "_test_sim.launch.py")).exists()
    assert not (PACKAGE_DIR / "scripts" / ("trajectory_plan" + "_test_node_sim.py")).exists()
    assert not (PACKAGE_DIR / "scripts" / ("collect_planning" + "_diagnostics.sh")).exists()


def test_benchmark_uses_selected_moveit_client():
    motion = LAUNCH_FILE.read_text(encoding="utf-8")
    assert '"moveit_clients"' in motion
    assert 'else (node_params["planning_client"],)' in motion
    assert '"fairino_ik_task_profile"' in motion
    assert 'else "grasp"' in motion


def test_benchmark_sets_private_planner_stats_paths():
    source = LAUNCH_FILE.read_text(encoding="utf-8")
    assert "SetEnvironmentVariable" in source
    assert "FAIRINO_PLANNER_DIAGNOSTICS_PATH" in source
    assert "FAIRINO_AAPF_STATS_PATH" in source
    assert "FAIRINO_MIRE_STATS_PATH" in source
    assert "FAIRINO_ANYTIME_TRACE_PATH" in source
    assert "FAIRINO_ROOT_DIAGNOSTICS_PATH" in source
    assert "FAIRINO_TRAJECTORY_PATHS_PATH" in source
    assert 'if text.endswith("*"):' in source
    assert 'text = f"{text[:-1]}_star"' in source


def test_headless_benchmark_launch_uses_gazebo_server_only():
    source = SIM_STACK_FILE.read_text(encoding="utf-8")
    assert "def sim_node(world: str, *, headless: bool = False):" in source
    assert 'gz_args = f"{gz_args} -s"' in source
    assert "sim_node(world, headless=not enable_rviz)" in source


def test_planner_parameters_and_root_mode_reach_both_move_groups():
    source = MOVEIT_STACK_FILE.read_text(encoding="utf-8")
    assert source.count('params["mire_biait_star_core"]') == 2
    assert source.count('params["rrt_core"]') == 2
    assert source.count('params["prm_core"]') == 2
    assert '"fairino": {"planner": {"random_seed": int(planner_random_seed)}}' in source
    assert "benchmark_protocol_params" in source
    assert '"max_iterations": max_iterations' in source
    assert '"validation_distance": validation_distance' in source
    assert '"post_solution_sample_attempts": post_solution_sample_attempts' in source
    assert '"cost_only_eager"' in source
    assert '"mire_enable_effort_focal_queue"' in source
    assert '"mire_enable_lazy_edge_validation"' in source
    assert '"enable_path_optimizer": bool(comparison.get("enable_path_optimizer", False))' in source
    assert '"planning_deadline_s": planning_deadline_s' in source
    assert '"max_goal_roots": 1' in source
    assert '"rrt",' not in source.split("for name in (", 1)[1].split(")", 1)[0]
    assert '"prm",' in source.split("for name in (", 1)[1].split(")", 1)[0]


def test_benchmark_disables_only_the_shared_path_optimizer_override():
    params = (PACKAGE_DIR / "config" / "motion_planning_demo_params.yaml").read_text(
        encoding="utf-8"
    )
    assert "enable_path_optimizer: false" in params
    assert "optimizer_fail_open_return_original" not in params
    assert "shortcut_trials:" not in params


def test_benchmark_projects_the_internal_deadline_without_an_outer_watchdog():
    source = LAUNCH_FILE.read_text(encoding="utf-8")
    params = (PACKAGE_DIR / "config" / "motion_planning_demo_params.yaml").read_text(
        encoding="utf-8"
    )
    node = NODE_FILE.read_text(encoding="utf-8")
    assert "planning_deadline_s: 15.0" in params
    assert "max_iterations: 3000" in params
    assert "post_solution_sample_attempts: 50" in params
    assert "watchdog_s" not in params
    assert '"benchmark_watchdog_s"' not in source
    plan_method = node.split("    def _plan_pose_from_start", 1)[1].split(
        "    def _publish_display_trajectory", 1)[0]
    assert "cancel_plan_future" not in plan_method


def test_normal_planning_uses_30_seconds_and_benchmark_requests_use_15_seconds():
    params = (PACKAGE_DIR / "config" / "motion_planning_demo_params.yaml").read_text(
        encoding="utf-8"
    )
    launch = LAUNCH_FILE.read_text(encoding="utf-8")
    node = NODE_FILE.read_text(encoding="utf-8")
    pipeline = (
        PACKAGE_DIR.parent
        / "myrobot_planning_ros"
        / "src"
        / "pipeline"
        / "fairino_planning_pipeline.cpp"
    ).read_text(encoding="utf-8")
    common = (
        PACKAGE_DIR.parent / "myrobot_planning_core" / "config" / "common_planning_params.yaml"
    ).read_text(encoding="utf-8")
    assert "allowed_planning_time: 30.0" in params
    assert "planning_deadline_s: 30.0" in common
    assert '"allowed_planning_time"' in launch
    assert "self.effective_allowed_planning_time" in node
    assert 'comparison.get("planning_deadline_s", 15.0)' in node
    assert "std::min(configured_deadline_s, requested_deadline_s)" in pipeline


def test_benchmark_uses_ik_mode_without_the_legacy_profile_key():
    params = (PACKAGE_DIR / "config" / "motion_planning_demo_params.yaml").read_text(
        encoding="utf-8"
    )
    node = NODE_FILE.read_text(encoding="utf-8")
    legacy_key = "benchmark_ik" + "_task_profile"
    assert "ik_mode: continuous" in params
    assert "goal_root_mode:" in params
    assert any(mode in params for mode in ("goal_root_mode: single_root", "goal_root_mode: multi_root"))
    assert "validation_distance: 0.03" in params
    assert legacy_key not in params
    assert legacy_key not in node


def test_start_joints_replace_the_legacy_home_joint_contract():
    params = (PACKAGE_DIR / "config" / "motion_planning_demo_params.yaml").read_text(
        encoding="utf-8"
    )
    node = NODE_FILE.read_text(encoding="utf-8")
    assert "start_joints:" in params
    assert "start_id: home" in params
    assert "go_start_before_demo" in params
    assert "def go_start" in node
    assert "def _plan_pose_from_start" in node
    legacy_joint_key = "home_" + "joints"
    legacy_command = "go_" + "home"
    assert "start_pose" not in params
    assert legacy_joint_key not in params
    assert legacy_command not in node


def test_deadline_and_fixed_post_solution_budget_are_centralized():
    stack = MOVEIT_STACK_FILE.read_text(encoding="utf-8")
    assert '"max_iterations": max_iterations' in stack
    assert '"validation_distance": validation_distance' in stack
    assert '"planning_deadline_s": planning_deadline_s' in stack
    assert '"post_solution_sample_attempts": post_solution_sample_attempts' in stack
    assert "benchmark_goal_root_mode" in stack
    core_root = PACKAGE_DIR.parent / "myrobot_planning_core" / "src" / "algorithms"
    assert any(
        "post_solution_sample_attempts" in path.read_text(encoding="utf-8")
        for path in core_root.rglob("*.cpp")
    )


def test_normal_fairino_planners_use_the_fixed_post_solution_budget():
    for filename, planner in (
        ("mire_biait*_params.yaml", "mire_biait_star"),
        ("aapf_birrt*_params.yaml", "aapf_birrt_star"),
        ("birrt*_params.yaml", "birrt_star"),
        ("rrt*_params.yaml", "rrt_star"),
        ("rrt_params.yaml", "rrt"),
        ("prm_params.yaml", "prm"),
    ):
        config = (CORE_CONFIG_DIR / filename).read_text(encoding="utf-8")
        assert f"    {planner}:" in config
        assert "max_iterations: 3000" in config
        assert "post_solution_sample_attempts: 50" in config

    hardware_launch = (
        PACKAGE_DIR.parent
        / "myrobot_support_ws"
        / "fairino_arm_moveit_config"
        / "launch"
        / "moveit_hardware.launch.py"
    ).read_text(encoding="utf-8")
    assert '"mire_biait*_params.yaml"' in hardware_launch
    assert '"prm_params.yaml"' in hardware_launch


def test_benchmark_root_mode_is_explicit_and_removed_planner_is_not_loadable():
    stack = MOVEIT_STACK_FILE.read_text(encoding="utf-8")
    node = NODE_FILE.read_text(encoding="utf-8")
    assert '"max_goal_roots": 1' in stack
    assert "single_root" in node and "multi_root" in node
    for path in (
        PACKAGE_DIR.parent / "myrobot_planning_core",
        PACKAGE_DIR.parent / "myrobot_planning_ros",
    ):
        assert ("tu" + "be_" + "birrt") not in "\n".join(
            file.read_text(encoding="utf-8")
            for file in path.rglob("*")
            if file.is_file() and file.suffix in {".cpp", ".h", ".hpp", ".yaml"}
        )


def test_root_diagnostics_preserve_per_root_effort_fields():
    pipeline = (
        PACKAGE_DIR.parent
        / "myrobot_planning_ros"
        / "src"
        / "pipeline"
        / "fairino_planning_pipeline.cpp"
    ).read_text(encoding="utf-8")
