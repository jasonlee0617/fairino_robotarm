#!/usr/bin/env python3
"""Regression checks for decoupled goal collections and benchmark runs."""

import csv
import math
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import motion_planning_node_sim as planning_node_module
from motion_planning_node_sim import BENCHMARK_PLANNER_IDS, MotionPlanningNodeSim  # noqa: E402
from planning_benchmark import (  # noqa: E402
    ALGORITHM_DIAGNOSTIC_FIELDS,
    ANYTIME_FIELDS,
    GOAL_COLLECTION_FORMAT_VERSION,
    GoalSetSpec,
    RESULT_FIELDS,
    ROOT_DIAGNOSTIC_FIELDS,
    STANDARD_CSV_SCHEMAS,
    TRAJECTORY_PATH_FIELDS,
    benchmark_slug,
    canonical_sha256,
    finalize_run_manifest,
    write_benchmark_summary,
    goal_set_id,
    goal_set_paths,
    initialize_standard_csvs,
    load_goal_collection,
    obstacle_half_extents,
    prepare_benchmark_run,
    validate_complete_run,
    write_csv_atomic,
    write_goal_collection,
    write_results,
    write_yaml_atomic,
)


def _spec(**changes):
    values = dict(
        scene_name="dense_scene",
        goal_mode="adaptive_obstacle_challenge_region",
        goal_seed=17,
        ik_mode="continuous",
        obstacle_signature="a" * 64,
        target_rpy_deg=(0.0, -180.0, 0.0),
        repetitions=2,
        min_separation_m=0.04,
        clearance_min_m=0.06,
        clearance_max_m=0.14,
        corridor_clearance_max_m=0.10,
        sampling_start_xyz=(0.0, 0.0, 0.0),
        sampling_start_joints=(-1.0, -1.5, 1.5, -1.5, -1.5, 0.0),
        start_id="home",
    )
    values.update(changes)
    return GoalSetSpec(**values)


def _goals():
    return [
        ((0.20, 0.10, 0.30), (0.0, -180.0, 0.0)),
        ((0.30, 0.10, 0.30), (0.0, -180.0, 0.0)),
    ]


def _report():
    return {"candidate_attempts": 4, "accepted_goals": 2, "acceptance_rate": 0.5}


def test_complete_goal_collection_is_reused_and_pose_only_csv():
    with tempfile.TemporaryDirectory() as directory:
        paths, goals, manifest, reused = write_goal_collection(
            directory, _spec(), _goals(), "b" * 64, _report()
        )
        assert not reused
        assert goals == _goals()
        assert manifest["format_version"] == GOAL_COLLECTION_FORMAT_VERSION
        assert manifest["collection_key"] == goal_set_id(_spec())
        assert manifest["goal_set_id"] == goal_set_id(_spec())
        assert list(csv.DictReader(open(paths["goals"], encoding="utf-8")).fieldnames) == [
            "goal_index", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"
        ]
        same_paths, loaded, _manifest, reused = write_goal_collection(
            directory, _spec(), _goals(), "b" * 64, _report()
        )
        assert reused and same_paths == paths and loaded == _goals()


def test_goal_collection_rejects_changed_identity_and_missing_collection():
    with tempfile.TemporaryDirectory() as directory:
        paths, _goals_value, _manifest, _reused = write_goal_collection(
            directory, _spec(), _goals(), "b" * 64, _report()
        )
        for changed in (
            _spec(goal_seed=29), _spec(ik_mode="grasp"),
            _spec(obstacle_signature="c" * 64), _spec(clearance_max_m=0.15),
            _spec(sampling_start_xyz=(0.10, 0.20, 0.30)),
            _spec(sampling_start_joints=(-1.1, -1.5, 1.5, -1.5, -1.5, 0.0)),
        ):
            try:
                load_goal_collection(directory, changed)
            except (FileNotFoundError, ValueError):
                pass
            else:
                raise AssertionError("changed goal identity was accepted")
        try:
            load_goal_collection(str(Path(directory) / "missing"), _spec())
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing goal collection was accepted")
        Path(paths["sampling"]).unlink()
        try:
            write_goal_collection(directory, _spec(), _goals(), "b" * 64, _report())
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete goal collection was regenerated")


def test_goal_collection_rejects_legacy_manifest_without_writing():
    with tempfile.TemporaryDirectory() as directory:
        paths = goal_set_paths(directory, _spec())
        Path(paths["directory"]).mkdir(parents=True)
        with open(paths["goals"], "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "goal_index", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg",
            ])
            writer.writeheader()
            for index, (xyz, rpy) in enumerate(_goals(), 1):
                writer.writerow(dict(zip(writer.fieldnames, (index, *xyz, *rpy))))
        Path(paths["sampling"]).write_text("goal_set_id: old\n", encoding="utf-8")
        Path(paths["manifest"]).write_text("goal_set_id: old\n", encoding="utf-8")
        try:
            load_goal_collection(directory, _spec())
        except ValueError as exc:
            assert "legacy goal collection unsupported" in str(exc)
        else:
            raise AssertionError("legacy manifest was accepted")
        try:
            write_goal_collection(directory, _spec(), _goals(), "b" * 64, _report())
        except ValueError as exc:
            assert "legacy goal collection unsupported" in str(exc)
        else:
            raise AssertionError("legacy manifest was overwritten")


def test_v1_named_directory_is_not_scanned_or_reused():
    with tempfile.TemporaryDirectory() as directory:
        legacy = Path(directory) / "dense_scene" / "goal_seed17_n2"
        legacy.mkdir(parents=True)
        marker = legacy / "goal_set_manifest.yaml"
        marker.write_text("goal_set_id: goal_seed17_n2\n", encoding="utf-8")
        paths, _goals_value, manifest, reused = write_goal_collection(
            directory, _spec(), _goals(), "b" * 64, _report()
        )
        assert not reused
        assert Path(paths["directory"]).name == manifest["collection_key"]
        assert marker.read_text(encoding="utf-8") == "goal_set_id: goal_seed17_n2\n"


def test_sr_mr_and_planner_runs_share_one_goal_collection_and_use_safe_paths():
    with tempfile.TemporaryDirectory() as directory:
        collection_root = Path(directory) / "collections"
        result_root = Path(directory) / "results"
        paths, _goals_value, manifest, _reused = write_goal_collection(
            str(collection_root), _spec(), _goals(), "b" * 64, _report()
        )
        sr = prepare_benchmark_run(
            str(result_root), "dense_scene", manifest["goal_set_id"], "mire_biait*", 7,
            "single_root", {"planner_id": "mire_biait*", "goal_set_csv_sha256": manifest["goal_set_csv_sha256"]}, "20260903_120000",
        )
        mr = prepare_benchmark_run(
            str(result_root), "dense_scene", manifest["goal_set_id"], "birrt*", 29,
            "multi_root", {"planner_id": "birrt*", "goal_set_csv_sha256": manifest["goal_set_csv_sha256"]}, "20260903_120001",
        )
        assert Path(sr).parts[-5:-1] == (manifest["goal_set_id"], "sr", "mire_biait_star", "planner_seed07")
        assert Path(mr).parts[-5:-1] == (manifest["goal_set_id"], "mr", "birrt_star", "planner_seed29")
        assert paths["goals"].startswith(str(collection_root))
        assert not list(result_root.rglob("goal_set.csv"))


def _common_key(goal_index, run_signature="d" * 64):
    return {
        "scene_name": "dense_scene", "goal_set_id": "goal_seed17_n2",
        "goal_set_signature_sha256": "c" * 64, "goal_index": goal_index,
        "goal_root_mode": "multi_root", "planner_id": "birrt*",
        "planner_seed": 7, "variant": "full",
        "run_signature_sha256": run_signature,
    }


def _complete_run(directory):
    run_dir = Path(directory)
    manifest = {
        "scene_name": "dense_scene", "goal_set_id": "goal_seed17_n2",
        "goal_set_signature_sha256": "c" * 64, "goal_root_mode": "multi_root",
        "planner_id": "birrt*", "planner_random_seed": 7, "variant": "full",
        "run_signature_sha256": "d" * 64, "status": "running",
    }
    write_yaml_atomic(str(run_dir / "run_manifest.yaml"), manifest)
    initialize_standard_csvs(str(run_dir))
    results = []
    for goal in (1, 2):
        row = {field: 0 for field in RESULT_FIELDS}
        row.update(
            _common_key(goal),
            plan_success="true",
            first_solution_time_s=0.5,
            first_solution_path_cost_rad=1.0,
            joint_path_length_rad=1.0,
            tcp_path_length_m=0.2,
            joint_turn_total_variation_rad=0.2,
            waypoint_count=2,
            goal_root_count=1,
            selected_goal_root=0,
        )
        results.append(row)
    write_results(str(run_dir / "results.csv"), results)
    checkpoints = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15)
    trace = []
    for goal in (1, 2):
        for checkpoint in checkpoints:
            row = {field: 0 for field in ANYTIME_FIELDS}
            row.update(_common_key(goal), checkpoint_s=checkpoint, has_solution="false")
            trace.append(row)
    roots = []
    diagnostics = []
    trajectories = []
    for goal in (1, 2):
        root = {field: 0 for field in ROOT_DIAGNOSTIC_FIELDS}
        root.update(
            _common_key(goal), passed_hard_filter="true",
            selected_final="true", filter_reason="accepted")
        roots.append(root)
        diagnostic = {field: 0 for field in ALGORITHM_DIAGNOSTIC_FIELDS}
        diagnostic.update(
            _common_key(goal), metric_name="rewires", metric_value=0,
            metric_text="", ablation_variant="full")
        diagnostics.append(diagnostic)
        for stage in ("raw", "final"):
            for waypoint_index in (0, 1):
                waypoint = {field: 0 for field in TRAJECTORY_PATH_FIELDS}
                waypoint.update(
                    _common_key(goal), path_stage=stage,
                    waypoint_index=waypoint_index, selected_goal_root=0)
                trajectories.append(waypoint)
    write_csv_atomic(str(run_dir / "anytime_trace.csv"), trace, ANYTIME_FIELDS)
    write_csv_atomic(str(run_dir / "root_diagnostics.csv"), roots, ROOT_DIAGNOSTIC_FIELDS)
    write_csv_atomic(str(run_dir / "algorithm_diagnostics.csv"), diagnostics, ALGORITHM_DIAGNOSTIC_FIELDS)
    write_csv_atomic(str(run_dir / "trajectory_paths.csv"), trajectories, TRAJECTORY_PATH_FIELDS)
    finalize_run_manifest(str(run_dir), "complete", 2)
    return run_dir


def test_summary_is_written_for_complete_and_aborted_runs():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _complete_run(directory)
        summary_path = Path(write_benchmark_summary(str(run_dir)))
        text = summary_path.read_text(encoding="utf-8")
        assert "# Planning Benchmark Summary" in text
        assert "Success rate: 100.00%" in text
        assert "Wall time" not in text
        assert "wall_time_s" not in RESULT_FIELDS
        assert "minimum_clearance_m" not in RESULT_FIELDS
        import yaml
        manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
        assert manifest["summary"]["file"] == "summary.md"
        finalize_run_manifest(str(run_dir), "aborted", 2, "test interruption")
        text = Path(write_benchmark_summary(str(run_dir))).read_text(encoding="utf-8")
        assert "Status: aborted" in text and "test interruption" in text


def test_summary_resource_metrics_include_failed_goals_and_p95():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _complete_run(directory)
        with open(run_dir / "results.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0].update(planner_accepted_samples=10, planner_iterations=100)
        rows[1].update(
            plan_success="false", planner_accepted_samples=30, planner_iterations=300,
            first_solution_time_s="nan", first_solution_path_cost_rad="nan",
            joint_path_length_rad="nan", tcp_path_length_m="nan",
            joint_turn_total_variation_rad="nan", waypoint_count=0,
        )
        write_results(str(run_dir / "results.csv"), rows)
        text = Path(write_benchmark_summary(str(run_dir))).read_text(encoding="utf-8")
        assert "Accepted samples: mean=20.000000, median=20.000000, p95=30.000000, n=2" in text
        assert "Iterations: mean=200.000000, median=200.000000, p95=300.000000, n=2" in text


def test_five_csv_schema_hashes_and_composite_keys_are_validated():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _complete_run(directory)
        assert set(STANDARD_CSV_SCHEMAS) == {
            "results.csv", "anytime_trace.csv", "root_diagnostics.csv",
            "algorithm_diagnostics.csv", "trajectory_paths.csv",
        }
        assert validate_complete_run(str(run_dir), expected_goals=2) == (True, "complete")
        import yaml
        manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
        assert manifest["completed_goals"] == 2 and manifest["expected_goals"] == 2
        rows = list(csv.DictReader(open(run_dir / "results.csv", encoding="utf-8")))
        rows[0]["planner_seed"] = "29"
        write_results(str(run_dir / "results.csv"), rows)
        valid, reason = validate_complete_run(str(run_dir), expected_goals=2)
        assert not valid and ("composite keys" in reason or "hash" in reason)


def test_complete_run_rejects_semantically_incomplete_trajectory_sidecar():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _complete_run(directory)
        rows = list(csv.DictReader(open(run_dir / "trajectory_paths.csv", encoding="utf-8")))
        rows = [
            row for row in rows
            if not (row["goal_index"] == "1" and row["path_stage"] == "raw")
        ]
        write_csv_atomic(str(run_dir / "trajectory_paths.csv"), rows, TRAJECTORY_PATH_FIELDS)
        finalize_run_manifest(str(run_dir), "complete", 2)
        valid, reason = validate_complete_run(str(run_dir), expected_goals=2)
        assert not valid and "stages mismatch" in reason


def test_complete_run_rejects_missing_selected_root_for_success():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _complete_run(directory)
        rows = list(csv.DictReader(open(run_dir / "root_diagnostics.csv", encoding="utf-8")))
        rows[0]["selected_final"] = "false"
        write_csv_atomic(str(run_dir / "root_diagnostics.csv"), rows, ROOT_DIAGNOSTIC_FIELDS)
        finalize_run_manifest(str(run_dir), "complete", 2)
        valid, reason = validate_complete_run(str(run_dir), expected_goals=2)
        assert not valid and "selected root mismatch" in reason


def test_run_signature_hash_is_stable_for_equivalent_payloads():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_star_planner_temp_diagnostic_paths_use_the_canonical_slug():
    node = object.__new__(MotionPlanningNodeSim)
    node.benchmark_output_dir = "/tmp/benchmark"
    node.planner_random_seed = 7
    expected = {
        "rrt*": "rrt_star",
        "informed_rrt*": "informed_rrt_star",
        "birrt*": "birrt_star",
        "aapf_birrt*": "aapf_birrt_star",
        "mire_biait*": "mire_biait_star",
        "prm": "prm",
    }
    for planner_id, slug in expected.items():
        node.default_planner_id = planner_id
        assert benchmark_slug(planner_id) == slug
        assert node._planner_diagnostics_temp_path().endswith(
            f".planner_diagnostics_{slug}_seed7.csv"
        )


def test_planner_diagnostics_populate_common_benchmark_metrics(monkeypatch):
    node = object.__new__(MotionPlanningNodeSim)
    node.default_planner_id = "mire_biait*"
    node.planner_random_seed = 7
    node.scene_name = "dense_scene"
    node.benchmark_goal_root_mode = "multi_root"
    node.benchmark_variant = "full"
    node._active_run_manifest = {
        "goal_set_id": "goal_seed17_n30",
        "goal_set_signature_sha256": "c" * 64,
        "run_signature_sha256": "d" * 64,
    }
    node.planner_random_seed = 7
    diagnostics = {
        "first_solution_time_s": "0.12",
        "first_solution_path_cost_rad": "1.3",
        "optimized_tcp_path_length_m": "0.4",
        "sample_attempts": "10",
        "accepted_samples": "6",
        "iterations": "9",
        "num_nodes": "7",
        "collision_state_checks": "20",
        "collision_motion_checks": "8",
        "valid_motion_edges": "5",
        "invalid_motion_edges": "3",
        "stop_reason": "deadline_incumbent",
        "work_units": "11",
    }
    monkeypatch.setattr(node, "_latest_planner_diagnostics", lambda: diagnostics)
    row = node._benchmark_result_row(1)
    row["plan_success"] = "true"
    node._apply_planner_diagnostics(row)
    assert row["tcp_path_length_m"] == "0.4"
    assert row["collision_state_checks"] == "20"
    assert row["collision_motion_checks"] == "8"
    assert row["invalid_motion_edges"] == "3"


def test_failed_attempt_does_not_gain_success_only_diagnostics(monkeypatch):
    node = object.__new__(MotionPlanningNodeSim)
    monkeypatch.setattr(node, "_latest_planner_diagnostics", lambda: {
        "first_solution_time_s": "0.12",
        "first_solution_path_cost_rad": "1.3",
        "optimized_tcp_path_length_m": "0.4",
    })
    node._active_run_manifest = {
        "goal_set_id": "goal_seed17_n30",
        "goal_set_signature_sha256": "c" * 64,
        "run_signature_sha256": "d" * 64,
    }
    node.scene_name = "dense_scene"
    node.benchmark_goal_root_mode = "multi_root"
    node.default_planner_id = "mire_biait*"
    node.planner_random_seed = 7
    node.benchmark_variant = "full"
    row = node._benchmark_result_row(1)
    node._apply_planner_diagnostics(row)
    assert math.isnan(row["first_solution_time_s"])
    assert math.isnan(row["tcp_path_length_m"])


def test_blank_algorithm_diagnostic_fields_are_not_archived(monkeypatch):
    node = object.__new__(MotionPlanningNodeSim)
    node.scene_name = "dense_scene"
    node.benchmark_goal_root_mode = "multi_root"
    node.default_planner_id = "mire_biait*"
    node.planner_random_seed = 7
    node.benchmark_variant = "full"
    node.benchmark_effective_config = {"comparison": {"anytime_checkpoints_s": [0.1]}}
    node._active_run_manifest = {
        "goal_set_id": "goal_seed17_n30",
        "goal_set_signature_sha256": "c" * 64,
        "run_signature_sha256": "d" * 64,
    }
    node._anytime_rows = []
    node._root_rows = []
    node._algorithm_rows = []
    node._trajectory_rows = []
    monkeypatch.setattr(node, "_latest_planner_diagnostics", lambda: {
        "collision_motion_checks": "5", "stop_reason": "", "unused_value": "nan",
    })
    monkeypatch.setattr(node, "_read_temp_rows", lambda _path: [])
    monkeypatch.setattr(node, "_anytime_trace_temp_path", lambda: "")
    monkeypatch.setattr(node, "_root_diagnostics_temp_path", lambda: "")
    monkeypatch.setattr(node, "_sampling_stats_temp_path", lambda: "")
    monkeypatch.setattr(node, "_mire_stats_temp_path", lambda: "")
    monkeypatch.setattr(node, "_trajectory_paths_temp_path", lambda: "")

    node._collect_goal_sidecars(1, {"plan_success": "false", "goal_root_count": 0})

    assert [row["metric_name"] for row in node._algorithm_rows] == ["collision_motion_checks"]


def test_core_timeout_response_is_recorded_without_outer_cancellation():
    class TimedOutFuture:
        def done(self):
            return True

        def result(self):
            return SimpleNamespace(
                motion_plan_response=SimpleNamespace(
                    error_code=SimpleNamespace(
                        val=planning_node_module.MoveItErrorCodes.TIMED_OUT),
                    trajectory=SimpleNamespace(joint_trajectory=SimpleNamespace(points=[])),
                    planning_time=15.0,
                )
            )

    class Arm:
        def plan_async(self, **_kwargs):
            return TimedOutFuture()

    node = object.__new__(MotionPlanningNodeSim)
    node.moveit2_arm = Arm()
    node.start_joint_state = [0.0] * 6
    node.pose_to_pose_stamped = lambda pose: pose
    result = node._plan_pose_from_start(object())
    assert result == {
        "success": False,
        "failure_code": "planning_timeout",
        "core_planning_time_s": 15.0,
        "trajectory": None,
    }


def test_start_joints_are_used_and_collision_checked():
    node = object.__new__(MotionPlanningNodeSim)
    node.start_joints = (-1.0, -1.5, 1.5, -1.5, -1.5, 0.0)
    node.start_joint_state = None
    node._is_joint_state_valid_for_benchmark = lambda joints: joints == list(node.start_joints)

    assert node._resolve_start_joint_state(True) == list(node.start_joints)


def test_start_joints_collision_failure_is_not_fallbacked():
    node = object.__new__(MotionPlanningNodeSim)
    node.start_joints = (-1.0, -1.5, 1.5, -1.5, -1.5, 0.0)
    node.start_joint_state = None
    node._is_joint_state_valid_for_benchmark = lambda _joints: False

    try:
        node._resolve_start_joint_state(True)
    except RuntimeError as exc:
        assert str(exc) == "start_joints_state_invalid"
    else:
        raise AssertionError("invalid start joints were accepted")


def test_goal_collection_directory_uses_the_manual_start_id():
    assert Path(goal_set_paths("/tmp/collections", _spec())["directory"]).name == (
        "goal_seed17_n2_start_id_home"
    )
    assert goal_set_id(_spec(start_id="02")) == "goal_seed17_n2_start_id_02"


def test_same_start_id_rejects_changed_start_joints(tmp_path):
    write_goal_collection(tmp_path, _spec(), _goals(), "scene-sha", {"accepted_goals": 2})
    try:
        load_goal_collection(
            tmp_path,
            _spec(sampling_start_joints=(-1.1, -1.5, 1.5, -1.5, -1.5, 0.0)),
        )
    except ValueError as exc:
        assert "请更换 start_id" in str(exc)
    else:
        raise AssertionError("changed start joints reused an existing start_id")


def test_benchmark_planner_set_excludes_standard_rrt():
    assert BENCHMARK_PLANNER_IDS == {
        "rrt*", "informed_rrt*", "birrt*", "aapf_birrt*", "mire_biait*", "prm",
    }


def test_sphere_extents_do_not_read_none_height():
    assert obstacle_half_extents({"shape": "sphere", "radius": 0.06, "height": None}) == (0.06, 0.06, 0.06)


def test_archive_paths_expand_user_and_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_ARCHIVE_ROOT", str(tmp_path))
    assert MotionPlanningNodeSim._resolve_benchmark_output_dir("$BENCHMARK_ARCHIVE_ROOT/case") == str(tmp_path / "case")
