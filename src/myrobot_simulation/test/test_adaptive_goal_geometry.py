#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from motion_planning_node_sim import MotionPlanningNodeSim  # noqa: E402
from planning_benchmark import (  # noqa: E402
    GoalSetSpec,
    adaptive_challenge_metrics,
    build_goal_sampling_report,
    distance_to_obstacle_surface,
    iter_random_candidates,
    load_goal_collection,
    obstacle_signature,
    write_goal_collection,
)


class AdaptiveGoalGeometryTest(unittest.TestCase):
    def setUp(self):
        self.node = object.__new__(MotionPlanningNodeSim)
        self.node.active_obstacles = [
            {
                "name": "left",
                "shape": "cylinder",
                "position": (0.25, 0.32, 0.25),
                "radius": 0.055,
                "height": 0.34,
            },
            {
                "name": "goal",
                "shape": "sphere",
                "position": (0.42, -0.10, 0.36),
                "radius": 0.060,
            },
            {
                "name": "lower",
                "shape": "sphere",
                "position": (0.25, 0.10, 0.17),
                "radius": 0.040,
            },
            {
                "name": "right",
                "shape": "box",
                "position": (0.46, -0.12, 0.20),
                "size": (0.08, 0.08, 0.16),
            },
        ]
        self.node.planning_scene_obstacle_padding_m = 0.03
        self.node.benchmark_goal_corridor_clearance_max_m = 0.10
        self.node.scene_name = "multi_obstacle_3d_avoidance"
        self.node.benchmark_goal_mode = "adaptive_obstacle_challenge_region"
        self.node.ik_mode = "continuous"
        self.node.benchmark_goal_seed = 17

    def test_central_goal_is_adaptive_challenge(self):
        goal = (0.25, 0.11, 0.30)
        metrics = adaptive_challenge_metrics(
            goal,
            start_xyz=(0.40, 0.20, 0.20),
            obstacles=self.node.active_obstacles,
            corridor_clearance_max_m=self.node.benchmark_goal_corridor_clearance_max_m,
        )
        clearance = distance_to_obstacle_surface(
            goal, self.node.active_obstacles
        )

        self.assertTrue(metrics["accepted"])
        self.assertGreaterEqual(metrics["angular_coverage_deg"], 180.0)
        self.assertLessEqual(metrics["corridor_min_clearance_m"], 0.10)
        self.assertGreaterEqual(clearance, 0.06)
        self.assertLessEqual(clearance, 0.14)

    def test_surface_distance_handles_box_sphere_and_cylinder(self):
        obstacles = [
            {"shape": "box", "position": (0.0, 0.0, 0.0), "size": (2.0, 2.0, 2.0)},
            {"shape": "sphere", "position": (5.0, 0.0, 0.0), "radius": 1.0},
            {"shape": "cylinder", "position": (9.0, 0.0, 0.0), "radius": 1.0, "height": 2.0},
        ]
        self.assertAlmostEqual(distance_to_obstacle_surface((2.0, 0.0, 0.0), obstacles[:1]), 1.0)
        self.assertAlmostEqual(distance_to_obstacle_surface((7.0, 0.0, 0.0), obstacles[1:2]), 1.0)
        self.assertAlmostEqual(distance_to_obstacle_surface((11.0, 0.0, 0.0), obstacles[2:]), 1.0)

    def test_only_adaptive_goal_mode_is_accepted(self):
        self.assertEqual(
            self.node._normalize_benchmark_goal_mode("adaptive"),
            "adaptive_obstacle_challenge_region",
        )
        self.assertEqual(
            self.node._normalize_benchmark_goal_mode(
                "adaptive_obstacle_challenge_region"
            ),
            "adaptive_obstacle_challenge_region",
        )
        for removed_mode in (
            "fixed",
            "random_obstacle_envelope",
            "random_pose_goal_region",
        ):
            with self.assertRaisesRegex(ValueError, "仅支持"):
                self.node._normalize_benchmark_goal_mode(removed_mode)

    def test_scene_config_has_no_scene_level_start_pose(self):
        scene_file = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "scenes"
            / "pathplanning_scenes_params.yaml"
        )
        scenes = yaml.safe_load(scene_file.read_text(encoding="utf-8"))["scenes"]
        for scene in scenes.values():
            self.assertNotIn("start_pose", scene.get("benchmark", {}))
            self.assertNotIn("goal_pose", scene.get("benchmark", {}))

    def test_layout_signature_changes_with_obstacle_position(self):
        original = obstacle_signature(self.node.active_obstacles)
        self.node.active_obstacles[0]["position"] = (0.26, 0.32, 0.25)
        self.assertNotEqual(original, obstacle_signature(self.node.active_obstacles))

    def _goal_spec(self):
        return GoalSetSpec(
            scene_name=self.node.scene_name,
            goal_mode=self.node.benchmark_goal_mode,
            goal_seed=self.node.benchmark_goal_seed,
            ik_mode=self.node.ik_mode,
            obstacle_signature=obstacle_signature(self.node.active_obstacles),
            target_rpy_deg=(0.0, -180.0, 0.0),
            repetitions=self.node.benchmark_repetitions,
            min_separation_m=0.04,
            clearance_min_m=0.06,
            clearance_max_m=0.14,
            corridor_clearance_max_m=0.10,
            sampling_start_xyz=(0.40, 0.20, 0.20),
            sampling_start_joints=(-1.1170, -1.6214, 1.5465, -1.5877, -1.6368, 0.0),
            start_id="home",
        )

    def test_shared_goals_reject_changed_layout(self):
        goal = ((0.25, 0.11, 0.30), (0.0, -180.0, 0.0))
        self.node.benchmark_repetitions = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            write_goal_collection(
                tmp_dir, self._goal_spec(), [goal], "scene-sha", {"accepted_goals": 1}
            )
            _paths, loaded, _manifest = load_goal_collection(tmp_dir, self._goal_spec())
            self.assertEqual(loaded, [goal])

            self.assertEqual(
                load_goal_collection(tmp_dir, self._goal_spec())[1],
                [goal],
            )

            self.node.ik_mode = "grasp"
            with self.assertRaises((FileNotFoundError, ValueError)):
                load_goal_collection(tmp_dir, self._goal_spec())
            self.node.ik_mode = "continuous"

            self.node.active_obstacles[0]["position"] = (0.26, 0.32, 0.25)
            with self.assertRaises((FileNotFoundError, ValueError)):
                load_goal_collection(tmp_dir, self._goal_spec())

    def test_random_goal_stream_is_deterministic(self):
        first = iter_random_candidates((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 17)
        second = iter_random_candidates((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 17)
        self.assertEqual([next(first) for _ in range(20)], [next(second) for _ in range(20)])

    def test_goal_sampling_report_preserves_rejection_reasons(self):
        report = build_goal_sampling_report(
            "dense_extreme_multi_obstacle_3d_avoidance",
            goal_seed=17,
            requested_goals=20,
            attempts=80,
            accepted_goals=20,
            rejected={"geometry": 30, "ik": 20, "state": 5, "separation": 5},
            endpoint_clearances=[0.06, 0.08, 0.10],
        )
        self.assertEqual(report["accepted_goals"], 20)
        self.assertEqual(report["candidate_attempts"], 80)
        self.assertAlmostEqual(report["acceptance_rate"], 0.25)
        self.assertEqual(report["rejected_ik"], 20)
        self.assertAlmostEqual(report["accepted_endpoint_clearance_mean_m"], 0.08)

    def test_streaming_goals_skip_invalid_candidates_until_accepted(self):
        self.node.benchmark_goal_min_separation_m = 0.04
        self.node.get_logger = lambda: type("Logger", (), {"info": lambda *_args: None})()
        self.node._benchmark_candidate_status = lambda point, *_args: (
            (point[0] > 0.5, "geometry")
        )
        with patch(
            "motion_planning_node_sim.goal_bounds",
            return_value=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ):
            goals = list(self.node._iter_benchmark_goals(1, (0.0, 0.0, 0.0), (0.0, -180.0, 0.0)))
        self.assertEqual(len(goals), 1)


if __name__ == "__main__":
    unittest.main()
