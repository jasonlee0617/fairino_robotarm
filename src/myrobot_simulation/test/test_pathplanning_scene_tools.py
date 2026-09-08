#!/usr/bin/env python3
"""Geometry contract checks for YAML-driven Gazebo scene spawning."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR / "scripts"))

from pathplanning_scene_tools import SceneLoader, SimSceneSpawner  # noqa: E402
from planning_benchmark import (  # noqa: E402
    goal_bounds,
    obstacle_signature,
    obstacle_world_half_extents,
)


def _obstacle(shape, **kwargs):
    return SceneLoader.make_obstacle(
        name="obstacle",
        position=(0.0, 0.0, 0.0),
        shape=shape,
        color=(0.1, 0.2, 0.3, 0.4),
        **kwargs,
    )


def _sdf_root(obstacle):
    return ET.fromstring(SimSceneSpawner.sdf_xml(obstacle, "scene_obstacle"))


def test_sdf_geometry_uses_raw_yaml_dimensions_for_all_shapes():
    box = _sdf_root(_obstacle("box", size=(0.3, 0.3, 0.8)))
    cylinder = _sdf_root(_obstacle("cylinder", radius=0.055, height=0.34))
    sphere = _sdf_root(_obstacle("sphere", radius=0.08))

    assert box.findtext(".//collision/geometry/box/size") == "0.3 0.3 0.8"
    assert box.findtext(".//visual/geometry/box/size") == "0.3 0.3 0.8"
    assert cylinder.findtext(".//collision/geometry/cylinder/radius") == "0.055"
    assert cylinder.findtext(".//collision/geometry/cylinder/length") == "0.34"
    assert sphere.findtext(".//collision/geometry/sphere/radius") == "0.08"
    assert sphere.findtext(".//visual/material/diffuse") == "0.1 0.2 0.3 0.4"
    assert box.findtext(".//model/static") == "true"


def test_dense_box_sdf_is_not_bound_to_a_legacy_asset():
    scene_file = PACKAGE_DIR / "config" / "scenes" / "pathplanning_scenes_params.yaml"
    loader = SceneLoader("dense_obstacle_3d_avoidance-1", str(scene_file), _NullLogger())
    dense_box = next(item for item in loader.load()
                     if item.shape == "box")

    sdf_xml = SimSceneSpawner.sdf_xml(dense_box, "dense_box")
    expected_size = " ".join(format(value, ".12g") for value in dense_box.size)
    assert expected_size in sdf_xml
    assert ".sdf" not in sdf_xml


class _NullLogger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass


def test_scene_loader_requires_nonempty_yaml(tmp_path):
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    loader = SceneLoader("single_obstacle", str(empty_file), _NullLogger())
    with pytest.raises(ValueError, match="为空或格式无效"):
        loader.load()


def test_scene_loader_missing_yaml_fails_without_single_obstacle_fallback(tmp_path):
    loader = SceneLoader(
        "single_obstacle", str(tmp_path / "missing.yaml"), _NullLogger()
    )
    with pytest.raises(FileNotFoundError, match="配置文件不存在"):
        loader.load()


def test_dense_difficulty_levels_parse_and_preserve_baseline():
    scene_file = PACKAGE_DIR / "config" / "scenes" / "pathplanning_scenes_params.yaml"
    baseline = SceneLoader("dense_obstacle_3d_avoidance-1", str(scene_file), _NullLogger())
    hard = SceneLoader("dense_obstacle_3d_avoidance-2", str(scene_file), _NullLogger())
    extreme = SceneLoader("dense_obstacle_3d_avoidance-3", str(scene_file), _NullLogger())

    baseline_obstacles = baseline.load()
    hard_obstacles = hard.load()
    extreme_obstacles = extreme.load()

    assert len(baseline_obstacles) == 6
    assert len(hard_obstacles) == 10
    assert len(extreme_obstacles) == 9
    assert hard.benchmark["difficulty_level"] == "hard"
    assert extreme.benchmark["difficulty_level"] == "extreme"
    assert any(item.rpy_deg[2] != 0.0 for item in extreme_obstacles if item.shape == "box")
    assert obstacle_signature(baseline_obstacles) != obstacle_signature(hard_obstacles)


def test_extreme_gate_layout_is_robot_workspace_aware():
    scene_file = PACKAGE_DIR / "config" / "scenes" / "pathplanning_scenes_params.yaml"
    loader = SceneLoader(
        "dense_obstacle_3d_avoidance-3", str(scene_file), _NullLogger()
    )
    obstacles = loader.load()
    gates = {item.name: item for item in obstacles if "gate_" in item.name}
    assert set(gates) == {"gate_a", "gate_b", "gate_c"}
    assert gates["gate_a"].position == (0.388, 0.22, 0.25)
    assert gates["gate_b"].position == (0.54, 0.20, 0.25)
    assert gates["gate_c"].position == (0.51, 0.05, 0.25)
    assert all(item.size == (0.04, 0.08, 0.50) for item in gates.values())
    assert gates["gate_a"].rpy_deg[2] == 26.0
    assert gates["gate_b"].rpy_deg[2] == 0.0
    assert gates["gate_c"].rpy_deg[2] == -0.3
    assert all(0.10 < item.position[2] < 0.40 for item in gates.values())


def test_rotated_box_changes_layout_signature():
    base = SceneLoader.make_obstacle(
        "box", (0.0, 0.0, 0.0), size=(0.2, 0.1, 0.3), rpy_deg=(0.0, 0.0, 0.0)
    )
    rotated = SceneLoader.make_obstacle(
        "box", (0.0, 0.0, 0.0), size=(0.2, 0.1, 0.3), rpy_deg=(0.0, 0.0, 25.0)
    )
    assert obstacle_signature([base]) != obstacle_signature([rotated])


def test_extreme_obstacles_do_not_have_overlapping_world_aabbs():
    scene_file = PACKAGE_DIR / "config" / "scenes" / "pathplanning_scenes_params.yaml"
    loader = SceneLoader("dense_obstacle_3d_avoidance-3", str(scene_file), _NullLogger())
    obstacles = loader.load()
    bounds = [
        (
            np.asarray(item.position) - obstacle_world_half_extents(item),
            np.asarray(item.position) + obstacle_world_half_extents(item),
        )
        for item in obstacles
    ]
    for index, (_lower, upper) in enumerate(bounds):
        for lower, _upper in bounds[index + 1:]:
            assert not np.all(np.minimum(upper, _upper) - np.maximum(_lower, lower) > 0.0)
