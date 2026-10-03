import ast
from pathlib import Path
import sys

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_perception_utils.depth_estimation import (  # noqa: E402
    robust_box_placement_from_depth,
    robust_center3d_from_obb_depth,
)
from visual_perception_utils.obb_geometry import cube_edge_axis, pca_major_axis, yaw_0_to_pi_right0_left180  # noqa: E402


CAMERA_INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
def _obb():
    return np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)


@pytest.mark.parametrize(
    "filename",
    ["yolo_kalman_detector_obb.py", "yolo_detector_obb.py", "yolo_detector.py"],
)
def test_yolo_nodes_latch_three_stable_camera_info_frames(filename):
    source = (Path(__file__).resolve().parents[1] / "visual_perception" / "nodes" / filename).read_text()
    tree = ast.parse(source)
    callback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "camera_info_callback"
    )

    assert "validate_rgbd_camera_info" not in source
    assert "self._camera_info_stable_count < 3" in source
    assert "CameraInfo locked after 3 stable frames" in source
    assert "def _ready_for_3d" not in source
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "destroy_subscription"
        for call in ast.walk(callback)
    )


def test_depth_estimation_has_no_camera_info_validation_helpers():
    source = (Path(__file__).resolve().parents[1] / "visual_perception_utils" / "depth_estimation.py").read_text()

    for symbol in ("_camera_frame_root", "_same_camera_frame", "validate_rgbd_camera_info"):
        assert symbol not in source


def test_center3d_rejects_sparse_depth_points():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[45:50, 45:50] = 1.0

    center, _quality = robust_center3d_from_obb_depth(
        poly_2d=_obb(),
        depth=depth,
        camera_intrinsics=CAMERA_INTRINSICS,
        stride=1,
        min_points=50,
        max_points=10000,
        depth_max_range=10.0,
        depth_inlier_m=0.08,
        depth_mad_scale=3.0,
        min_depth_inlier_ratio=0.6,
    )

    assert center is None


def test_center3d_uses_object_depth_when_mask_has_outliers():
    depth = np.ones((100, 100), dtype=np.float32)
    depth[25:75, 25:75] = 1.0
    depth[25:75:5, 25:75] = 1.8

    center, quality = robust_center3d_from_obb_depth(
        poly_2d=_obb(),
        depth=depth,
        camera_intrinsics=CAMERA_INTRINSICS,
        stride=1,
        min_points=20,
        max_points=10000,
        depth_max_range=10.0,
        depth_inlier_m=0.08,
        depth_mad_scale=3.0,
        min_depth_inlier_ratio=0.6,
    )

    assert center is not None
    assert abs(float(center[2]) - 1.0) < 0.02
    assert quality > 0.6


def test_center3d_can_project_xy_from_obb_center_for_open_box():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[65:75, 25:75] = 1.0

    center, quality = robust_center3d_from_obb_depth(
        poly_2d=_obb(),
        depth=depth,
        camera_intrinsics=CAMERA_INTRINSICS,
        stride=1,
        min_points=20,
        max_points=10000,
        depth_max_range=10.0,
        depth_inlier_m=0.08,
        depth_mad_scale=3.0,
        min_depth_inlier_ratio=0.6,
        xy_from_obb_center=True,
    )

    assert center is not None
    assert abs(float(center[0])) < 0.001
    assert abs(float(center[1])) < 0.001
    assert abs(float(center[2]) - 1.0) < 0.001
    assert quality == 1.0


def test_center3d_rejects_highly_split_depth_distribution():
    depth = np.ones((100, 100), dtype=np.float32)
    depth[:, ::2] = 1.4

    center, _quality = robust_center3d_from_obb_depth(
        poly_2d=_obb(),
        depth=depth,
        camera_intrinsics=CAMERA_INTRINSICS,
        stride=1,
        min_points=20,
        max_points=10000,
        depth_max_range=10.0,
        depth_inlier_m=0.08,
        depth_mad_scale=3.0,
        min_depth_inlier_ratio=0.6,
    )

    assert center is None


def test_box_placement_uses_free_floor_cell_when_center_contains_an_object():
    depth = np.ones((100, 100), dtype=np.float32)
    depth[40:60, 40:60] = 0.75
    occupied = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.float32)

    point, quality, uv, free_cells = robust_box_placement_from_depth(
        _obb(), depth, CAMERA_INTRINSICS, occupied_polys=(occupied,)
    )

    assert point is not None
    assert float(point[2]) == pytest.approx(1.0)
    assert quality >= 0.6
    assert cv2.pointPolygonTest(occupied, tuple(map(float, uv)), False) < 0
    assert 0 < free_cells < 9


def test_box_placement_rejects_a_fully_occupied_inner_region():
    occupied = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)

    point, quality, uv, free_cells = robust_box_placement_from_depth(
        _obb(), np.ones((100, 100), dtype=np.float32), CAMERA_INTRINSICS,
        occupied_polys=(occupied,),
    )

    assert point is None and uv is None
    assert quality == 0.0
    assert free_cells == 0


def test_uniform_sampling_covers_a_large_obb_before_max_points_limit():
    depth = np.ones((300, 300), dtype=np.float32)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 150.0, "cy": 150.0}
    center, quality = robust_center3d_from_obb_depth(
        poly_2d=np.array([[20, 20], [280, 20], [280, 280], [20, 280]], dtype=np.float32),
        depth=depth, camera_intrinsics=intrinsics, stride=1, min_points=20, max_points=200,
        depth_max_range=10.0, depth_inlier_m=0.08, depth_mad_scale=3.0, min_depth_inlier_ratio=0.6,
    )
    assert quality == pytest.approx(1.0)
    assert center is not None
    assert abs(float(center[0])) < 0.03 and abs(float(center[1])) < 0.03


def test_obb_yaw_preserves_slash_backslash_and_corner_order_equivalence():
    slash = np.array([[0, 0], [2, 2], [1, 3], [-1, 1]], dtype=np.float32)
    backslash = np.array([[0, 0], [2, -2], [3, -1], [1, 1]], dtype=np.float32)
    assert np.degrees(yaw_0_to_pi_right0_left180(slash)) == pytest.approx(45.0)
    assert np.degrees(yaw_0_to_pi_right0_left180(backslash)) == pytest.approx(135.0)
    assert yaw_0_to_pi_right0_left180(slash[::-1]) == pytest.approx(yaw_0_to_pi_right0_left180(slash))


def test_pca_and_cube_axis_have_expected_safety_behavior():
    elongated = np.column_stack((np.linspace(-1.0, 1.0, 20), np.zeros(20), np.ones(20)))
    axis, quality = pca_major_axis(elongated)
    assert axis is not None and abs(float(axis[0])) > 0.99 and quality > 0.99
    circular = np.column_stack((np.cos(np.linspace(0, 2 * np.pi, 40, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False)), np.ones(40)))
    assert pca_major_axis(circular, min_quality=0.30)[0] is None
    corners = np.array([[0, 0], [10, 0], [10, 2], [0, 2]], dtype=np.float32)
    uv = np.column_stack((np.linspace(0, 10, 20), np.ones(20)))
    points = np.column_stack((uv[:, 0], np.zeros(20), np.ones(20)))
    cube_axis = cube_edge_axis(points, uv, corners, min_points=8)
    assert cube_axis is not None and abs(float(cube_axis[0])) > 0.99
