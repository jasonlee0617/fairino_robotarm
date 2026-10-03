import cv2
import numpy as np

def robust_obb_depth_samples(
    poly_2d: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: dict,
    stride: int,
    min_points: int,
    max_points: int,
    depth_max_range: float,
    depth_inlier_m: float,
    depth_mad_scale: float,
    min_depth_inlier_ratio: float,
    xy_from_obb_center: bool = False,
):
    """Return robust OBB center, depth quality, 3-D inliers and their pixels."""
    H, W = depth.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_2d.astype(np.int32)], 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    ys, xs = np.where(mask > 0)
    if xs.size < int(min_points):
        return None, 0.0, None, None

    stride = max(1, int(stride))
    sample_count = min(xs.size, max(1, int(np.ceil(xs.size / stride))), int(max_points))
    sample_indices = np.linspace(0, xs.size - 1, num=sample_count, dtype=np.int64)
    sample_xs = xs[sample_indices]
    sample_ys = ys[sample_indices]
    if sample_xs.size < min_points:
        return None, 0.0, None, None

    zs = depth[sample_ys, sample_xs].astype(np.float32)
    valid = np.isfinite(zs) & (zs > 0.0) & (zs <= depth_max_range)
    if int(np.count_nonzero(valid)) < min_points:
        return None, 0.0, None, None

    sample_xs = sample_xs[valid].astype(np.float32)
    sample_ys = sample_ys[valid].astype(np.float32)
    zs = zs[valid]

    z_median = float(np.median(zs))
    abs_dev = np.abs(zs - z_median)
    mad = float(np.median(abs_dev))
    cutoff = max(0.001, float(depth_inlier_m))
    if depth_mad_scale > 0.0 and mad > 0.0:
        robust_sigma = 1.4826 * mad
        cutoff = min(cutoff, max(0.005, float(depth_mad_scale) * robust_sigma))

    inlier = abs_dev <= cutoff
    inlier_count = int(np.count_nonzero(inlier))
    if inlier_count < min_points:
        return None, 0.0, None, None

    inlier_ratio = float(inlier_count) / float(zs.size)
    if inlier_ratio < min(1.0, max(0.0, float(min_depth_inlier_ratio))):
        return None, inlier_ratio, None, None

    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])
    xs_in = sample_xs[inlier]
    ys_in = sample_ys[inlier]
    zs_in = zs[inlier]
    points = np.column_stack(((xs_in - cx) * zs_in / fx, (ys_in - cy) * zs_in / fy, zs_in)).astype(np.float32)
    pixels_uv = np.column_stack((xs_in, ys_in)).astype(np.float32)
    if xy_from_obb_center:
        u, v = np.mean(poly_2d.reshape(-1, 2), axis=0).astype(np.float32)
        z = np.float32(np.median(zs_in))
        center = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float32)
        return center, inlier_ratio, points, pixels_uv

    return (
        np.mean(points, axis=0).astype(np.float32),
        inlier_ratio,
        points,
        pixels_uv,
    )


def robust_center3d_from_obb_depth(
    poly_2d: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: dict,
    stride: int,
    min_points: int,
    max_points: int,
    depth_max_range: float,
    depth_inlier_m: float,
    depth_mad_scale: float,
    min_depth_inlier_ratio: float,
    xy_from_obb_center: bool = False,
):
    """Backward-compatible two-value center/quality API."""
    center, quality, _points, _uv = robust_obb_depth_samples(
        poly_2d=poly_2d,
        depth=depth,
        camera_intrinsics=camera_intrinsics,
        stride=stride,
        min_points=min_points,
        max_points=max_points,
        depth_max_range=depth_max_range,
        depth_inlier_m=depth_inlier_m,
        depth_mad_scale=depth_mad_scale,
        min_depth_inlier_ratio=min_depth_inlier_ratio,
        xy_from_obb_center=xy_from_obb_center,
    )
    return center, quality


def robust_box_placement_from_depth(
    poly_2d: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: dict,
    *,
    occupied_polys=(),
    inner_scale: float = 0.65,
    grid_size: int = 3,
):
    """Find a locally consistent free placement point inside an open box."""
    corners = np.asarray(poly_2d, dtype=np.float32).reshape(4, 2)
    center_uv = np.mean(corners, axis=0)
    inner = center_uv + (corners - center_uv) * float(inner_scale)
    transform = cv2.getPerspectiveTransform(
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        inner,
    )
    candidates = []
    free_cells = 0
    grid_size = max(1, int(grid_size))
    for row in range(grid_size):
        for column in range(grid_size):
            x0, x1 = column / grid_size, (column + 1) / grid_size
            y0, y1 = row / grid_size, (row + 1) / grid_size
            unit_cell = np.array(
                [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
                dtype=np.float32,
            )
            cell = cv2.perspectiveTransform(unit_cell, transform)[0]
            cell_uv = np.mean(cell, axis=0)
            if any(
                cv2.pointPolygonTest(
                    np.asarray(occupied, dtype=np.float32),
                    (float(cell_uv[0]), float(cell_uv[1])),
                    False,
                ) >= 0
                for occupied in occupied_polys
            ):
                continue
            free_cells += 1
            point, quality = robust_center3d_from_obb_depth(
                poly_2d=cell,
                depth=depth,
                camera_intrinsics=camera_intrinsics,
                stride=1,
                min_points=12,
                max_points=800,
                depth_max_range=10.0,
                depth_inlier_m=0.08,
                depth_mad_scale=3.0,
                min_depth_inlier_ratio=0.6,
                xy_from_obb_center=True,
            )
            if point is not None:
                candidates.append((point, float(quality), cell_uv))
    if not candidates:
        return None, 0.0, None, free_cells

    farthest_depth = max(float(item[0][2]) for item in candidates)
    floor_candidates = [
        item for item in candidates
        if farthest_depth - float(item[0][2]) <= 0.08
    ]
    selected = min(
        floor_candidates,
        key=lambda item: float(np.linalg.norm(item[2] - center_uv)),
    )
    return selected[0], selected[1], selected[2], free_cells
