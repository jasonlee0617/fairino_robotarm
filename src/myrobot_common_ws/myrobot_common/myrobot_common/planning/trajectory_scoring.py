from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TrajectoryScore:
    total_cost: float
    path_length: float
    wrist_length: float
    max_joint_step: float
    smoothness: float
    duration: float
    num_points: int
    valid: bool = True
    reason: str = ""


@dataclass(frozen=True)
class TrajectoryScoreConfig:
    path_length_weight: float = 1.0
    wrist_length_weight: float = 50.0
    max_joint_step_weight: float = 2.0
    smoothness_weight: float = 0.5
    duration_weight: float = 0.0
    wrist_joint_indices: tuple[int, ...] = (2, 3, 4)
    wrist_joint_names: Optional[tuple[str, ...]] = None
    invalid_cost: float = 1.0e9


def _duration_seconds(duration) -> float:
    return (
        float(getattr(duration, "sec", 0))
        + float(getattr(duration, "nanosec", 0)) * 1.0e-9
    )


def _trajectory_matrix(trajectory):
    points = list(getattr(trajectory, "points", []) or [])
    if not points:
        return np.zeros((0, 0), dtype=float), "trajectory has no points"

    rows = []
    expected_dim = None
    for i, point in enumerate(points):
        positions = list(getattr(point, "positions", []) or [])
        if not positions:
            return (
                np.zeros((0, 0), dtype=float),
                f"point {i} has no positions",
            )
        if expected_dim is None:
            expected_dim = len(positions)
        elif len(positions) != expected_dim:
            return np.zeros((0, 0), dtype=float), (
                f"point {i} position dimension {len(positions)} "
                f"!= {expected_dim}"
            )
        row = np.array(positions, dtype=float)
        if not np.all(np.isfinite(row)):
            return (
                np.zeros((0, 0), dtype=float),
                f"point {i} contains non-finite positions",
            )
        rows.append(row)

    return np.vstack(rows), ""


def _joint_deltas(trajectory):
    matrix, reason = _trajectory_matrix(trajectory)
    if reason:
        return np.zeros((0, 0), dtype=float), reason
    if matrix.shape[0] < 2:
        return np.zeros((0, matrix.shape[1]), dtype=float), ""
    return np.diff(matrix, axis=0), ""


def _resolve_wrist_indices(
    trajectory,
    joint_count: int,
    config: TrajectoryScoreConfig,
):
    joint_names = list(getattr(trajectory, "joint_names", []) or [])
    if config.wrist_joint_names and joint_names:
        missing = [
            name for name in config.wrist_joint_names
            if name not in joint_names
        ]
        if missing:
            return (), f"missing wrist joint names: {missing}"
        indices = tuple(
            joint_names.index(name) for name in config.wrist_joint_names
        )
        return indices, ""

    indices = tuple(int(i) for i in config.wrist_joint_indices)
    invalid = [i for i in indices if i < 0 or i >= joint_count]
    if invalid:
        return (), (
            f"wrist joint indices out of range: {invalid}, "
            f"joint_count={joint_count}"
        )
    return indices, ""


def path_length(trajectory) -> float:
    """Return total joint-space path length."""
    deltas, reason = _joint_deltas(trajectory)
    if reason:
        return float("inf")
    if deltas.size == 0:
        return 0.0
    return float(np.linalg.norm(deltas, axis=1).sum())


def joint_subset_path_length(trajectory, joint_indices) -> float:
    """Return joint-space path length for a subset of joint indices."""
    deltas, reason = _joint_deltas(trajectory)
    if reason:
        return float("inf")
    if deltas.size == 0:
        return 0.0
    indices = tuple(int(i) for i in joint_indices)
    invalid = [i for i in indices if i < 0 or i >= deltas.shape[1]]
    if invalid:
        return float("inf")
    return float(np.linalg.norm(deltas[:, indices], axis=1).sum())


def score_trajectory(
    trajectory,
    config: TrajectoryScoreConfig = TrajectoryScoreConfig(),
) -> TrajectoryScore:
    matrix, reason = _trajectory_matrix(trajectory)
    num_points = len(getattr(trajectory, "points", []) or [])
    if reason:
        return TrajectoryScore(
            total_cost=float(config.invalid_cost),
            path_length=0.0,
            wrist_length=0.0,
            max_joint_step=0.0,
            smoothness=0.0,
            duration=0.0,
            num_points=num_points,
            valid=False,
            reason=reason,
        )

    if matrix.shape[0] < 2:
        duration = _duration_seconds(
            getattr(trajectory.points[-1], "time_from_start", None)
        )
        total_cost = float(config.duration_weight) * duration
        return TrajectoryScore(
            total_cost=total_cost,
            path_length=0.0,
            wrist_length=0.0,
            max_joint_step=0.0,
            smoothness=0.0,
            duration=duration,
            num_points=num_points,
        )

    deltas = np.diff(matrix, axis=0)
    total_length = float(np.linalg.norm(deltas, axis=1).sum())
    max_step = float(np.abs(deltas).max()) if deltas.size else 0.0
    smoothness = (
        float(np.linalg.norm(np.diff(deltas, axis=0), axis=1).sum())
        if deltas.shape[0] > 1
        else 0.0
    )
    duration = _duration_seconds(
        getattr(trajectory.points[-1], "time_from_start", None)
    )

    wrist_indices, wrist_reason = _resolve_wrist_indices(
        trajectory,
        matrix.shape[1],
        config,
    )
    if wrist_reason:
        return TrajectoryScore(
            total_cost=float(config.invalid_cost),
            path_length=total_length,
            wrist_length=0.0,
            max_joint_step=max_step,
            smoothness=smoothness,
            duration=duration,
            num_points=num_points,
            valid=False,
            reason=wrist_reason,
        )
    wrist_length = (
        float(np.linalg.norm(deltas[:, wrist_indices], axis=1).sum())
        if wrist_indices
        else 0.0
    )

    total_cost = (
        float(config.path_length_weight) * total_length
        + float(config.wrist_length_weight) * wrist_length
        + float(config.max_joint_step_weight) * max_step
        + float(config.smoothness_weight) * smoothness
        + float(config.duration_weight) * duration
    )
    return TrajectoryScore(
        total_cost=float(total_cost),
        path_length=total_length,
        wrist_length=wrist_length,
        max_joint_step=max_step,
        smoothness=smoothness,
        duration=duration,
        num_points=num_points,
    )


def rank_paths(
    paths: Sequence,
    config: TrajectoryScoreConfig = TrajectoryScoreConfig(),
):
    scored = [
        (traj, score_trajectory(traj, config))
        for traj in list(paths or [])
    ]
    return sorted(scored, key=lambda item: item[1].total_cost)


def select_best_path(
    paths,
    wrist_weight: float = 50.0,
    wrist_joint_indices=(2, 3, 4),
    wrist_joint_names=None,
    return_score: bool = False,
    config: Optional[TrajectoryScoreConfig] = None,
):
    if config is None:
        config = TrajectoryScoreConfig(
            wrist_length_weight=float(wrist_weight),
            wrist_joint_indices=tuple(int(i) for i in wrist_joint_indices),
            wrist_joint_names=(
                tuple(wrist_joint_names) if wrist_joint_names else None
            ),
        )
    ranked = rank_paths(paths, config)
    if not ranked:
        return (None, None) if return_score else None
    best_path, best_score = ranked[0]
    return (best_path, best_score) if return_score else best_path


__all__ = [
    "TrajectoryScore",
    "TrajectoryScoreConfig",
    "path_length",
    "joint_subset_path_length",
    "score_trajectory",
    "rank_paths",
    "select_best_path",
]
