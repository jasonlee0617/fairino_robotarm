"""Pure helpers for deterministic planning-benchmark goal sets and reports."""

import csv
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass

import numpy as np
import yaml


GOAL_FIELDS = (
    "goal_index", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg",
)
GOAL_COLLECTION_FORMAT_VERSION = 4


@dataclass(frozen=True)
class GoalSetSpec:
    """Identity and validation constraints for one reusable pose-goal set."""

    scene_name: str
    goal_mode: str
    goal_seed: int
    ik_mode: str
    obstacle_signature: str
    target_rpy_deg: tuple[float, float, float]
    repetitions: int
    min_separation_m: float
    clearance_min_m: float
    clearance_max_m: float
    corridor_clearance_max_m: float
    sampling_start_xyz: tuple[float, float, float]
    sampling_start_joints: tuple[float, float, float, float, float, float]
    start_id: str


def benchmark_slug(value):
    """Create a stable, filesystem-safe artifact name from a public planner ID."""
    text = str(value).strip()
    if text.endswith("*"):
        text = f"{text[:-1]}_star"
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in text)


def sampling_identity_payload(spec):
    """Return exactly the inputs that can change the accepted pose sequence."""
    return {
        "scene_name": spec.scene_name,
        "goal_mode": spec.goal_mode,
        "goal_seed": int(spec.goal_seed),
        "ik_mode": spec.ik_mode,
        "obstacle_signature": spec.obstacle_signature,
        "target_rpy_deg": [float(value) for value in spec.target_rpy_deg],
        "repetitions": int(spec.repetitions),
        "min_separation_m": float(spec.min_separation_m),
        "clearance_min_m": float(spec.clearance_min_m),
        "clearance_max_m": float(spec.clearance_max_m),
        "corridor_clearance_max_m": float(spec.corridor_clearance_max_m),
        "sampling_start_xyz": [float(value) for value in spec.sampling_start_xyz],
        "sampling_start_joints": [float(value) for value in spec.sampling_start_joints],
        "start_id": str(spec.start_id),
    }


def collection_key(spec):
    """Human-selected collection directory key for one start configuration."""
    start_id = str(spec.start_id).strip()
    if not start_id or not start_id.isascii() or not start_id.isalnum():
        raise ValueError("start_id must contain only ASCII letters and digits")
    return (
        f"goal_seed{int(spec.goal_seed)}_n{int(spec.repetitions)}_start_id_{start_id}"
    )


def goal_set_id(spec):
    return collection_key(spec)


def goal_set_signature(spec, scene_yaml_sha256, goal_set_csv_sha256):
    payload = {
        "format_version": GOAL_COLLECTION_FORMAT_VERSION,
        "collection_key": collection_key(spec),
        "sampling_identity": sampling_identity_payload(spec),
        "scene_yaml_sha256": str(scene_yaml_sha256),
        "goal_set_csv_sha256": str(goal_set_csv_sha256),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def goal_set_directory(collection_root, spec):
    return os.path.join(
        os.path.abspath(collection_root), benchmark_slug(spec.scene_name), goal_set_id(spec)
    )


def goal_set_paths(collection_root, spec):
    directory = goal_set_directory(collection_root, spec)
    return {
        "directory": directory,
        "goals": os.path.join(directory, "goal_set.csv"),
        "manifest": os.path.join(directory, "goal_set_manifest.yaml"),
        "sampling": os.path.join(directory, "goal_sampling_stats.yaml"),
    }


def obstacle_attr(obstacle, key, default=None):
    return obstacle.get(key, default) if isinstance(obstacle, dict) else getattr(obstacle, key, default)


def obstacle_center(obstacle):
    return tuple(float(value) for value in obstacle_attr(obstacle, "position", (0.0, 0.0, 0.0)))


def obstacle_half_extents(obstacle):
    shape = str(obstacle_attr(obstacle, "shape", "box")).lower()
    if shape == "box":
        return tuple(float(value) * 0.5 for value in obstacle_attr(obstacle, "size", (0.1, 0.1, 0.1)))
    radius = float(obstacle_attr(obstacle, "radius", 0.05))
    if shape == "sphere":
        return (radius, radius, radius)
    height = obstacle_attr(obstacle, "height", None)
    if height is None:
        raise ValueError(f"cylinder obstacle '{obstacle_attr(obstacle, 'name', '')}' missing height")
    return (radius, radius, 0.5 * float(height))


def obstacle_rpy_deg(obstacle):
    return tuple(float(value) for value in obstacle_attr(obstacle, "rpy_deg", (0.0, 0.0, 0.0)))


def _rotation_matrix(rpy_deg):
    roll, pitch, yaw = (math.radians(value) for value in rpy_deg)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return np.asarray((
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    ))


def obstacle_world_half_extents(obstacle):
    extents = np.asarray(obstacle_half_extents(obstacle), dtype=float)
    if str(obstacle_attr(obstacle, "shape", "box")).lower() != "box":
        return tuple(float(value) for value in extents)
    return tuple(float(value) for value in np.abs(_rotation_matrix(obstacle_rpy_deg(obstacle))) @ extents)


def obstacle_signature(obstacles):
    parts = []
    for obstacle in obstacles:
        parts.append({
            "name": str(obstacle_attr(obstacle, "name", "")),
            "shape": str(obstacle_attr(obstacle, "shape", "box")),
            "position": [round(value, 6) for value in obstacle_center(obstacle)],
            "half_extents": [round(value, 6) for value in obstacle_world_half_extents(obstacle)],
        })
        rotation = obstacle_rpy_deg(obstacle)
        if any(abs(value) > 1e-9 for value in rotation):
            parts[-1]["rpy_deg"] = [round(value, 6) for value in rotation]
    return hashlib.sha256(repr(sorted(parts, key=lambda item: item["name"])).encode()).hexdigest()


def distance_to_obstacle_surface(point_xyz, obstacles):
    point = np.asarray(point_xyz, dtype=float)
    distances = []
    for obstacle in obstacles:
        center = np.asarray(obstacle_center(obstacle), dtype=float)
        shape = str(obstacle_attr(obstacle, "shape", "box")).lower()
        extents = obstacle_half_extents(obstacle)
        if shape == "box":
            local_point = _rotation_matrix(obstacle_rpy_deg(obstacle)).T @ (point - center)
            distances.append(float(np.linalg.norm(np.maximum(np.abs(local_point) - extents, 0.0))))
        elif shape == "cylinder":
            radius, _, half_height = extents
            radial = max(0.0, float(np.linalg.norm(point[:2] - center[:2])) - radius)
            vertical = max(0.0, abs(point[2] - center[2]) - half_height)
            distances.append(float(np.hypot(radial, vertical)))
        else:
            distances.append(max(0.0, float(np.linalg.norm(point - center)) - extents[0]))
    return min(distances) if distances else float("inf")


def adaptive_challenge_metrics(point_xyz, start_xyz, obstacles, corridor_clearance_max_m):
    point = np.asarray(point_xyz, dtype=float)
    centers = [np.asarray(obstacle_center(item), dtype=float) for item in obstacles]
    angles = sorted(math.atan2(center[1] - point[1], center[0] - point[0]) for center in centers)
    if len(angles) >= 2:
        gaps = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)]
        gaps.append(2.0 * math.pi - angles[-1] + angles[0])
        angular_coverage = math.degrees(2.0 * math.pi - max(gaps))
    else:
        angular_coverage = 0.0
    vertical = sum(abs(center[2] - point[2]) > 0.03 for center in centers)
    clearance = distance_to_obstacle_surface(point_xyz, obstacles)
    corridor = min(
        distance_to_obstacle_surface(point * (1.0 - alpha) + np.asarray(start_xyz) * alpha, obstacles)
        for alpha in (0.25, 0.5, 0.75)
    )
    accepted = (
        len(centers) >= 3
        and vertical >= 2
        and angular_coverage >= 180.0
        and corridor <= corridor_clearance_max_m
    )
    return {
        "accepted": accepted,
        "inside_obstacle_hull": angular_coverage >= 180.0,
        "surrounding_obstacle_count": len(centers),
        "vertical_obstacle_count": vertical,
        "angular_coverage_deg": angular_coverage,
        "corridor_min_clearance_m": corridor,
        "endpoint_clearance_m": clearance,
    }


def goal_bounds(obstacles):
    centers = np.asarray([obstacle_center(item) for item in obstacles], dtype=float)
    extents = np.asarray([obstacle_world_half_extents(item) for item in obstacles], dtype=float)
    if not len(centers):
        raise ValueError("benchmark scene has no obstacles")
    return np.min(centers - extents, axis=0), np.max(centers + extents, axis=0)


def goal_is_separated(point_xyz, goals, min_separation_m):
    point = np.asarray(point_xyz)
    return not any(
        np.linalg.norm(point - np.asarray(other[0])) < min_separation_m
        for other in goals
    )


def iter_random_candidates(minimum, maximum, seed):
    """Yield deterministic uniform candidates without precomputing a pool."""
    rng = np.random.default_rng(seed)
    extent = np.asarray(maximum, dtype=float) - np.asarray(minimum, dtype=float)
    lower = np.asarray(minimum, dtype=float)
    while True:
        yield tuple(float(value) for value in lower + rng.random(3) * extent)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_goal_rows(path, goals):
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOAL_FIELDS)
        writer.writeheader()
        for index, (xyz, rpy) in enumerate(goals, 1):
            writer.writerow(dict(zip(GOAL_FIELDS, (index, *xyz, *rpy))))
    os.replace(path + ".tmp", path)


def _read_goal_rows(path, spec):
    goals = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != GOAL_FIELDS:
            raise ValueError("goal_set.csv 表头不符合当前归档格式")
        for expected_index, row in enumerate(reader, 1):
            if int(row["goal_index"]) != expected_index:
                raise ValueError("goal_set.csv 的目标索引不连续")
            xyz = tuple(float(row[key]) for key in ("x", "y", "z"))
            rpy = tuple(float(row[key]) for key in ("roll_deg", "pitch_deg", "yaw_deg"))
            if not np.allclose(rpy, spec.target_rpy_deg, atol=1e-6):
                raise ValueError("goal_set.csv 的姿态与当前采集条件不一致")
            if not goal_is_separated(xyz, goals, spec.min_separation_m):
                raise ValueError("goal_set.csv 不满足最小间距约束")
            goals.append((xyz, rpy))
    if len(goals) != spec.repetitions:
        raise ValueError("goal_set.csv 数量与采集配置不一致")
    return goals


def write_yaml_atomic(path, content):
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, allow_unicode=True, sort_keys=True)
    os.replace(path + ".tmp", path)


def build_goal_sampling_report(
    scene_name,
    goal_seed,
    requested_goals,
    attempts,
    accepted_goals,
    rejected,
    endpoint_clearances,
):
    clearances = [float(value) for value in endpoint_clearances]
    return {
        "scene_name": str(scene_name),
        "goal_seed": int(goal_seed),
        "requested_goals": int(requested_goals),
        "accepted_goals": int(accepted_goals),
        "candidate_attempts": int(attempts),
        "acceptance_rate": (
            float(accepted_goals) / float(attempts) if attempts else 0.0
        ),
        "rejected_geometry": int(rejected.get("geometry", 0)),
        "rejected_ik_geometry": int(rejected.get("ik_geometry", 0)),
        "rejected_ik_other": int(rejected.get("ik_other", 0)),
        "rejected_ik": int(
            rejected.get("ik", 0)
            + rejected.get("ik_geometry", 0)
            + rejected.get("ik_other", 0)
        ),
        "rejected_state": int(rejected.get("state", 0)),
        "rejected_separation": int(rejected.get("separation", 0)),
        "accepted_endpoint_clearance_min_m": min(clearances) if clearances else 0.0,
        "accepted_endpoint_clearance_max_m": max(clearances) if clearances else 0.0,
        "accepted_endpoint_clearance_mean_m": (
            statistics.fmean(clearances) if clearances else 0.0
        ),
        "accepted_endpoint_clearances_m": clearances,
    }


def write_goal_sampling_report(path, report):
    write_yaml_atomic(path, report)


def _load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_goal_collection(collection_root, spec):
    """Load exactly one complete, identity-matched goal collection."""
    paths = goal_set_paths(collection_root, spec)
    if not os.path.isdir(paths["directory"]):
        raise FileNotFoundError(f"goal collection not found: {paths['directory']}")
    if not all(os.path.isfile(paths[key]) for key in ("goals", "manifest", "sampling")):
        raise ValueError(f"goal collection is incomplete: {paths['directory']}")
    manifest = _load_yaml(paths["manifest"])
    if manifest.get("format_version") != GOAL_COLLECTION_FORMAT_VERSION:
        raise ValueError(
            f"legacy goal collection unsupported: expected format_version={GOAL_COLLECTION_FORMAT_VERSION}"
        )
    if manifest.get("collection_key") != collection_key(spec):
        raise ValueError("goal collection key 与当前采集条件不一致")
    if manifest.get("goal_set_id") != goal_set_id(spec):
        raise ValueError("goal collection ID 与当前采集条件不一致")
    if manifest.get("sampling_identity") != sampling_identity_payload(spec):
        raise ValueError("goal collection manifest 与当前采集条件不一致; 请更换 start_id")
    goals_sha = _sha256_file(paths["goals"])
    if manifest.get("goal_set_csv_sha256") != goals_sha:
        raise ValueError("goal collection CSV SHA256 校验失败")
    expected_signature = goal_set_signature(spec, manifest.get("scene_yaml_sha256", ""), goals_sha)
    if manifest.get("goal_set_signature_sha256") != expected_signature:
        raise ValueError("goal collection signature SHA256 校验失败")
    return paths, _read_goal_rows(paths["goals"], spec), manifest


def write_goal_collection(collection_root, spec, goals, scene_yaml_sha256, sampling_report):
    """Atomically materialize a collection once; complete matching sets are reused."""
    try:
        paths, loaded_goals, manifest = load_goal_collection(collection_root, spec)
        return paths, loaded_goals, manifest, True
    except FileNotFoundError:
        pass
    paths = goal_set_paths(collection_root, spec)
    parent = os.path.dirname(paths["directory"])
    os.makedirs(parent, exist_ok=True)
    staging = f"{paths['directory']}.tmp-{os.getpid()}"
    if os.path.exists(staging):
        raise RuntimeError(f"goal collection staging path already exists: {staging}")
    os.makedirs(staging)
    staging_paths = {
        key: os.path.join(staging, os.path.basename(value))
        for key, value in paths.items() if key != "directory"
    }
    _write_goal_rows(staging_paths["goals"], goals)
    manifest = {
        "format_version": GOAL_COLLECTION_FORMAT_VERSION,
        "collection_key": collection_key(spec),
        "goal_set_id": goal_set_id(spec),
        "sampling_identity": sampling_identity_payload(spec),
        "scene_yaml_sha256": str(scene_yaml_sha256),
        "goal_set_csv_sha256": _sha256_file(staging_paths["goals"]),
        "goal_set_signature_sha256": goal_set_signature(
            spec, scene_yaml_sha256, _sha256_file(staging_paths["goals"])),
        "files": {"goals": "goal_set.csv", "sampling": "goal_sampling_stats.yaml"},
    }
    report = dict(sampling_report)
    report["format_version"] = GOAL_COLLECTION_FORMAT_VERSION
    report["collection_key"] = manifest["collection_key"]
    report["goal_set_id"] = manifest["goal_set_id"]
    write_goal_sampling_report(staging_paths["sampling"], report)
    write_yaml_atomic(staging_paths["manifest"], manifest)
    os.replace(staging, paths["directory"])
    return paths, list(goals), manifest, False


def prepare_benchmark_run(
    output_root, scene_name, collection_id, planner_id, planner_seed,
    goal_root_mode, manifest, stamp, variant="full",
):
    if not output_root:
        raise RuntimeError("benchmark_output_dir is required")
    root = os.path.abspath(output_root)
    mode = "sr" if goal_root_mode == "single_root" else "mr"
    planner_parts = [benchmark_slug(planner_id)]
    if variant != "full":
        planner_parts.append(benchmark_slug(variant))
    run_dir = os.path.join(
        root, benchmark_slug(scene_name), collection_id, mode, *planner_parts,
        f"planner_seed{int(planner_seed):02d}", stamp,
    )
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(
            root, benchmark_slug(scene_name), collection_id, mode, *planner_parts,
            f"planner_seed{int(planner_seed):02d}", f"{stamp}_{suffix}",
        )
        suffix += 1
    os.makedirs(run_dir)
    write_yaml_atomic(os.path.join(run_dir, "run_manifest.yaml"), manifest)
    initialize_standard_csvs(run_dir)
    return run_dir


COMMON_KEY_FIELDS = (
    "scene_name", "goal_set_id", "goal_set_signature_sha256", "goal_index",
    "goal_root_mode", "planner_id", "planner_seed", "variant",
    "run_signature_sha256",
)

RESULT_FIELDS = COMMON_KEY_FIELDS + (
    "plan_success", "failure_stage", "failure_code", "stop_reason",
    "core_planning_time_s", "first_solution_time_s",
    "first_solution_path_cost_rad", "joint_path_length_rad", "tcp_path_length_m",
    "joint_turn_total_variation_rad", "waypoint_count",
    "planner_sample_attempts", "planner_accepted_samples", "planner_iterations",
    "planner_work_units", "planner_nodes", "planner_edges",
    "post_solution_sample_attempts", "post_solution_budget_complete",
    "collision_state_checks", "collision_motion_checks", "valid_motion_edges",
    "invalid_motion_edges", "ik_time_s", "root_generation_time_s", "search_time_s",
    "final_validation_time_s", "trajectory_construction_time_s",
    "goal_root_count", "selected_goal_root",
)

ANYTIME_FIELDS = COMMON_KEY_FIELDS + (
    "checkpoint_s", "has_solution", "incumbent_joint_cost_rad",
    "cumulative_iterations", "cumulative_sample_attempts",
    "cumulative_accepted_samples", "cumulative_nodes", "cumulative_work_units",
)

ROOT_DIAGNOSTIC_FIELDS = COMMON_KEY_FIELDS + (
    "root_index", "q1", "q2", "q3", "q4", "q5", "q6",
    "passed_hard_filter", "filter_reason", "total_cost",
    "assigned_sample_attempts", "accepted_samples", "path_improvements",
    "selected_final",
)

ALGORITHM_DIAGNOSTIC_FIELDS = COMMON_KEY_FIELDS + (
    "metric_name", "metric_value", "metric_text", "ablation_variant",
)

TRAJECTORY_PATH_FIELDS = COMMON_KEY_FIELDS + (
    "path_stage", "waypoint_index", "q1", "q2", "q3", "q4", "q5", "q6",
    "tcp_x", "tcp_y", "tcp_z", "selected_goal_root",
)

STANDARD_CSV_SCHEMAS = {
    "results.csv": RESULT_FIELDS,
    "anytime_trace.csv": ANYTIME_FIELDS,
    "root_diagnostics.csv": ROOT_DIAGNOSTIC_FIELDS,
    "algorithm_diagnostics.csv": ALGORITHM_DIAGNOSTIC_FIELDS,
    "trajectory_paths.csv": TRAJECTORY_PATH_FIELDS,
}


def canonical_sha256(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv_atomic(path, rows, fieldnames):
    rows = list(rows)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_results(path, rows):
    write_csv_atomic(path, rows, RESULT_FIELDS)


def initialize_standard_csvs(run_dir):
    for filename, fields in STANDARD_CSV_SCHEMAS.items():
        path = os.path.join(run_dir, filename)
        if not os.path.exists(path):
            write_csv_atomic(path, (), fields)


def csv_row_count(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def finalize_run_manifest(run_dir, status, expected_goals, reason=""):
    manifest_path = os.path.join(run_dir, "run_manifest.yaml")
    manifest = _load_yaml(manifest_path)
    files = {}
    for filename in STANDARD_CSV_SCHEMAS:
        path = os.path.join(run_dir, filename)
        if os.path.isfile(path):
            files[filename] = {
                "sha256": _sha256_file(path),
                "rows": csv_row_count(path),
            }
    completed_goals = files.get("results.csv", {}).get("rows", 0)
    manifest.update({
        "status": str(status),
        "expected_goals": int(expected_goals),
        "completed_goals": int(completed_goals),
        "failure_reason": str(reason),
        "files": files,
    })
    write_yaml_atomic(manifest_path, manifest)
    return manifest


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_numbers(rows, field):
    values = []
    for row in rows:
        try:
            value = float(row.get(field, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _summary_stat(rows, field, include_p95=False):
    values = _summary_numbers(rows, field)
    if not values:
        return "N/A"
    summary = f"mean={statistics.fmean(values):.6f}, median={statistics.median(values):.6f}"
    if include_p95:
        values.sort()
        p95_index = math.ceil(0.95 * len(values)) - 1
        summary += f", p95={values[p95_index]:.6f}"
    return f"{summary}, n={len(values)}"


def write_benchmark_summary(run_dir):
    """Write an auditable Markdown summary for one benchmark launch."""
    manifest_path = os.path.join(run_dir, "run_manifest.yaml")
    manifest = _load_yaml(manifest_path)
    results_path = os.path.join(run_dir, "results.csv")
    rows = _read_csv(results_path) if os.path.isfile(results_path) else []
    succeeded = [row for row in rows if str(row.get("plan_success", "")).strip().lower() == "true"]
    expected = manifest.get("expected_goals", "N/A")
    completed = manifest.get("completed_goals", len(rows))
    success_rate = "N/A" if not rows else f"{len(succeeded) / len(rows):.2%}"
    lines = [
        "# Planning Benchmark Summary", "",
        "## Run identity", "",
        f"- Status: {manifest.get('status', 'missing')}",
        f"- Scene: {manifest.get('scene_name', 'N/A')}",
        f"- Goal set: {manifest.get('goal_set_id', 'N/A')}",
        f"- Root mode: {manifest.get('goal_root_mode', 'N/A')}",
        f"- Planner: {manifest.get('planner_id', 'N/A')}",
        f"- Planner seed: {manifest.get('planner_random_seed', 'N/A')}",
        f"- Variant: {manifest.get('variant', 'full')}", "",
        "## Outcome", "",
        f"- Expected goals: {expected}",
        f"- Completed goals: {completed}",
        f"- Successful goals: {len(succeeded)}",
        f"- Success rate: {success_rate}",
        f"- Failure reason: {manifest.get('failure_reason') or 'N/A'}", "",
        "## Timing and successful-path metrics", "",
        f"- Core planning time (s): {_summary_stat(rows, 'core_planning_time_s')}",
        f"- First solution time (s): {_summary_stat(succeeded, 'first_solution_time_s')}",
        f"- Joint path length (rad): {_summary_stat(succeeded, 'joint_path_length_rad')}",
        f"- TCP path length (m): {_summary_stat(succeeded, 'tcp_path_length_m')}",
        f"- Joint turn variation (rad): {_summary_stat(succeeded, 'joint_turn_total_variation_rad')}", "",
        "## Planning resource metrics", "",
        f"- Accepted samples: {_summary_stat(rows, 'planner_accepted_samples', include_p95=True)}",
        f"- Iterations: {_summary_stat(rows, 'planner_iterations', include_p95=True)}", "",
        "## Standard artifacts", "",
    ]
    for filename in STANDARD_CSV_SCHEMAS:
        info = manifest.get('files', {}).get(filename, {})
        lines.append(f"- {filename}: rows={info.get('rows', 'N/A')}")
    path = os.path.join(run_dir, "summary.md")
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(temporary, path)
    manifest["summary"] = {"file": "summary.md", "sha256": _sha256_file(path)}
    write_yaml_atomic(manifest_path, manifest)
    return path

def validate_complete_run(run_dir, expected_goals=30, checkpoints=(0.1, 0.2, 0.5, 1, 2, 5, 10, 15)):
    manifest_path = os.path.join(run_dir, "run_manifest.yaml")
    if not os.path.isfile(manifest_path):
        return False, "missing run_manifest.yaml"
    manifest = _load_yaml(manifest_path)
    if manifest.get("status") != "complete":
        current_status = manifest.get("status", "missing")
        return False, f"manifest status is {current_status}"
    try:
        manifest_expected = int(manifest.get("expected_goals", -1))
        manifest_completed = int(manifest.get("completed_goals", -1))
    except (TypeError, ValueError):
        return False, "manifest goal counts are invalid"
    if manifest_expected != int(expected_goals) or manifest_completed != int(expected_goals):
        return False, "manifest goal counts do not match the expected complete run"
    for filename, fields in STANDARD_CSV_SCHEMAS.items():
        path = os.path.join(run_dir, filename)
        if not os.path.isfile(path):
            return False, f"missing {filename}"
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                return False, f"schema mismatch in {filename}"
    results = _read_csv(os.path.join(run_dir, "results.csv"))
    expected = list(range(1, int(expected_goals) + 1))
    try:
        actual = sorted(int(row["goal_index"]) for row in results)
    except (KeyError, TypeError, ValueError):
        return False, "invalid goal_index in results.csv"
    if actual != expected:
        return False, "results.csv does not contain exactly the expected goal indices"
    resource_fields = (
        "core_planning_time_s", "planner_sample_attempts", "planner_accepted_samples",
        "planner_iterations", "planner_work_units", "planner_nodes", "planner_edges",
        "collision_state_checks", "collision_motion_checks", "valid_motion_edges",
        "invalid_motion_edges", "ik_time_s", "root_generation_time_s", "search_time_s",
        "final_validation_time_s", "trajectory_construction_time_s", "goal_root_count",
    )
    path_fields = (
        "first_solution_time_s", "first_solution_path_cost_rad",
        "joint_path_length_rad", "tcp_path_length_m", "joint_turn_total_variation_rad",
    )
    for row in results:
        success_text = str(row.get("plan_success", "")).lower()
        if success_text not in ("true", "false"):
            return False, "invalid plan_success in results.csv"
        try:
            resources = [float(row[field]) for field in resource_fields]
        except (KeyError, TypeError, ValueError):
            return False, "invalid resource metric in results.csv"
        if any(not math.isfinite(value) or value < 0 for value in resources):
            return False, "non-finite or negative resource metric in results.csv"
        try:
            path_values = [float(row[field]) for field in path_fields]
            waypoint_count = float(row["waypoint_count"])
        except (KeyError, TypeError, ValueError):
            return False, "invalid path metric in results.csv"
        if success_text == "true":
            if any(not math.isfinite(value) or value < 0 for value in path_values):
                return False, "successful goal has invalid path metric"
            if (path_values[0] > float(checkpoints[-1]) + 1e-9 or waypoint_count < 2 or
                    waypoint_count != int(waypoint_count)):
                return False, "successful goal violates deadline or waypoint contract"
            try:
                selected_root = int(float(row["selected_goal_root"]))
                root_count = int(float(row["goal_root_count"]))
            except (KeyError, TypeError, ValueError):
                return False, "successful goal has invalid root selection"
            if selected_root < 0 or root_count < 1:
                return False, "successful goal has invalid root selection"
        elif any(math.isfinite(value) for value in path_values) or waypoint_count != 0:
            return False, "failed goal contains success-only path metrics"
    key_fields = tuple(field for field in COMMON_KEY_FIELDS if field != "goal_index")
    result_keys = {
        tuple(row.get(field, "") for field in key_fields)
        for row in results
    }
    if len(result_keys) != 1:
        return False, "results.csv contains inconsistent composite keys"
    expected_key = next(iter(result_keys))
    manifest_key = {
        "scene_name": manifest.get("scene_name", ""),
        "goal_set_id": manifest.get("goal_set_id", ""),
        "goal_set_signature_sha256": manifest.get("goal_set_signature_sha256", ""),
        "goal_root_mode": manifest.get("goal_root_mode", ""),
        "planner_id": manifest.get("planner_id", ""),
        "planner_seed": str(manifest.get("planner_random_seed", "")),
        "variant": manifest.get("variant", "full"),
        "run_signature_sha256": manifest.get("run_signature_sha256", ""),
    }
    for index, field in enumerate(key_fields):
        if str(expected_key[index]) != str(manifest_key[field]):
            return False, f"manifest/composite-key mismatch for {field}"

    def validate_sidecar(filename, rows, require_goal_coverage=True):
        if require_goal_coverage:
            try:
                covered = sorted(set(int(row["goal_index"]) for row in rows))
            except (KeyError, TypeError, ValueError):
                return False, f"invalid goal_index in {filename}"
            if covered != expected:
                return False, f"{filename} does not cover every goal"
        for row in rows:
            row_key = tuple(row.get(field, "") for field in key_fields)
            if row_key != expected_key:
                return False, f"composite-key mismatch in {filename}"
        return True, ""

    trace = _read_csv(os.path.join(run_dir, "anytime_trace.csv"))
    expected_trace_rows = int(expected_goals) * len(tuple(checkpoints))
    if len(trace) != expected_trace_rows:
        return False, f"anytime_trace.csv rows={len(trace)} expected={expected_trace_rows}"
    valid, reason = validate_sidecar("anytime_trace.csv", trace)
    if not valid:
        return valid, reason
    expected_checkpoints = sorted(float(value) for value in checkpoints)
    for goal_index in expected:
        goal_trace = [row for row in trace if int(row["goal_index"]) == goal_index]
        try:
            actual_checkpoints = sorted(float(row["checkpoint_s"]) for row in goal_trace)
        except (TypeError, ValueError):
            return False, f"anytime_trace.csv has invalid checkpoint for goal {goal_index}"
        if actual_checkpoints != expected_checkpoints:
            return False, f"anytime_trace.csv checkpoints mismatch for goal {goal_index}"
        cumulative_fields = (
            "cumulative_iterations", "cumulative_sample_attempts",
            "cumulative_accepted_samples", "cumulative_nodes", "cumulative_work_units",
        )
        for row in goal_trace:
            has_solution = str(row.get("has_solution", "")).lower()
            if has_solution not in ("true", "false"):
                return False, f"anytime_trace.csv has invalid has_solution for goal {goal_index}"
            try:
                cumulative = [float(row[field]) for field in cumulative_fields]
            except (KeyError, TypeError, ValueError):
                return False, f"anytime_trace.csv has invalid work metric for goal {goal_index}"
            if any(not math.isfinite(value) or value < 0 for value in cumulative):
                return False, f"anytime_trace.csv has invalid work metric for goal {goal_index}"
            if has_solution == "true":
                try:
                    incumbent = float(row["incumbent_joint_cost_rad"])
                except (KeyError, TypeError, ValueError):
                    return False, f"anytime_trace.csv has invalid incumbent for goal {goal_index}"
                if not math.isfinite(incumbent) or incumbent < 0:
                    return False, f"anytime_trace.csv has invalid incumbent for goal {goal_index}"
    roots = _read_csv(os.path.join(run_dir, "root_diagnostics.csv"))
    valid, reason = validate_sidecar("root_diagnostics.csv", roots)
    if not valid:
        return valid, reason
    success_by_goal = {
        int(row["goal_index"]): str(row.get("plan_success", "")).lower() == "true"
        for row in results
    }
    selected_by_goal = {
        int(row["goal_index"]): int(float(row["selected_goal_root"]))
        for row in results if success_by_goal[int(row["goal_index"])]
    }
    for goal_index in expected:
        goal_roots = [row for row in roots if int(row["goal_index"]) == goal_index]
        selected_count = 0
        selected_indices = []
        for row in goal_roots:
            passed = str(row.get("passed_hard_filter", "")).lower()
            selected = str(row.get("selected_final", "")).lower()
            if passed not in ("true", "false") or selected not in ("true", "false"):
                return False, f"root_diagnostics.csv has invalid boolean for goal {goal_index}"
            selected_count += int(selected == "true")
            if selected == "true":
                try:
                    selected_indices.append(int(float(row["root_index"])))
                except (KeyError, TypeError, ValueError):
                    return False, f"root_diagnostics.csv has invalid selected root for goal {goal_index}"
            try:
                allocation = [
                    float(row[field]) for field in
                    ("assigned_sample_attempts", "accepted_samples", "path_improvements")
                ]
            except (KeyError, TypeError, ValueError):
                return False, f"root_diagnostics.csv has invalid work metric for goal {goal_index}"
            if any(not math.isfinite(value) or value < 0 for value in allocation):
                return False, f"root_diagnostics.csv has invalid work metric for goal {goal_index}"
            if passed == "true":
                try:
                    joints = [float(row[f"q{joint}"]) for joint in range(1, 7)]
                except (KeyError, TypeError, ValueError):
                    return False, f"root_diagnostics.csv has invalid joint state for goal {goal_index}"
                if any(not math.isfinite(value) for value in joints):
                    return False, f"root_diagnostics.csv has invalid joint state for goal {goal_index}"
        if selected_count > 1 or (success_by_goal[goal_index] and selected_count != 1):
            return False, f"root_diagnostics.csv selected root mismatch for goal {goal_index}"
        if success_by_goal[goal_index] and selected_indices[0] != selected_by_goal[goal_index]:
            return False, f"results.csv selected root does not match diagnostics for goal {goal_index}"
    diagnostics = _read_csv(os.path.join(run_dir, "algorithm_diagnostics.csv"))
    valid, reason = validate_sidecar("algorithm_diagnostics.csv", diagnostics)
    if not valid:
        return valid, reason
    for row in diagnostics:
        if not str(row.get("metric_name", "")).strip():
            return False, "algorithm_diagnostics.csv has an empty metric_name"
        metric_text = str(row.get("metric_text", "")).strip()
        try:
            metric_value = float(row.get("metric_value", ""))
        except (TypeError, ValueError):
            metric_value = math.nan
        if not math.isfinite(metric_value) and not metric_text:
            return False, "algorithm_diagnostics.csv has no metric value or text"
    trajectories = _read_csv(os.path.join(run_dir, "trajectory_paths.csv"))
    valid, reason = validate_sidecar(
        "trajectory_paths.csv", trajectories, require_goal_coverage=False)
    if not valid:
        return valid, reason
    successful_goals = {
        int(row["goal_index"]) for row in results
        if str(row.get("plan_success", "")).lower() == "true"
    }
    trajectory_goals = {int(row["goal_index"]) for row in trajectories}
    if trajectory_goals != successful_goals:
        return False, "trajectory_paths.csv coverage does not match successful goals"
    for goal_index in successful_goals:
        goal_paths = [row for row in trajectories if int(row["goal_index"]) == goal_index]
        stages = {str(row.get("path_stage", "")) for row in goal_paths}
        if stages != {"raw", "final"}:
            return False, f"trajectory_paths.csv stages mismatch for goal {goal_index}"
        for stage in ("raw", "final"):
            stage_rows = [row for row in goal_paths if row.get("path_stage") == stage]
            try:
                waypoint_indices = sorted(int(row["waypoint_index"]) for row in stage_rows)
            except (KeyError, TypeError, ValueError):
                return False, f"trajectory_paths.csv has invalid waypoint index for goal {goal_index}"
            if waypoint_indices != list(range(len(stage_rows))) or len(stage_rows) < 2:
                return False, f"trajectory_paths.csv waypoints are incomplete for goal {goal_index}"
            for row in stage_rows:
                try:
                    coordinates = [float(row[f"q{joint}"]) for joint in range(1, 7)]
                    coordinates.extend(float(row[field]) for field in ("tcp_x", "tcp_y", "tcp_z"))
                except (KeyError, TypeError, ValueError):
                    return False, f"trajectory_paths.csv has invalid coordinates for goal {goal_index}"
                if any(not math.isfinite(value) for value in coordinates):
                    return False, f"trajectory_paths.csv has invalid coordinates for goal {goal_index}"
    file_records = manifest.get("files", {})
    for filename in STANDARD_CSV_SCHEMAS:
        record = file_records.get(filename, {})
        path = os.path.join(run_dir, filename)
        if record.get("sha256") != _sha256_file(path) or record.get("rows") != csv_row_count(path):
            return False, f"manifest hash or row count mismatch for {filename}"
    return True, "complete"
