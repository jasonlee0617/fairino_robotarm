"""Load the single source of truth for position-servo launch parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from myrobot_common.launch_utils.yaml_loader import flatten_moveit_parameters


_CONTROLLER_TYPES = {"PID", "PD", "PI_FF", "ADAPTIVE_PID", "LADRC", "NLADRC", "MPC"}


def config_path() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("visual_servo_bringup")) / "config" / "visual_position_servo_params.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or config_path()).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise RuntimeError("visual_position_servo_params.yaml must contain a mapping")
    return config


def _flat_params(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            result.update(_flat_params(value, key))
        else:
            result[key] = value
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def sim_camera_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    environments = _mapping(config.get("environments"), "environments")
    sim = _mapping(environments.get("sim"), "environments.sim")
    launch = _mapping(sim.get("launch"), "environments.sim.launch")
    return _flat_params(_mapping(launch.get("camera"), "environments.sim.launch.camera"))


def sim_target_motion_parameters(
    perception_source: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the active simulation target-motion node parameters from YAML."""
    config = config or load_config()
    environments = _mapping(config.get("environments"), "environments")
    sim = _mapping(environments.get("sim"), "environments.sim")
    launch = _mapping(sim.get("launch"), "environments.sim.launch")
    target = {"yolo_kalman": "cube", "aruco": "aruco"}.get(str(perception_source).strip().lower())
    if target is None:
        raise ValueError("perception_source must be 'yolo_kalman' or 'aruco'")
    motions = _mapping(launch.get("target_motion"), "environments.sim.launch.target_motion")
    return _flat_params(_mapping(motions.get(target), f"environments.sim.launch.target_motion.{target}"))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for name, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = _deep_merge(result[name], value)
        else:
            result[name] = value
    return result


def visual_servo_parameters(environment: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    if environment not in {"sim", "real"}:
        raise ValueError("environment must be 'sim' or 'real'")
    config = config or load_config()
    common = _mapping(
        _mapping(_mapping(config.get("common"), "common").get("nodes"), "common.nodes").get("visual_position_servo"),
        "common.nodes.visual_position_servo",
    )
    environments = _mapping(config.get("environments"), "environments")
    selected = _mapping(
        _mapping(_mapping(environments.get(environment), f"environments.{environment}").get("nodes"), f"environments.{environment}.nodes").get("visual_position_servo"),
        f"environments.{environment}.nodes.visual_position_servo",
    )
    result = sim_camera_defaults(config) if environment == "sim" else {}
    merged = _deep_merge(common, selected)
    for name in ("task", "planning", "runtime"):
        result.update(_flat_params(_mapping(merged.get(name), f"nodes.visual_position_servo.{name}")))
    perception = _mapping(merged.get("perception"), "common.nodes.visual_position_servo.perception")
    active_source = str(perception.get("active_source", "")).strip().lower()
    if active_source not in {"yolo_kalman", "aruco"}:
        raise RuntimeError(f"Unsupported position-servo perception source: {active_source}")
    result["perception_source"] = active_source
    result.update(aruco_parameters(config))
    result.update(flatten_moveit_parameters(_mapping(merged.get("moveit"), "common.nodes.visual_position_servo.moveit")))

    controllers = _mapping(merged.get("controllers"), "common.nodes.visual_position_servo.controllers")
    active = str(controllers.get("active", "")).strip().upper()
    profiles = _mapping(controllers.get("profiles"), "nodes.visual_position_servo.controllers.profiles")
    if active not in _CONTROLLER_TYPES or active not in profiles:
        raise RuntimeError(f"Unsupported position-servo controller: {active}")
    result["servo_controller_type"] = active
    for section in _mapping(profiles[active], f"controllers.profiles.{active}").values():
        result.update(_flat_params(_mapping(section, f"controllers.profiles.{active}")))
    return result


def yolo_kalman_parameters(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    node = _mapping(
        _mapping(
            _mapping(
                _mapping(config.get("common"), "common").get("nodes"), "common.nodes"
            ).get("visual_position_servo"),
            "common.nodes.visual_position_servo",
        ).get("perception"),
        "common.nodes.visual_position_servo.perception",
    )
    return _flat_params(_mapping(node.get("yolo_kalman"), "perception.yolo_kalman"))


def aruco_parameters(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the shared ArUco source parameters in existing ROS names."""
    config = config or load_config()
    perception = _mapping(
        _mapping(
            _mapping(
                _mapping(config.get("common"), "common").get("nodes"), "common.nodes"
            ).get("visual_position_servo"),
            "common.nodes.visual_position_servo",
        ).get("perception"),
        "common.nodes.visual_position_servo.perception",
    )
    aruco = _mapping(perception.get("aruco"), "perception.aruco")
    detection = _mapping(aruco.get("detection"), "perception.aruco.detection")
    detector = _mapping(detection.get("detector"), "perception.aruco.detection.detector")
    position_source = _mapping(aruco.get("position_source"), "perception.aruco.position_source")
    visualization = _mapping(aruco.get("visualization"), "perception.aruco.visualization")
    tracking = _mapping(aruco.get("tracking"), "perception.aruco.tracking")
    return {
        "aruco_marker_size_m": detection["marker_size_m"],
        "aruco_dictionary": detection["dictionary"],
        "aruco_image_topic": detection["image_topic"],
        "aruco_camera_info_topic": detection["camera_info_topic"],
        "aruco_camera_frame": detection["camera_frame"],
        "aruco_marker_id": position_source["marker_id"],
        "aruco_markers_topic": position_source["input_topic"],
        "aruco_marker_pose_topic": position_source["output_topic"],
        "aruco_visualization_image_topic": visualization["image_topic"],
        "aruco_visualization_marker_id": visualization["marker_id"],
        "aruco_prediction_hold_sec": tracking["prediction_hold_sec"],
        "aruco_motion_auto_start_topic": aruco["motion_auto_start_topic"],
        **{f"aruco_{name}": value for name, value in detector.items()},
    }


def aruco_detector_parameters(parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map shared ArUco configuration to ros2_aruco's parameter names."""
    params = parameters or aruco_parameters()
    names = {
        "marker_size": "aruco_marker_size_m",
        "aruco_dictionary_id": "aruco_dictionary",
        "image_topic": "aruco_image_topic",
        "camera_info_topic": "aruco_camera_info_topic",
        "camera_frame": "aruco_camera_frame",
        "visualization_image_topic": "aruco_visualization_image_topic",
        "visualization_marker_id": "aruco_visualization_marker_id",
        "adaptive_thresh_win_size_min": "aruco_adaptive_thresh_win_size_min",
        "adaptive_thresh_win_size_max": "aruco_adaptive_thresh_win_size_max",
        "adaptive_thresh_win_size_step": "aruco_adaptive_thresh_win_size_step",
        "adaptive_thresh_constant": "aruco_adaptive_thresh_constant",
        "min_marker_perimeter_rate": "aruco_min_marker_perimeter_rate",
        "max_marker_perimeter_rate": "aruco_max_marker_perimeter_rate",
        "polygonal_approx_accuracy_rate": "aruco_polygonal_approx_accuracy_rate",
        "corner_refinement_method": "aruco_corner_refinement_method",
        "corner_refinement_win_size": "aruco_corner_refinement_win_size",
        "corner_refinement_max_iterations": "aruco_corner_refinement_max_iterations",
        "corner_refinement_min_accuracy": "aruco_corner_refinement_min_accuracy",
    }
    return {target: params[source] for target, source in names.items()}


def aruco_pose_source_parameters(parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map shared ArUco configuration to the selected-pose publisher."""
    params = parameters or aruco_parameters()
    return {
        "marker_id": params["aruco_marker_id"],
        "aruco_topic": params["aruco_markers_topic"],
        "output_topic": params["aruco_marker_pose_topic"],
    }
