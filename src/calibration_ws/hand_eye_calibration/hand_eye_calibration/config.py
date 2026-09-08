"""Typed configuration for the fixed-joint Fairino collector.

The quality and solve parameters intentionally mirror the WVCSC simulation
collector.  Robot frames, topics and fixed Fairino joint rows stay local.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory

_DEFAULT_JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
INITIAL_JOINT_DEG = (-108.48, -95.742, 82.245, -77.492, -90.86, -19.254)
JOINT_WAYPOINT_SLOTS = 20


def flatten_ros_parameters(parameters: dict) -> dict:
    """Flatten grouped YAML defaults before handing them to ROS parameters."""
    if not isinstance(parameters, dict):
        raise ValueError("ros__parameters must be a mapping")

    flattened: Dict[str, object] = {}

    def visit(values: dict) -> None:
        for key, value in values.items():
            if not isinstance(key, str):
                raise ValueError("parameter names must be strings")
            if isinstance(value, dict):
                visit(value)
                continue
            if key in flattened:
                raise ValueError(f"duplicate grouped parameter: {key}")
            flattened[key] = value

    visit(parameters)
    return flattened


class CalibrationType(str, Enum):
    EYE_IN_HAND = "eye_in_hand"
    EYE_ON_BASE = "eye_on_base"


def normalize_calibration_type(value) -> CalibrationType:
    if isinstance(value, CalibrationType):
        return value
    text = str(value).strip().lower().replace("-", "_")
    try:
        return CalibrationType(text)
    except ValueError as exc:
        raise ValueError("calibration_type must be eye_in_hand or eye_on_base") from exc


@dataclass(frozen=True)
class JointWaypointSpec:
    """One configured fixed joint pose, retained locally even when other slots are TODO."""

    index: int
    label: str
    joints_deg: Tuple[float, float, float, float, float, float]

    @property
    def joints_rad(self) -> Tuple[float, float, float, float, float, float]:
        return tuple(math.radians(value) for value in self.joints_deg)  # type: ignore[return-value]


@dataclass(frozen=True)
class CollectorFramesConfig:
    calibration_type: CalibrationType
    base_frame: str
    ee_frame: str
    tracking_base_frame: str
    tracking_marker_frame: str
    marker_id: int
    image_topic: str
    aruco_dictionary_id: str
    camera_info_topic: str
    camera_intrinsics_source: str


@dataclass(frozen=True)
class CollectorMotionConfig:
    move_group_name: str
    move_group_ns_fairino: str
    planning_pipeline_id: str
    planner_id: str
    joint_names: Tuple[str, ...]
    max_velocity: float
    max_acceleration: float
    allowed_planning_time: float
    max_step_size: float
    position_tolerance: float
    orientation_tolerance: float
    allowed_start_tolerance: float
    action_delay: float
    settle_time_sec: float
    keyboard_poll_period: float
    start_wait_poll_period: float
    step_between_actions: bool


@dataclass(frozen=True)
class CollectorSamplingConfig:
    marker_size_m: float
    marker_distance_min_m: float
    marker_distance_max_m: float
    stable_frames: int
    stable_marker_timeout_sec: float
    minimum_corner_margin_px: float
    minimum_marker_side_px: float
    maximum_center_std_px: float
    maximum_marker_depth_std_m: float
    maximum_marker_angle_std_deg: float
    joint_stationary_max_position_delta_rad: float
    joint_stationary_window_sec: float
    joint_stationary_timeout_sec: float
    stable_tf_settle_sec: float
    minimum_samples: int
    minimum_solution_samples: int
    minimum_translation_delta_m: float
    minimum_rotation_delta_deg: float
    minimum_translation_span_m: float
    minimum_rotation_span_deg: float
    algorithm_names: Tuple[str, ...]
    maximum_algorithm_translation_delta_m: float
    maximum_algorithm_rotation_delta_deg: float
    maximum_camera_translation_norm_m: float
    maximum_eye_on_base_camera_translation_norm_m: float
    maximum_marker_position_rms_m: float
    maximum_marker_rotation_rms_deg: float
    fixed_marker_refinement_translation_sigma_m: float
    fixed_marker_refinement_rotation_sigma_deg: float
    fixed_marker_refinement_max_iterations: int
    ground_truth_check_enabled: bool
    ground_truth_max_translation_error_m: float
    ground_truth_max_axis_error_m: float
    ground_truth_max_rotation_error_deg: float
    calibration_output_directory: str
    calibration_file_prefix: str
    moveit_ready_timeout: float
    moveit_ready_poll_interval: float
    joint_waypoints_deg: Tuple[str, ...] = ()
    waypoint_specs: Tuple[JointWaypointSpec, ...] = ()
    joint_limits_deg: Tuple[Tuple[float, float], ...] = ()


def _load_yaml_defaults(filename: str = "auto_calibration_collector_params.yaml", node_name: str = "auto_calibration_collector") -> dict:
    paths = [os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "config", filename,
    ))]
    try:
        paths.append(os.path.join(
            get_package_share_directory("hand_eye_calibration"),
            "config", filename,
        ))
    except Exception:
        pass
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid calibration YAML configuration {path}: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Cannot read calibration YAML configuration {path}: {exc}") from exc
        parameters = data.get(node_name, {}).get("ros__parameters", {})
        if isinstance(parameters, dict):
            return flatten_ros_parameters(parameters)
    raise FileNotFoundError(
        f"Cannot locate {filename}; searched " + ", ".join(paths)
    )


def yaml_use_sim_time() -> bool:
    """Return the direct-collector time source selected in the shared YAML."""
    value = _load_yaml_defaults().get("use_sim_time", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError("use_sim_time in auto_calibration_collector_params.yaml must be true or false")


def _param(node, name, default, cast):
    node.declare_parameter(name, default)
    return cast(node.get_parameter(name).value)


def _bool(value) -> bool:
    return value.lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)


def _list(node, name, default) -> list:
    node.declare_parameter(name, default)
    value = node.get_parameter(name).value
    return list(default if value is None else value)


def _parse_joint_waypoints(raw) -> Tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) != JOINT_WAYPOINT_SLOTS:
        got = len(raw) if isinstance(raw, list) else "not-a-list"
        raise ValueError(f"joint_waypoints_deg must have exactly {JOINT_WAYPOINT_SLOTS} slots, got {got}")
    slots, seen = [], set()
    for index, value in enumerate(raw, start=1):
        slot = str(value).strip()
        if slot.upper() == "TODO":
            if index == 1:
                raise ValueError("joint_waypoints_deg slot 1 must be the real initial pose, not TODO")
            slots.append(slot)
            continue
        try:
            joints = tuple(float(part.strip()) for part in slot.split(","))
        except ValueError as exc:
            raise ValueError(f"joint_waypoints_deg slot {index} contains non-numeric value: {slot!r}") from exc
        if len(joints) != 6 or not all(math.isfinite(item) for item in joints):
            raise ValueError(f"joint_waypoints_deg slot {index} must contain six finite values")
        if index == 1 and joints != INITIAL_JOINT_DEG:
            raise ValueError(f"joint_waypoints_deg slot 1 must equal {INITIAL_JOINT_DEG}, got {joints}")
        if joints in seen:
            raise ValueError(f"joint_waypoints_deg slot {index} duplicates an earlier waypoint: {joints}")
        seen.add(joints)
        slots.append(slot)
    return tuple(slots)


def _waypoint_specs(slots: Tuple[str, ...]) -> Tuple[JointWaypointSpec, ...]:
    return tuple(
        JointWaypointSpec(index, f"waypoint{index:02d}", tuple(float(part) for part in slot.split(",")))
        for index, slot in enumerate(slots, start=1) if slot.upper() != "TODO"
    )


def _parse_joint_limits(raw, count: int) -> Tuple[Tuple[float, float], ...]:
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError(f"joint_limits_deg must contain {count} 'lower,upper' strings")
    limits = []
    for index, raw_limit in enumerate(raw, start=1):
        try:
            lower, upper = (float(part.strip()) for part in str(raw_limit).split(","))
        except ValueError as exc:
            raise ValueError(f"joint_limits_deg slot {index} must be 'lower,upper'") from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"joint_limits_deg slot {index} must have finite lower < upper")
        limits.append((lower, upper))
    return tuple(limits)


def _validate_positive(**values) -> None:
    invalid = [name for name, value in values.items() if not math.isfinite(float(value)) or float(value) <= 0.0]
    if invalid:
        raise ValueError(f"collector parameters must be finite and positive: {', '.join(invalid)}")


def load_collector_config(node):
    return _load_config(node, "auto_calibration_collector_params.yaml", "auto_calibration_collector", with_waypoints=True)


def load_manual_config(node):
    return _load_config(node, "manual_calibration_assistant_params.yaml", "manual_calibration_assistant", with_waypoints=False)


def _load_config(node, filename: str, node_name: str, *, with_waypoints: bool):
    defaults = _load_yaml_defaults(filename, node_name)
    d = defaults.get
    slots = (
        _parse_joint_waypoints(_list(node, "joint_waypoints_deg", d("joint_waypoints_deg", _default_waypoint_template())))
        if with_waypoints else ()
    )
    frames = CollectorFramesConfig(
        calibration_type=normalize_calibration_type(_param(
            node, "calibration_type", d("calibration_type", CalibrationType.EYE_IN_HAND.value), str,
        )),
        base_frame=_param(node, "base_frame", d("base_frame", "base_link"), str),
        ee_frame=_param(node, "ee_frame", d("ee_frame", "tool0"), str),
        tracking_base_frame=_param(node, "tracking_base_frame", d("tracking_base_frame", "camera_color_optical_frame"), str),
        tracking_marker_frame=_param(node, "tracking_marker_frame", d("tracking_marker_frame", "calibration_aruco"), str),
        marker_id=_param(node, "marker_id", d("marker_id", 1), int),
        image_topic=_param(node, "image_topic", d("image_topic", "/camera/camera/color/image_raw"), str),
        aruco_dictionary_id=_param(node, "aruco_dictionary_id", d("aruco_dictionary_id", "DICT_5X5_250"), str),
        camera_info_topic=_param(node, "camera_info_topic", d("camera_info_topic", "/camera/camera/color/camera_info"), str),
        camera_intrinsics_source=_param(node, "camera_intrinsics_source", d("camera_intrinsics_source", "p"), str).lower(),
    )
    motion = CollectorMotionConfig(
        move_group_name=_param(node, "move_group_name", d("move_group_name", "robot_arm"), str),
        move_group_ns_fairino=_param(node, "move_group_ns_fairino", d("move_group_ns_fairino", "/move_group_fairino"), str),
        planning_pipeline_id=_param(node, "planning_pipeline_id", d("planning_pipeline_id", "fairino"), str),
        planner_id=_param(node, "planner_id", d("planner_id", "birrt*"), str),
        joint_names=tuple(_list(node, "joint_names", d("joint_names", list(_DEFAULT_JOINT_NAMES)))),
        max_velocity=_param(node, "max_velocity", d("max_velocity", 0.3), float),
        max_acceleration=_param(node, "max_acceleration", d("max_acceleration", 0.3), float),
        allowed_planning_time=_param(node, "allowed_planning_time", d("allowed_planning_time", 30.0), float),
        max_step_size=_param(node, "max_step_size", d("max_step_size", 0.05), float),
        position_tolerance=_param(node, "position_tolerance", d("position_tolerance", 0.005), float),
        orientation_tolerance=_param(node, "orientation_tolerance", d("orientation_tolerance", 0.005), float),
        allowed_start_tolerance=_param(node, "allowed_start_tolerance", d("allowed_start_tolerance", 0.1), float),
        action_delay=_param(node, "action_delay", d("action_delay", 0.2), float),
        settle_time_sec=_param(node, "settle_time_sec", d("settle_time_sec", 1.0), float),
        keyboard_poll_period=_param(node, "keyboard_poll_period", d("keyboard_poll_period", 0.1), float),
        start_wait_poll_period=_param(node, "start_wait_poll_period", d("start_wait_poll_period", 0.1), float),
        step_between_actions=_param(node, "step_between_actions", d("step_between_actions", True), _bool),
    )
    sampling = CollectorSamplingConfig(
        marker_size_m=_param(node, "marker_size_m", d("marker_size_m", 0.07), float),
        marker_distance_min_m=_param(node, "marker_distance_min_m", d("marker_distance_min_m", 0.20), float),
        marker_distance_max_m=_param(node, "marker_distance_max_m", d("marker_distance_max_m", 0.80), float),
        stable_frames=_param(node, "stable_frames", d("stable_frames", 10), int),
        stable_marker_timeout_sec=_param(node, "stable_marker_timeout_sec", d("stable_marker_timeout_sec", 5.0), float),
        minimum_corner_margin_px=_param(node, "minimum_corner_margin_px", d("minimum_corner_margin_px", 60.0), float),
        minimum_marker_side_px=_param(node, "minimum_marker_side_px", d("minimum_marker_side_px", 90.0), float),
        maximum_center_std_px=_param(node, "maximum_center_std_px", d("maximum_center_std_px", 4.0), float),
        maximum_marker_depth_std_m=_param(node, "maximum_marker_depth_std_m", d("maximum_marker_depth_std_m", 0.003), float),
        maximum_marker_angle_std_deg=_param(node, "maximum_marker_angle_std_deg", d("maximum_marker_angle_std_deg", 0.8), float),
        joint_stationary_max_position_delta_rad=_param(node, "joint_stationary_max_position_delta_rad", d("joint_stationary_max_position_delta_rad", 0.0001), float),
        joint_stationary_window_sec=_param(node, "joint_stationary_window_sec", d("joint_stationary_window_sec", 0.30), float),
        joint_stationary_timeout_sec=_param(node, "joint_stationary_timeout_sec", d("joint_stationary_timeout_sec", 5.0), float),
        stable_tf_settle_sec=_param(node, "stable_tf_settle_sec", d("stable_tf_settle_sec", 0.15), float),
        minimum_samples=_param(node, "minimum_samples", d("minimum_samples", 15), int),
        minimum_solution_samples=_param(node, "minimum_solution_samples", d("minimum_solution_samples", 14), int),
        minimum_translation_delta_m=_param(node, "minimum_translation_delta_m", d("minimum_translation_delta_m", 0.006), float),
        minimum_rotation_delta_deg=_param(node, "minimum_rotation_delta_deg", d("minimum_rotation_delta_deg", 3.0), float),
        minimum_translation_span_m=_param(node, "minimum_translation_span_m", d("minimum_translation_span_m", 0.04), float),
        minimum_rotation_span_deg=_param(node, "minimum_rotation_span_deg", d("minimum_rotation_span_deg", 20.0), float),
        algorithm_names=tuple(_list(node, "algorithm_names", d("algorithm_names", ["OpenCV/Park", "OpenCV/Horaud"]))),
        maximum_algorithm_translation_delta_m=_param(node, "maximum_algorithm_translation_delta_m", d("maximum_algorithm_translation_delta_m", 0.003), float),
        maximum_algorithm_rotation_delta_deg=_param(node, "maximum_algorithm_rotation_delta_deg", d("maximum_algorithm_rotation_delta_deg", 1.0), float),
        maximum_camera_translation_norm_m=_param(node, "maximum_camera_translation_norm_m", d("maximum_camera_translation_norm_m", 0.30), float),
        maximum_eye_on_base_camera_translation_norm_m=_param(
            node,
            "maximum_eye_on_base_camera_translation_norm_m",
            d("maximum_eye_on_base_camera_translation_norm_m", 2.0),
            float,
        ),
        maximum_marker_position_rms_m=_param(node, "maximum_marker_position_rms_m", d("maximum_marker_position_rms_m", 0.002), float),
        maximum_marker_rotation_rms_deg=_param(node, "maximum_marker_rotation_rms_deg", d("maximum_marker_rotation_rms_deg", 0.70), float),
        fixed_marker_refinement_translation_sigma_m=_param(node, "fixed_marker_refinement_translation_sigma_m", d("fixed_marker_refinement_translation_sigma_m", 0.0005), float),
        fixed_marker_refinement_rotation_sigma_deg=_param(node, "fixed_marker_refinement_rotation_sigma_deg", d("fixed_marker_refinement_rotation_sigma_deg", 0.30), float),
        fixed_marker_refinement_max_iterations=max(1, _param(node, "fixed_marker_refinement_max_iterations", d("fixed_marker_refinement_max_iterations", 25), int)),
        ground_truth_check_enabled=_param(node, "ground_truth_check_enabled", d("ground_truth_check_enabled", True), _bool),
        ground_truth_max_translation_error_m=_param(node, "ground_truth_max_translation_error_m", d("ground_truth_max_translation_error_m", 0.003), float),
        ground_truth_max_axis_error_m=_param(node, "ground_truth_max_axis_error_m", d("ground_truth_max_axis_error_m", 0.002), float),
        ground_truth_max_rotation_error_deg=_param(node, "ground_truth_max_rotation_error_deg", d("ground_truth_max_rotation_error_deg", 1.0), float),
        calibration_output_directory=os.path.expanduser(os.path.expandvars(_param(
            node, "calibration_output_directory",
            d("calibration_output_directory", "$HOME/fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim"), str,
        ))),
        calibration_file_prefix=_param(node, "calibration_file_prefix", d("calibration_file_prefix", "robot_calibration"), str),
        moveit_ready_timeout=_param(node, "moveit_ready_timeout", d("moveit_ready_timeout", 30.0), float),
        moveit_ready_poll_interval=_param(node, "moveit_ready_poll_interval", d("moveit_ready_poll_interval", 0.2), float),
        joint_waypoints_deg=slots,
        waypoint_specs=_waypoint_specs(slots) if with_waypoints else (),
        joint_limits_deg=(
            _parse_joint_limits(_list(node, "joint_limits_deg", d("joint_limits_deg", _default_joint_limits_deg())), len(_DEFAULT_JOINT_NAMES))
            if with_waypoints else ()
        ),
    )
    if frames.camera_intrinsics_source != "p":
        raise ValueError("camera_intrinsics_source must be p for the WVCSC-aligned collector")
    if len(motion.joint_names) != 6:
        raise ValueError("joint_names must contain six joints")
    if with_waypoints and not sampling.waypoint_specs:
        raise ValueError("joint_waypoints_deg has no real waypoint")
    if sampling.stable_frames != 10:
        raise ValueError("stable_frames must be 10 for the WVCSC-aligned collector")
    if (sampling.minimum_samples, sampling.minimum_solution_samples) != (15, 14):
        raise ValueError("minimum_samples and minimum_solution_samples must be 15 and 14")
    if tuple(sampling.algorithm_names) != ("OpenCV/Park", "OpenCV/Horaud"):
        raise ValueError("algorithm_names must be OpenCV/Park and OpenCV/Horaud")
    _validate_positive(
        max_velocity=motion.max_velocity,
        max_acceleration=motion.max_acceleration,
        allowed_planning_time=motion.allowed_planning_time,
        max_step_size=motion.max_step_size,
        position_tolerance=motion.position_tolerance,
        orientation_tolerance=motion.orientation_tolerance,
        allowed_start_tolerance=motion.allowed_start_tolerance,
        action_delay=motion.action_delay,
        settle_time_sec=motion.settle_time_sec,
        keyboard_poll_period=motion.keyboard_poll_period,
        start_wait_poll_period=motion.start_wait_poll_period,
        marker_size_m=sampling.marker_size_m,
        marker_distance_min_m=sampling.marker_distance_min_m,
        marker_distance_max_m=sampling.marker_distance_max_m,
        stable_marker_timeout_sec=sampling.stable_marker_timeout_sec,
        minimum_corner_margin_px=sampling.minimum_corner_margin_px,
        minimum_marker_side_px=sampling.minimum_marker_side_px,
        maximum_center_std_px=sampling.maximum_center_std_px,
        maximum_marker_depth_std_m=sampling.maximum_marker_depth_std_m,
        maximum_marker_angle_std_deg=sampling.maximum_marker_angle_std_deg,
        joint_stationary_max_position_delta_rad=sampling.joint_stationary_max_position_delta_rad,
        joint_stationary_window_sec=sampling.joint_stationary_window_sec,
        joint_stationary_timeout_sec=sampling.joint_stationary_timeout_sec,
        stable_tf_settle_sec=sampling.stable_tf_settle_sec,
        minimum_translation_delta_m=sampling.minimum_translation_delta_m,
        minimum_rotation_delta_deg=sampling.minimum_rotation_delta_deg,
        minimum_translation_span_m=sampling.minimum_translation_span_m,
        minimum_rotation_span_deg=sampling.minimum_rotation_span_deg,
        maximum_algorithm_translation_delta_m=sampling.maximum_algorithm_translation_delta_m,
        maximum_algorithm_rotation_delta_deg=sampling.maximum_algorithm_rotation_delta_deg,
        maximum_camera_translation_norm_m=sampling.maximum_camera_translation_norm_m,
        maximum_eye_on_base_camera_translation_norm_m=sampling.maximum_eye_on_base_camera_translation_norm_m,
        maximum_marker_position_rms_m=sampling.maximum_marker_position_rms_m,
        maximum_marker_rotation_rms_deg=sampling.maximum_marker_rotation_rms_deg,
        fixed_marker_refinement_translation_sigma_m=sampling.fixed_marker_refinement_translation_sigma_m,
        fixed_marker_refinement_rotation_sigma_deg=sampling.fixed_marker_refinement_rotation_sigma_deg,
        ground_truth_max_translation_error_m=sampling.ground_truth_max_translation_error_m,
        ground_truth_max_axis_error_m=sampling.ground_truth_max_axis_error_m,
        ground_truth_max_rotation_error_deg=sampling.ground_truth_max_rotation_error_deg,
        moveit_ready_timeout=sampling.moveit_ready_timeout,
        moveit_ready_poll_interval=sampling.moveit_ready_poll_interval,
    )
    if sampling.marker_distance_min_m >= sampling.marker_distance_max_m:
        raise ValueError("require marker_distance_min_m < marker_distance_max_m")
    return frames, motion, sampling


def _default_waypoint_template() -> List[str]:
    return [",".join(f"{value:g}" for value in INITIAL_JOINT_DEG)] + ["TODO"] * (JOINT_WAYPOINT_SLOTS - 1)


def _default_joint_limits_deg() -> List[str]:
    return ["-175,175", "-265,85", "-162,162", "-265,85", "-175,175", "-175,175"]
