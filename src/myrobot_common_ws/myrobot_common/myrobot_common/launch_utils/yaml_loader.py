"""YAML loading helpers used by myrobot_simulation launch files."""

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Dict

from ament_index_python.packages import get_package_share_directory
import yaml


def load_yaml(package_name: str, relative_path: str) -> Dict[str, Any]:
    """Load a YAML file from a package share directory as a Python dict."""
    pkg_path = get_package_share_directory(package_name)
    abs_path = os.path.join(pkg_path, relative_path)
    with open(abs_path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_environment_yaml(
    package_name: str,
    relative_path: str,
    environment: str | None = None,
) -> Dict[str, Any]:
    """Load one business YAML, merging ``common`` with its selected environment."""
    data = load_yaml(package_name, relative_path)
    if "common" not in data and "environments" not in data:
        return data
    if environment not in {"real", "sim"}:
        raise ValueError(f"environment must be 'real' or 'sim', got {environment!r}")
    common = data.get("common", {})
    environments = data.get("environments", {})
    selected = environments.get(environment, {}) if isinstance(environments, dict) else {}
    if not isinstance(common, dict) or not isinstance(selected, dict):
        raise ValueError(f"{relative_path} must contain mapping common/environments blocks")
    return _deep_merge(common, selected)


def load_launch_parameters_yaml(
    package_name: str,
    relative_path: str,
    environment: str,
) -> Dict[str, Any]:
    """Return the selected environment's launch defaults from a business YAML."""
    launch = load_environment_yaml(package_name, relative_path, environment).get("launch", {})
    return launch if isinstance(launch, dict) else {}


def launch_defaults_as_strings(values: Dict[str, Any]) -> Dict[str, str]:
    """Convert YAML launch defaults to values accepted by DeclareLaunchArgument."""
    return {
        name: str(value).lower() if isinstance(value, bool) else str(value)
        for name, value in values.items()
    }


def launch_parameter_value(raw: str, fallback: Any) -> Any:
    """Convert one launch substitution using the selected YAML value's type."""
    if isinstance(fallback, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(fallback, int):
        return int(raw)
    if isinstance(fallback, float):
        return float(raw)
    return raw


def load_ros_parameters_yaml(
    package_name: str,
    relative_path: str,
    node_scope: str,
    environment: str | None = None,
) -> Dict[str, Any]:
    """Load `ros__parameters` for a node scope from a ROS2 params YAML file."""
    try:
        data = load_environment_yaml(package_name, relative_path, environment)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    nodes = data.get("nodes", data)
    scoped = nodes.get(node_scope, {}) if isinstance(nodes, dict) else {}
    if not isinstance(scoped, dict):
        return {}

    params = scoped.get("ros__parameters", {})
    return params if isinstance(params, dict) else {}


def flatten_moveit_parameters(moveit: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the shared hierarchical MoveIt block to existing ROS names."""
    frames = moveit.get("frames", {})
    groups = moveit.get("groups", {})
    namespaces = moveit.get("move_groups", {})
    ik = moveit.get("ik", {})
    pipeline = moveit.get("pipeline", {})
    planner = moveit.get("planner", {})
    readiness = moveit.get("readiness", {})
    return {
        "base_frame": frames.get("base_frame", "base_link"),
        "ee_frame": frames.get("ee_frame", "tool0"),
        "arm_group_name": groups.get("arm", "robot_arm"),
        "hand_group_name": groups.get("gripper", "hand"),
        "move_group_ns_fairino": namespaces.get("fairino", "/move_group_fairino"),
        "move_group_ns_kdl": namespaces.get("kdl", "/move_group_kdl"),
        "ik_plugin": ik.get("default", "fairino"),
        "planning_pipeline_id": pipeline.get("default", "fairino"),
        "planner_id": planner.get("default", "tube_birrt*"),
        "move_group_ready_timeout_sec": readiness.get("timeout_sec", 10.0),
        "allow_cross_client_fallback": readiness.get("allow_cross_client_fallback", True),
        "ik_moveit_servo": moveit.get("ik_moveit_servo", "kdl"),
    }


def load_moveit_parameters_yaml(
    package_name: str,
    relative_path: str,
    node_scope: str,
    environment: str | None = None,
) -> Dict[str, Any]:
    """Load and flatten one node's ``moveit`` block from a ROS params file."""
    params = load_ros_parameters_yaml(package_name, relative_path, node_scope, environment)
    moveit = params.get("moveit", {})
    return flatten_moveit_parameters(moveit if isinstance(moveit, dict) else {})


def load_node_parameters_yaml(
    package_name: str,
    relative_path: str,
    node_scope: str,
    environment: str | None = None,
) -> Dict[str, Any]:
    """Load one node's parameters, flattening its hierarchical MoveIt block."""
    data = load_environment_yaml(package_name, relative_path, environment)
    nodes = data.get("nodes", data) if isinstance(data, dict) else {}
    shared = nodes.get("/**", {}) if isinstance(nodes, dict) else {}
    node = nodes.get(node_scope, {}) if isinstance(nodes, dict) else {}
    parameters: Dict[str, Any] = {}
    for scope in (shared, node):
        scoped = scope.get("ros__parameters", {}) if isinstance(scope, dict) else {}
        if isinstance(scoped, dict):
            parameters.update(scoped)
    moveit = parameters.pop("moveit", {})
    if isinstance(moveit, dict):
        parameters.update(flatten_moveit_parameters(moveit))
    return parameters


def package_file(package_name: str, relative_path: str) -> str:
    """Return an absolute path inside a package share directory."""
    return os.path.join(get_package_share_directory(package_name), relative_path)


def wrap_yaml_as_ros_params_file(
    package_name: str,
    relative_path: str,
    node_scope: str = "/**",
) -> str:
    """Wrap a plain YAML mapping into a ROS2 params file and return its temp path."""
    payload = load_yaml(package_name, relative_path)
    wrapped = {
        node_scope: {
            "ros__parameters": payload,
        }
    }

    content_hash = hashlib.sha1(
        yaml.safe_dump(wrapped, sort_keys=False).encode("utf-8")
    ).hexdigest()[:10]
    file_stem = Path(relative_path).stem
    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"{package_name}_{file_stem}_{content_hash}_ros_params.yaml",
    )
    with open(temp_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(wrapped, stream, sort_keys=False)
    return temp_path


def write_node_parameters_ros_file(
    package_name: str,
    relative_path: str,
    node_scope: str,
    environment: str,
) -> str:
    """Materialize selected business-node parameters for external ROS processes."""
    parameters = load_node_parameters_yaml(
        package_name, relative_path, node_scope, environment
    )
    wrapped = {node_scope: {"ros__parameters": parameters}}
    content_hash = hashlib.sha1(
        yaml.safe_dump(wrapped, sort_keys=False).encode("utf-8")
    ).hexdigest()[:10]
    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"{package_name}_{Path(relative_path).stem}_{environment}_{node_scope}_{content_hash}.yaml".replace("/", "_"),
    )
    with open(temp_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(wrapped, stream, sort_keys=False)
    return temp_path
