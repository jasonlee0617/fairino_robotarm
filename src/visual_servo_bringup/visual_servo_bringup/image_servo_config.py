"""Load the ArUco IBVS configuration shared by Gazebo and hardware launches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from myrobot_common.launch_utils.yaml_loader import flatten_moveit_parameters


def config_path() -> Path:
    from ament_index_python.packages import get_package_share_directory

    share = Path(get_package_share_directory("visual_servo_bringup"))
    return share / "config" / "visual_image_servo_params.yaml"


NODE_PARAMETER_DESCRIPTIONS = {
    "image_topic": "ArUco 检测图像话题。",
    "camera_info_topic": "图像内参话题。",
    "debug_image_topic": "带 ArUco 边框的调试图像话题。",
    "error_topic": "八维归一化图像误差话题。",
    "marker_dictionary": "OpenCV ArUco 字典名称。",
    "marker_id": "要跟踪的 ArUco 标定码 ID。",
    "marker_size_m": "标定码物理边长，单位米。",
    "base_frame": "机械臂基座坐标系。",
    "camera_frame": "相机光学坐标系。",
    "ee_frame": "机械臂末端坐标系。",
    "servo_ns": "MoveIt Servo 命名空间。",
    "control_rate_hz": "IBVS 控制周期频率。",
    "detector_rate_hz": "ArUco 完整检测频率；中间帧使用四角点跟踪。",
    "tracker_max_error_px": "光流角点最大允许误差，单位像素。",
    "debug_image_rate_hz": "调试图像最大发布频率；0 表示关闭。",
    "enable_subpixel_refinement": "完整检测时是否启用 ArUco 亚像素角点优化。",
    "lambda_gain": "IBVS 比例增益。",
    "damping": "交互矩阵阻尼系数。",
    "max_linear_speed": "末端线速度范数上限，单位 m/s。",
    "max_angular_speed": "末端角速度范数上限，单位 rad/s。",
    "feature_timeout_sec": "ArUco 特征最大允许时效，单位秒。",
    "servo_stop_timeout_sec": "特征过期后保持零速度多久再停用 MoveIt Servo，单位秒。",
    "image_error_tolerance": "图像误差死区阈值。",
    "servo_status_halt_codes": "触发安全停止的 MoveIt Servo 状态码列表。",
    "reference_path": "参考图像角点 YAML；留空时使用环境入口默认路径。",
    "auto_start": "检测到有效参考后是否自动开始 IBVS。",
}


def _node_config(config: dict[str, Any], environment: str) -> dict[str, Any]:
    if environment not in {"real", "sim"}:
        raise ValueError("environment must be 'real' or 'sim'")
    node = dict(config["common"]["nodes"]["visual_image_servo"])
    node.update(config["environments"].get(environment, {}).get("nodes", {}).get("visual_image_servo", {}))
    return node


def image_servo_parameters(path: Path | None = None, environment: str = "real") -> dict[str, Any]:
    with (path or config_path()).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    node = _node_config(config, environment)
    parameters: dict[str, Any] = {}
    for name, section in node.items():
        if name == "moveit":
            continue
        parameters.update(section)
    return parameters


def image_servo_moveit_parameters(path: Path | None = None, environment: str = "real") -> dict[str, Any]:
    with (path or config_path()).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return flatten_moveit_parameters(_node_config(config, environment).get("moveit", {}))
