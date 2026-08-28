"""启动 Easy Handeye2 GUI 与严格半自动采样助手."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml
from myrobot_common.launch_utils.yaml_loader import launch_parameter_value

_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

from handeye_launch_utils import default_storage_directory, load_handeye_profile
from hand_eye_calibration.config import flatten_ros_parameters


PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}

# 启动拓扑参数：YAML 只为此处提供默认值，命令行 launch 参数可覆盖。
# 这些值会同时传给 Easy Handeye2 和助手，不能让节点参数 YAML 再次覆盖。
_TOPOLOGY_PARAMETER_NAMES = frozenset({
    "calibration_type",
    "use_sim_time",
    "base_frame",
    "ee_frame",
    "tracking_base_frame",
    "tracking_marker_frame",
    "calibration_output_directory",
})
_PUBLIC_NODE_PARAMETER_NAMES = (
    "planning_pipeline_id", "planner_id", "max_velocity", "max_acceleration",
    "allowed_planning_time", "max_step_size", "position_tolerance",
    "orientation_tolerance", "allowed_start_tolerance",
)

_LAUNCH_CHOICES = {
    "calibration_type": ["eye_in_hand", "eye_on_base"],
}

_LAUNCH_DESCRIPTIONS = {
    "calibration_type": "标定类型；命令行覆盖 YAML 默认值，并同步给 Easy 与助手。",
    "calibration_name": "标定名称覆盖；留空时使用内置 profile。",
    "use_sim_time": "是否使用 Gazebo 仿真时钟。",
    "storage_directory": "标定 .calib 与 .samples 的保存目录。",
}


def _manual_yaml_defaults():
    """读取助手自身参数文件中的启动默认值."""
    path = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "manual_calibration_assistant_params.yaml",
    )
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    return flatten_ros_parameters(
        document.get("manual_calibration_assistant", {}).get("ros__parameters", {})
    )


_MANUAL_DEFAULTS = _manual_yaml_defaults()

# 集中管理半自动入口的 launch 默认值；profile 坐标系不在此复制。
_LAUNCH_DEFAULTS = {
    "calibration_type": str(_MANUAL_DEFAULTS.get("calibration_type", "eye_on_base")),
    "calibration_name": "",
    "use_sim_time": str(_MANUAL_DEFAULTS.get("use_sim_time", "false")).lower(),
    "storage_directory": os.path.expandvars(os.path.expanduser(str(
        _MANUAL_DEFAULTS.get(
            "calibration_output_directory", default_storage_directory("sim")
        )
    ))),
    **{
        name: str(_MANUAL_DEFAULTS[name]).lower()
        if isinstance(_MANUAL_DEFAULTS[name], bool) else str(_MANUAL_DEFAULTS[name])
        for name in _PUBLIC_NODE_PARAMETER_NAMES
    },
}

_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _LAUNCH_DEFAULTS
}

# 节点直接默认值 < launch 拓扑覆盖 < YAML 运行参数 < CLI ROS 参数。
_ASSISTANT_NODE_DEFAULTS = {
    "use_sim_time": False,
    "calibration_type": "eye_in_hand",
    "base_frame": "base_link",
    "ee_frame": "tool0",
    "tracking_base_frame": "camera_color_optical_frame",
    "tracking_marker_frame": "calibration_aruco",
    "calibration_output_directory": default_storage_directory("sim"),
}

_ASSISTANT_YAML_PARAMETERS = {
    name: item
    for name, item in _MANUAL_DEFAULTS.items()
    if name not in _TOPOLOGY_PARAMETER_NAMES
}


def _launch_value(context, name: str) -> str:
    """读取本入口已集中声明的 launch 参数。"""
    return str(_LAUNCH_CONFIGURATIONS[name].perform(context)).strip()


def _launch_setup(context, *_args, **_kwargs):
    calibration_type = _launch_value(context, "calibration_type")
    if calibration_type not in ("eye_in_hand", "eye_on_base"):
        raise RuntimeError(
            "calibration_type is required: pass calibration_type:=eye_in_hand or calibration_type:=eye_on_base"
        )
    profile = load_handeye_profile(calibration_type)
    easy_share = get_package_share_directory("easy_handeye2")
    base_frame = profile["robot_base_frame"]
    effector_frame = profile["robot_effector_frame"]
    tracking_base_frame = profile["tracking_base_frame"]
    tracking_marker_frame = profile["tracking_marker_frame"]
    use_sim_time = _launch_value(context, "use_sim_time").lower() == "true"
    storage_directory = _launch_value(context, "storage_directory")
    calibration_name = _launch_value(context, "calibration_name") or profile["calibration_name"]
    public_node_parameters = {
        name: launch_parameter_value(_launch_value(context, name), _MANUAL_DEFAULTS[name])
        for name in _PUBLIC_NODE_PARAMETER_NAMES
    }

    easy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(easy_share, "launch", "calibrate.launch.py")
        ),
        launch_arguments={
            "name": calibration_name,
            "calibration_type": calibration_type,
            "robot_base_frame": base_frame,
            "robot_effector_frame": effector_frame,
            "tracking_base_frame": tracking_base_frame,
            "tracking_marker_frame": tracking_marker_frame,
            "use_sim_time": str(use_sim_time).lower(),
            "storage_directory": storage_directory,
        }.items(),
    )
    assistant = Node(
        package="hand_eye_calibration",
        executable="manual_calibration_assistant.py",
        name="manual_calibration_assistant",
        output="screen",
        additional_env=PYTHON_NO_USER_SITE_ENV,
        parameters=[
            _ASSISTANT_NODE_DEFAULTS,
            {
                "use_sim_time": use_sim_time,
                "calibration_type": calibration_type,
                "base_frame": base_frame,
                "ee_frame": effector_frame,
                "tracking_base_frame": tracking_base_frame,
                "tracking_marker_frame": tracking_marker_frame,
                "calibration_output_directory": storage_directory,
            },
            # YAML 普通运行参数最后加载；拓扑键已过滤，不能破坏 Easy 同步。
            _ASSISTANT_YAML_PARAMETERS,
            public_node_parameters,
        ],
    )
    return [easy, assistant]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "calibration_type",
            default_value=_LAUNCH_DEFAULTS["calibration_type"],
            choices=_LAUNCH_CHOICES["calibration_type"],
            description=_LAUNCH_DESCRIPTIONS["calibration_type"],
        ),
        DeclareLaunchArgument(
            "calibration_name",
            default_value=_LAUNCH_DEFAULTS["calibration_name"],
            description=_LAUNCH_DESCRIPTIONS["calibration_name"],
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value=_LAUNCH_DEFAULTS["use_sim_time"],
            description=_LAUNCH_DESCRIPTIONS["use_sim_time"],
        ),
        DeclareLaunchArgument(
            "storage_directory",
            default_value=_LAUNCH_DEFAULTS["storage_directory"],
            description=_LAUNCH_DESCRIPTIONS["storage_directory"],
        ),
        *[
            DeclareLaunchArgument(
                name,
                default_value=_LAUNCH_DEFAULTS[name],
                description="采样助手的离散规划参数。",
            )
            for name in _PUBLIC_NODE_PARAMETER_NAMES
        ],
        OpaqueFunction(function=_launch_setup),
    ])
