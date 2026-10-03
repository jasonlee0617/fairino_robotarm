import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
)
from llm_arm_control_nodes.voice_launch import build_voice_launch_actions


_TASK_PARAMETERS = load_node_parameters_yaml(
    "llm_arm_control", "config/llm_robot_control_params.yaml", "llm_control_task_server", "sim"
)
_PERCEPTION_PARAMETERS = load_node_parameters_yaml(
    "llm_arm_control", "config/llm_robot_control_params.yaml", "llm_visual_perception", "sim"
)
_PUBLIC_TASK_PARAMETER_NAMES = (
    "ik_plugin", "planning_pipeline_id", "planner_id", "move_group_ready_timeout_sec",
    "allow_cross_client_fallback", "arm_max_velocity", "arm_max_acceleration",
)
_PUBLIC_TASK_FALLBACKS = {name: _TASK_PARAMETERS[name] for name in _PUBLIC_TASK_PARAMETER_NAMES}


# 场景与节点标量默认值集中维护；YAML 仅覆盖这里的 launch fallback。
_LAUNCH_ARGUMENT_SPECS = (
    ("robot_profile", "fairino_arm_gripper_inhand", "Gazebo 机器人配置。", None),
    ("world", "visual_world", "Gazebo 世界资源。", None),
    ("use_sim_time", "true", "是否使用 Gazebo /clock。", None),
    ("camera_profile", "d435_color_1280x720x30_depth_848x480x30", "D435 命名相机配置。", None),
    ("camera_profile_file", "", "外部 D435 配置 YAML。", None),
    ("camera_noise_mode", "off", "相机噪声模型。", ("off", "d435_empirical")),
    ("camera_depth_far_m", "3.0", "D435 深度远裁剪距离，单位米。", None),
    ("native_depth_enabled", "false", "LLM 任务是否额外启用未使用的原生深度流。", None),
    ("publish_frequency", "100.0", "机器人状态发布频率（Hz）。", None),
    ("enable_camera_model", "true", "是否生成仿真相机模型。", None),
    ("enable_camera_bridge", "true", "是否桥接相机话题到 ROS 2。", None),
    ("camera_fps", "30", "相机帧率。", None),
    ("camera_image_width", "1280", "彩色图像宽度。", None),
    ("camera_image_height", "720", "彩色图像高度。", None),
    ("spawn_z", "1.02", "机器人初始高度。", None),
    ("controller_spawn_delay", "5.0", "控制器启动等待时间。", None),
    ("enable_voice", "true", "是否启动语音控制链。", None),
    (
        "audio_input_device", "auto",
        "麦克风设备：auto、PulseAudio source 名称或 hw/plughw:CARD,DEVICE。",
        None,
    ),
    ("audio_input_volume_percent", "100", "麦克风输入音量百分比（1-100）。", None),
    *(
        (
            name,
            str(default).lower() if isinstance(default, bool) else str(default),
            "LLM 规划运行参数；默认值来自 llm_robot_control_params.yaml。",
            None,
        )
        for name, default in _PUBLIC_TASK_FALLBACKS.items()
    ),
)
_YAML_LAUNCH_DEFAULTS = launch_defaults_as_strings(
    load_launch_parameters_yaml("llm_arm_control", "config/llm_robot_control_params.yaml", "sim")
)
_LAUNCH_ARGUMENT_SPECS = tuple(
    (name, _YAML_LAUNCH_DEFAULTS.get(name, default), description, choices)
    for name, default, description, choices in _LAUNCH_ARGUMENT_SPECS
)
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name, *_ in _LAUNCH_ARGUMENT_SPECS
}


def _declare_launch_arguments():
    """声明可由 CLI 覆盖的场景和节点运行参数."""
    declarations = []
    for name, default, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": default, "description": description}
        if choices:
            kwargs["choices"] = list(choices)
        declarations.append(DeclareLaunchArgument(name, **kwargs))
    return declarations


def _public_task_parameters(context):
    return {
        name: launch_parameter_value(_LAUNCH_CONFIGURATIONS[name].perform(context), fallback)
        for name, fallback in _PUBLIC_TASK_FALLBACKS.items()
    }


def _launch_setup(context):
    gz_share = get_package_share_directory("myrobot_simulation")
    llm_arm_share = get_package_share_directory("llm_arm_control")

    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            # 眼在手外：fairino_arm_gripper_calibration_onbase；眼在手上：fairino_arm_gripper_inhand。
            "robot_profile": _LAUNCH_CONFIGURATIONS["robot_profile"],
            "world": _LAUNCH_CONFIGURATIONS["world"],
            "rviz_config": os.path.join(llm_arm_share, "rviz", "llm_robot_control.rviz"),
            "publish_frequency": _LAUNCH_CONFIGURATIONS["publish_frequency"],
            "enable_camera_model": _LAUNCH_CONFIGURATIONS["enable_camera_model"],
            "enable_camera_bridge": _LAUNCH_CONFIGURATIONS["enable_camera_bridge"],
            "camera_fps": _LAUNCH_CONFIGURATIONS["camera_fps"],
            "camera_image_width": _LAUNCH_CONFIGURATIONS["camera_image_width"],
            "camera_image_height": _LAUNCH_CONFIGURATIONS["camera_image_height"],
            "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
            "camera_profile": _LAUNCH_CONFIGURATIONS["camera_profile"],
            "camera_profile_file": _LAUNCH_CONFIGURATIONS["camera_profile_file"],
            "camera_noise_mode": _LAUNCH_CONFIGURATIONS["camera_noise_mode"],
            "camera_depth_far_m": _LAUNCH_CONFIGURATIONS["camera_depth_far_m"],
            "native_depth_enabled": _LAUNCH_CONFIGURATIONS["native_depth_enabled"],
            "spawn_z": _LAUNCH_CONFIGURATIONS["spawn_z"],
            "controller_spawn_delay": _LAUNCH_CONFIGURATIONS["controller_spawn_delay"],
        }.items(),
    )

    retime_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )

    yolo_obb = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("visual_perception"),
                "launch",
                "llm_visual_perception.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
            "require_cuda_for_visual_perception": launch_defaults_as_strings(
                _PERCEPTION_PARAMETERS
            )["require_cuda_for_visual_perception"],
            "sync_slop": launch_defaults_as_strings(
                _PERCEPTION_PARAMETERS
            )["sync_slop"],
            "sync_watchdog_sec": launch_defaults_as_strings(
                _PERCEPTION_PARAMETERS
            )["sync_watchdog_sec"],
            "expected_camera_rate_hz": launch_defaults_as_strings(
                _PERCEPTION_PARAMETERS
            )["expected_camera_rate_hz"],
        }.items(),
    )
    task_server_node = Node(
        package="llm_arm_control",
        executable="llm_control_task_server",
        name="llm_control_task_server",
        output="screen",
        parameters=[
            _TASK_PARAMETERS,
            {
                "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
                **_public_task_parameters(context),
            },
        ],
    )
    voice_actions = build_voice_launch_actions(
        context,
        environment="sim",
        enable_voice=_LAUNCH_CONFIGURATIONS["enable_voice"],
        audio_input_device=_LAUNCH_CONFIGURATIONS["audio_input_device"],
        audio_input_volume_percent=_LAUNCH_CONFIGURATIONS[
            "audio_input_volume_percent"
        ],
        use_sim_time=_LAUNCH_CONFIGURATIONS["use_sim_time"],
    )
    motion_control = Node(
        package="myrobot_common",
        executable="motion_control",
        name="motion_control",
        output="screen",
        parameters=[{
            "command_burst_count": int(_YAML_LAUNCH_DEFAULTS["command_burst_count"]),
            "enable_grasp_trigger_key": False,
            "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
        }],
    )

    return [
        myrobot_simulation,
        retime_server_launch,
        yolo_obb,
        task_server_node,
        motion_control,
        *voice_actions,
    ]


def generate_launch_description():
    return LaunchDescription([
        *_declare_launch_arguments(),
        OpaqueFunction(function=_launch_setup),
    ])
