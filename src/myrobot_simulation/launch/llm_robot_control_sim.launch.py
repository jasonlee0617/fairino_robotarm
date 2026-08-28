import os
import shlex
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
    write_node_parameters_ros_file,
)


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
    ("publish_frequency", "100.0", "机器人状态发布频率（Hz）。", None),
    ("enable_camera_model", "true", "是否生成仿真相机模型。", None),
    ("enable_camera_bridge", "true", "是否桥接相机话题到 ROS 2。", None),
    ("enable_servo", "true", "是否启动视觉伺服节点。", None),
    ("camera_fps", "30", "相机帧率。", None),
    ("camera_image_width", "1280", "彩色图像宽度。", None),
    ("camera_image_height", "720", "彩色图像高度。", None),
    ("spawn_z", "1.02", "机器人初始高度。", None),
    ("controller_spawn_delay", "5.0", "控制器启动等待时间。", None),
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


def _graspnet_inference_process(context):
    model_profile = _YAML_LAUNCH_DEFAULTS["graspnet_model_profile"]
    use_sim_time = _LAUNCH_CONFIGURATIONS["use_sim_time"].perform(context)
    install_setup = str(Path(get_package_prefix("graspnet_bringup")).parent / "setup.bash")
    source_share = get_package_share_directory("graspnet_source")
    config_path = write_node_parameters_ros_file(
        "llm_arm_control", "config/llm_robot_control_params.yaml", "graspnet_inference", "sim"
    )
    conda_setup = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    command = (
        "set -e; "
        f"source {shlex.quote(conda_setup)}; conda activate graspnet; "
        "source /opt/ros/humble/setup.bash; "
        f"source {shlex.quote(install_setup)}; "
        "export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/graspnet_mpl_config XDG_CACHE_HOME=/tmp/graspnet_xdg_cache; "
        "mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME; "
        "exec python -m graspnet_bringup.graspnet_inference_node --ros-args "
        f"--params-file {shlex.quote(config_path)} -r __node:=graspnet_inference "
        f"-p use_sim_time:={use_sim_time} "
        f"-p baseline_dir:={shlex.quote(os.path.join(source_share, 'graspnet_baseline'))} "
        f"-p checkpoint_path:={shlex.quote(os.path.join(source_share, 'models', f'checkpoint-{model_profile}.tar'))}"
    )
    return [ExecuteProcess(cmd=["bash", "-lc", command], output="screen")]


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
            "enable_servo": _LAUNCH_CONFIGURATIONS["enable_servo"],
            "camera_fps": _LAUNCH_CONFIGURATIONS["camera_fps"],
            "camera_image_width": _LAUNCH_CONFIGURATIONS["camera_image_width"],
            "camera_image_height": _LAUNCH_CONFIGURATIONS["camera_image_height"],
            "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
            "camera_profile": _LAUNCH_CONFIGURATIONS["camera_profile"],
            "camera_profile_file": _LAUNCH_CONFIGURATIONS["camera_profile_file"],
            "camera_noise_mode": _LAUNCH_CONFIGURATIONS["camera_noise_mode"],
            "camera_depth_far_m": _LAUNCH_CONFIGURATIONS["camera_depth_far_m"],
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

    yolo_obb = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("visual_perception"),
                        "launch",
                        "llm_visual_perception.launch.py",
                    )
                ),
                launch_arguments={
                    "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
                    "use_continuous_yolo": launch_defaults_as_strings(
                        _PERCEPTION_PARAMETERS
                    )["use_continuous_yolo"],
                }.items(),
            )
        ],
    )
    pose_monitor_node = Node(
        package="llm_arm_control",
        executable="robot_pose_monitor_node",
        name="robot_pose_monitor_node",
        output="screen",
        parameters=[{"use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"]}],
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
    # The CLI owns a real terminal. A launch Node only receives launch-managed
    # stdin, so its input() prompt would be hidden by the shared node output.
    cli_terminal = ExecuteProcess(
        cmd=[
            "gnome-terminal",
            "--title=LLM Robot CLI",
            "--wait",
            "--",
            "ros2",
            "run",
            "llm_arm_control",
            "llm_control_cli",
            "--ros-args",
            "-p",
            ["use_sim_time:=", _LAUNCH_CONFIGURATIONS["use_sim_time"]],
            "-p",
            ["command_burst_count:=", str(_YAML_LAUNCH_DEFAULTS["command_burst_count"])],
        ],
        output="screen",
    )

    return [
        myrobot_simulation,
        retime_server_launch,
        yolo_obb,
        OpaqueFunction(function=_graspnet_inference_process),
        pose_monitor_node,
        task_server_node,
        cli_terminal,
    ]


def generate_launch_description():
    return LaunchDescription([
        *_declare_launch_arguments(),
        OpaqueFunction(function=_launch_setup),
    ])
