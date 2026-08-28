import os
import shlex
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
    write_node_parameters_ros_file,
)


_TASK_PARAMETERS = load_node_parameters_yaml(
    "graspnet_bringup", "config/graspnet_grasping_params.yaml", "graspnet_visual_grasping", "sim"
)
_PUBLIC_TASK_PARAMETER_NAMES = (
    "ik_plugin", "planning_pipeline_id", "planner_id", "move_group_ready_timeout_sec",
    "allow_cross_client_fallback", "arm_max_velocity", "arm_max_acceleration",
    "allowed_planning_time", "position_tolerance", "orientation_tolerance",
    "allowed_start_tolerance",
)
_PUBLIC_TASK_FALLBACKS = {name: _TASK_PARAMETERS[name] for name in _PUBLIC_TASK_PARAMETER_NAMES}


# 场景与节点标量参数集中在此处。YAML、RViz 和权重等固定资源在使用位置
# 直接通过包共享目录定位，不作为可变的 launch 参数。
_LAUNCH_ARGUMENT_SPECS = (
    ("robot_profile", "fairino_arm_gripper_inhand", "Gazebo 机器人配置。", None),
    ("world", "visual_world", "Gazebo 世界资源。", None),
    ("enable_rviz", "true", "是否启动 RViz。", None),
    ("use_sim_time", "true", "是否使用 Gazebo 的 /clock。", None),
    ("publish_frequency", "30.0", "机器人状态发布频率（Hz）。", None),
    ("enable_camera_model", "true", "是否生成仿真相机模型。", None),
    ("enable_camera_bridge", "true", "是否桥接相机话题到 ROS 2。", None),
    ("enable_servo", "false", "是否启动 Gazebo 内置伺服节点。", None),
    ("camera_fps", "30", "仿真相机帧率。", None),
    ("camera_image_width", "1280", "仿真彩色图像宽度。", None),
    ("camera_image_height", "720", "仿真彩色图像高度。", None),
    ("camera_profile", "d435_color_1280x720x30_depth_848x480x30", "命名 D435 配置。", None),
    ("camera_profile_file", "", "外部 D435 配置文件；使用时清空 camera_profile。", None),
    ("camera_noise_mode", "off", "相机噪声模型。", ("off", "d435_empirical")),
    ("camera_depth_far_m", "3.0", "深度远裁剪距离（米）。", None),
    ("spawn_x", "0.0", "机器人初始 X 坐标（米）。", None),
    ("spawn_y", "0.0", "机器人初始 Y 坐标（米）。", None),
    ("spawn_z", "1.02", "机器人初始 Z 坐标（米）。", None),
    ("controller_spawn_delay", "5.0", "控制器启动前等待时间（秒）。", None),
    ("calibration_name", "robot_calibration", "手眼标定名称。", None),
    *(
        (
            name,
            str(default).lower() if isinstance(default, bool) else str(default),
            "GraspNet 抓取规划运行参数；默认值来自 graspnet_grasping_params.yaml。",
            None,
        )
        for name, default in _PUBLIC_TASK_FALLBACKS.items()
    ),
)
_YAML_LAUNCH_DEFAULTS = launch_defaults_as_strings(
    load_launch_parameters_yaml("graspnet_bringup", "config/graspnet_grasping_params.yaml", "sim")
)
_LAUNCH_ARGUMENT_SPECS = tuple(
    (name, _YAML_LAUNCH_DEFAULTS.get(name, default), description, choices)
    for name, default, description, choices in _LAUNCH_ARGUMENT_SPECS
)
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name, *_ in _LAUNCH_ARGUMENT_SPECS
}
_SCENE_ARGUMENT_NAMES = tuple(
    name
    for name, *_ in _LAUNCH_ARGUMENT_SPECS
    if name not in {"calibration_name", *_PUBLIC_TASK_PARAMETER_NAMES}
)


def _declare_launch_arguments():
    declarations = []
    for name, default_value, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": default_value, "description": description}
        if choices:
            kwargs["choices"] = list(choices)
        declarations.append(DeclareLaunchArgument(name, **kwargs))
    return declarations


def _graspnet_inference_process(context):
    model_profile = _YAML_LAUNCH_DEFAULTS["model_profile"]
    use_sim_time = _LAUNCH_CONFIGURATIONS["use_sim_time"].perform(context)
    install_setup = str(Path(get_package_prefix("graspnet_bringup")).parent / "setup.bash")
    config_path = write_node_parameters_ros_file(
        "graspnet_bringup", "config/graspnet_grasping_params.yaml", "graspnet_inference", "sim"
    )
    source_share = get_package_share_directory("graspnet_source")
    baseline_dir = os.path.join(source_share, "graspnet_baseline")
    checkpoint_path = os.path.join(source_share, "models", f"checkpoint-{model_profile}.tar")
    conda_setup = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    command_prefix = (
        "set -e; "
        f"source {shlex.quote(conda_setup)}; "
        "conda activate graspnet; "
        "source /opt/ros/humble/setup.bash; "
        f"source {shlex.quote(install_setup)}; "
        "export PYTHONUNBUFFERED=1; "
        "export MPLCONFIGDIR=/tmp/graspnet_mpl_config; "
        "export XDG_CACHE_HOME=/tmp/graspnet_xdg_cache; "
        "mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME; "
        "exec python -m graspnet_bringup.graspnet_inference_node "
        "--ros-args "
        f"--params-file {shlex.quote(config_path)} "
        "-r __node:=graspnet_inference "
        "-p use_sim_time:="
    )
    command_suffix = (
        " "
        f"-p baseline_dir:={shlex.quote(baseline_dir)} "
        f"-p checkpoint_path:={shlex.quote(checkpoint_path)}"
    )
    return [
        ExecuteProcess(
            cmd=["bash", "-lc", command_prefix + use_sim_time + command_suffix],
            output="screen",
        )
    ]


def _public_task_parameters(context):
    return {
        name: launch_parameter_value(_LAUNCH_CONFIGURATIONS[name].perform(context), fallback)
        for name, fallback in _PUBLIC_TASK_FALLBACKS.items()
    }


def _launch_setup(context):
    gz_share = get_package_share_directory("myrobot_simulation")
    graspnet_share = get_package_share_directory("graspnet_bringup")
    launch_config = _LAUNCH_CONFIGURATIONS

    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            **{name: launch_config[name] for name in _SCENE_ARGUMENT_NAMES},
            "rviz_config": os.path.join(gz_share, "rviz", "graspnet_visual_grasping.rviz"),
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
    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{
            "use_sim_time": launch_config["use_sim_time"],
            "calibration_name": launch_config["calibration_name"],
            "storage_directory": str(Path.home() / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim"),
        }],
        output="screen",
    )
    graspnet_visual_grasping = Node(
        package="graspnet_bringup",
        executable="graspnet_visual_grasping",
        name="graspnet_visual_grasping",
        output="screen",
        parameters=[
            _TASK_PARAMETERS,
            {
                "use_sim_time": launch_config["use_sim_time"],
                **_public_task_parameters(context),
            },
        ],
    )
    motion_control = Node(
        package="myrobot_common",
        executable="motion_control",
        name="motion_control",
        output="screen",
    )

    return [
        myrobot_simulation,
        retime_server_launch,
        hand_eye_tf_publisher,
        OpaqueFunction(function=_graspnet_inference_process),
        graspnet_visual_grasping,
        motion_control,
    ]


def generate_launch_description():
    return LaunchDescription([
        *_declare_launch_arguments(),
        OpaqueFunction(function=_launch_setup),
    ])
