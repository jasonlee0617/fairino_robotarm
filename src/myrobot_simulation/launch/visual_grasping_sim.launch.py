import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from myrobot_common.launch_utils.yaml_loader import (
    launch_defaults_as_strings,
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_node_parameters_yaml,
)


_TASK_PARAMETERS = load_node_parameters_yaml(
    "visual_grasping_bringup", "config/visual_grasping_params.yaml", "visual_grasping", "sim"
)
_PUBLIC_TASK_PARAMETER_NAMES = (
    "ik_plugin", "planning_pipeline_id", "planner_id", "move_group_ready_timeout_sec",
    "allow_cross_client_fallback", "arm_max_velocity", "arm_max_acceleration",
    "allowed_planning_time", "position_tolerance", "orientation_tolerance",
    "allowed_start_tolerance", "max_step_size",
)
_PUBLIC_TASK_FALLBACKS = {name: _TASK_PARAMETERS[name] for name in _PUBLIC_TASK_PARAMETER_NAMES}


# 所有可配置的标量参数都在这里声明，使 launch 接口一目了然。
# YAML 文件和模型路径则有意作为固定的文件/模型引用，放在下方代码中。
_LAUNCH_ARGUMENT_SPECS = (
    ("robot_profile", "fairino_arm_gripper_inhand", "Gazebo 机器人配置；选择机器人模型与传感器。", None),
    ("world", "visual_world", "Gazebo 世界资源。", None),
    ("enable_rviz", "true", "是否启用 Gazebo 的 RViz 实例。", None),
    ("use_sim_time", "true", "是否让所有运行时节点使用 Gazebo 的 /clock 仿真时间。", None),
    ("publish_frequency", "30.0", "机器人状态发布频率，单位 Hz。", None),
    ("enable_camera_model", "true", "是否生成仿真相机模型。", None),
    ("enable_camera_bridge", "true", "是否将相机话题桥接到 ROS 2。", None),
    ("enable_servo", "true", "是否在 Gazebo 中启动视觉伺服节点。", None),
    ("camera_fps", "30", "仿真相机帧率。", None),
    ("camera_image_width", "1280", "仿真彩色图像宽度。", None),
    ("camera_image_height", "720", "仿真彩色图像高度。", None),
    (
        "camera_profile",
        "d435_color_1280x720x30_depth_848x480x30",
        "用于视觉抓取仿真相机的命名 D435 配置文件。",
        None,
    ),
    (
        "camera_profile_file",
        "",
        "外部 D435 配置文件 YAML；使用时需要将 camera_profile 设为空字符串。",
        None,
    ),
    ("camera_noise_mode", "off", "相机噪声模型。", ("off", "d435_empirical")),
    ("camera_depth_far_m", "3.0", "D435 深度远裁剪距离（米），最大可设为 10.0。", None),
    ("spawn_z", "1.02", "机器人初始高度，单位米。", None),
    ("controller_spawn_delay", "5.0", "控制器启动前的等待时间，单位秒。", None),
    ("calibration_name", "robot_calibration", "手眼标定名称。", None),
    ("camera_mode", "eye_in_hand", "视觉抓取相机模式。", None),
    *(
        (
            name,
            str(default).lower() if isinstance(default, bool) else str(default),
            "视觉抓取规划运行参数；默认值来自 visual_grasping_params.yaml。",
            None,
        )
        for name, default in _PUBLIC_TASK_FALLBACKS.items()
    ),
)
_YAML_LAUNCH_DEFAULTS = launch_defaults_as_strings(
    load_launch_parameters_yaml("visual_grasping_bringup", "config/visual_grasping_params.yaml", "sim")
)
_LAUNCH_ARGUMENT_SPECS = tuple(
    (name, _YAML_LAUNCH_DEFAULTS.get(name, default), description, choices)
    for name, default, description, choices in _LAUNCH_ARGUMENT_SPECS
)
# 前 17 个参数属于场景/仿真相关参数，会传递给被包含的 Gazebo launch 文件
_SCENE_ARGUMENT_NAMES = tuple(name for name, *_ in _LAUNCH_ARGUMENT_SPECS[:17])
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name, *_ in _LAUNCH_ARGUMENT_SPECS
}


def _declare_launch_arguments():
    """根据规格表创建所有参数的声明."""
    declarations = []
    for name, default_value, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": default_value, "description": description}
        if choices is not None:
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
    launch_config = _LAUNCH_CONFIGURATIONS

    # 包含的子 launch 文件参数均使用 LaunchConfiguration 传递，
    # 从而保证命令行覆盖（例如 --ros-args -p use_sim_time:=false）仍然有效。
    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            **{
                name: launch_config[name]
                for name in _SCENE_ARGUMENT_NAMES
            },
            "rviz_config": os.path.join(
                gz_share, "rviz", "visual_grasping_sim.rviz"
            ),
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
        parameters=[
            {
                "use_sim_time": launch_config["use_sim_time"],
                "calibration_name": launch_config["calibration_name"],
                "storage_directory": "/home/robot/fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim",
            },
        ],
        output="screen",
    )
    yolo_obb = Node(
        package="visual_perception",
        executable="yolo_detector_obb.py",
        name="yolo_detector_obb",
        output="screen",
        parameters=[
            load_node_parameters_yaml(
                "visual_grasping_bringup", "config/visual_grasping_params.yaml", "yolo_detector_obb", "sim"
            ),
            {
                "use_sim_time": launch_config["use_sim_time"],
                "model_path": "yolo-obb-1280.pt",
            },
        ],
    )
    visual_grasping = Node(
        package="visual_grasping_bringup",
        executable="visual_grasping",
        name="visual_grasping",
        output="screen",
        parameters=[
            _TASK_PARAMETERS,
            {
                "use_sim_time": launch_config["use_sim_time"],
                "camera_mode": launch_config["camera_mode"],
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
        yolo_obb,
        visual_grasping,
        motion_control,
    ]


def generate_launch_description():
    return LaunchDescription([
        *_declare_launch_arguments(),
        OpaqueFunction(function=_launch_setup),
    ])
