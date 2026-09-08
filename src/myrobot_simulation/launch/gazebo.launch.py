import json
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.sim_stack import base_simulation_actions  # noqa: E402
from launch_utils.launch_parsing import as_bool, spawn_pose_from_context  # noqa: E402
from launch_utils.perception_stack import camera_bridge_nodes, servo_node  # noqa: E402
from launch_utils.d435_profile import d435_mappings  # noqa: E402
from launch_utils.robot_profiles import load_robot_profile  # noqa: E402
from myrobot_common.launch_utils.yaml_loader import load_yaml  # noqa: E402


CALIBRATION_BOARD_MOUNT_DEFAULTS = {
    "calibration_board_x": "0.055",
    "calibration_board_y": "-0.050",
    "calibration_board_z": "0.2168",
    "calibration_board_roll": "0.0",
    "calibration_board_pitch": "1.5707963267948966",
    "calibration_board_yaw": "0.0",
}

# 通用 Gazebo 场景生成器的公开启动接口。场景、xacro、桥接器和控制器
# 在节点创建前就需要这些值，因此仍通过 LaunchConfiguration 传递。
_LAUNCH_ARGUMENT_SPECS = (
    ("robot_profile", "fairino_arm_gripper_onbase", "机器人配置名称。", None),
    ("world", "empty", "Gazebo 世界名称。", None),
    ("rviz_config", "", "RViz 配置文件；空值使用默认配置。", None),
    ("enable_rviz", "true", "是否启动 RViz。", None),
    ("use_sim_time", "true", "是否使用 Gazebo 的 /clock。", None),
    ("publish_frequency", "100.0", "机器人状态发布频率（Hz）。", None),
    ("initial_positions_file", "", "可选的 xacro 初始关节 YAML。", None),
    ("enable_camera_model", "false", "是否生成相机模型与传感器插件。", None),
    ("enable_camera_bridge", "false", "是否桥接相机话题。", None),
    ("enable_servo", "false", "是否启动视觉伺服。", None),
    ("camera_fps", "60", "相机帧率。", None),
    ("camera_image_width", "640", "彩色图像宽度。", None),
    ("camera_image_height", "480", "彩色图像高度。", None),
    ("camera_depth_far_m", "3.0", "D435 深度远裁剪距离（米）。", None),
    ("camera_profile", "", "命名 D435 配置；启用相机时需提供配置或文件。", None),
    ("camera_profile_file", "", "外部 D435 配置文件，与 camera_profile 二选一。", None),
    ("camera_noise_mode", "off", "相机噪声模型。", ("off", "d435_empirical")),
    ("robot_spawn_delay", "5.0", "机器人生成等待时间（秒）。", None),
    ("controller_spawn_delay", "8.0", "控制器启动等待时间（秒）。", None),
    ("planner_random_seed", "0", "规划器随机种子。", None),
    ("fairino_ik_task_profile", "grasp", "Fairino IK 任务配置。", ("grasp", "continuous")),
    ("benchmark_goal_root_mode", "", "Benchmark 目标根模式。", ("", "single_root", "multi_root")),
    ("benchmark_comparison_json", "{}", "Benchmark 公平公共参数 JSON。", None),
    ("moveit_clients", "fairino,kdl", "启动的 MoveIt client，逗号分隔。", None),
    ("spawn_name", "", "生成实体名称覆盖。", None),
    ("spawn_x", "0.0", "生成 X 坐标（米）。", None),
    ("spawn_y", "0.0", "生成 Y 坐标（米）。", None),
    ("spawn_z", "0.0", "生成 Z 坐标（米）。", None),
    ("spawn_roll", "0.0", "生成滚转角（弧度）。", None),
    ("spawn_pitch", "0.0", "生成俯仰角（弧度）。", None),
    ("spawn_yaw", "0.0", "生成偏航角（弧度）。", None),
    ("scene_assets_dir", "", "场景资源目录。", None),
    ("scene_config_file", "", "路径规划场景配置文件。", None),
    ("scene_name", "single_obstacle", "路径规划场景名称。", None),
    ("spawn_sim_scene_models", "false", "是否生成场景模型。", None),
    ("publish_planning_scene", "true", "是否发布 MoveIt 规划场景。", None),
    ("publish_obstacle_markers", "true", "是否发布障碍物标记。", None),
    ("obstacle_marker_topic", "/demo_pathplanning/obstacle_markers", "障碍物标记话题。", None),
)


def _declare_launch_arguments(gz_share):
    fixed_defaults = {
        "rviz_config": os.path.join(gz_share, "rviz", "ik_test.rviz"),
        "scene_assets_dir": os.path.join(gz_share, "config", "scenes"),
        "scene_config_file": os.path.join(gz_share, "config", "scenes", "pathplanning_scenes_params.yaml"),
        **CALIBRATION_BOARD_MOUNT_DEFAULTS,
    }
    declarations = []
    for name, default_value, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": fixed_defaults.get(name, default_value), "description": description}
        if choices:
            kwargs["choices"] = list(choices)
        declarations.append(DeclareLaunchArgument(name, **kwargs))
    declarations.extend(
        DeclareLaunchArgument(name, default_value=default, description="Eye-on-Base 标定板相对法兰安装偏置。")
        for name, default in CALIBRATION_BOARD_MOUNT_DEFAULTS.items()
    )
    return declarations


def _launch_setup(context, *args, **kwargs):
    profile = load_robot_profile(LaunchConfiguration("robot_profile").perform(context))

    spawn_name = LaunchConfiguration("spawn_name").perform(context) or profile.spawn_name
    spawn_xyz, spawn_rpy = spawn_pose_from_context(context)
    initial_positions_file = LaunchConfiguration("initial_positions_file").perform(context)
    camera_fps = LaunchConfiguration("camera_fps").perform(context)
    camera_depth_far_m = LaunchConfiguration("camera_depth_far_m").perform(context)
    extra_mappings = {
        "camera_fps": camera_fps,
        "camera_image_width": LaunchConfiguration("camera_image_width").perform(context),
        "camera_image_height": LaunchConfiguration("camera_image_height").perform(context),
        **{
            name: LaunchConfiguration(name).perform(context)
            for name in CALIBRATION_BOARD_MOUNT_DEFAULTS
        },
    }
    enable_camera_model = as_bool(LaunchConfiguration("enable_camera_model").perform(context))
    camera_profile = LaunchConfiguration("camera_profile").perform(context)
    camera_profile_file = LaunchConfiguration("camera_profile_file").perform(context)
    named_profile_selected = camera_profile.strip().lower() not in ("", "none")
    external_profile_selected = bool(camera_profile_file.strip())
    if (
        enable_camera_model
        and not named_profile_selected
        and not external_profile_selected
    ):
        raise ValueError(
            "Camera simulation requires camera_profile or camera_profile_file when "
            "enable_camera_model:=true. Select a named D435 profile instead of using "
            "nominal intrinsics."
        )
    if named_profile_selected or external_profile_selected:
        d435 = d435_mappings(
            camera_profile,
            camera_profile_file,
            LaunchConfiguration("camera_noise_mode").perform(context),
            camera_fps=camera_fps,
            camera_depth_far_m=camera_depth_far_m,
        )
        extra_mappings.update(d435)
    if initial_positions_file:
        extra_mappings["initial_positions_file"] = initial_positions_file

    use_sim_time = as_bool(LaunchConfiguration("use_sim_time").perform(context))
    moveit_clients = tuple(
        value.strip()
        for value in LaunchConfiguration("moveit_clients").perform(context).split(",")
        if value.strip()
    )
    try:
        benchmark_comparison = json.loads(
            LaunchConfiguration("benchmark_comparison_json").perform(context)
        )
    except json.JSONDecodeError as exc:
        raise ValueError("benchmark_comparison_json must be a JSON object") from exc
    if not isinstance(benchmark_comparison, dict):
        raise ValueError("benchmark_comparison_json must be a JSON object")
    actions, moveit_config = base_simulation_actions(
        profile,
        world=LaunchConfiguration("world").perform(context),
        rviz_config=LaunchConfiguration("rviz_config").perform(context),
        spawn_xyz=spawn_xyz,
        spawn_rpy=spawn_rpy,
        spawn_name=spawn_name,
        use_sim_time=use_sim_time,
        enable_rviz=as_bool(LaunchConfiguration("enable_rviz").perform(context)),
        publish_frequency=float(LaunchConfiguration("publish_frequency").perform(context)),
        enable_camera_model=enable_camera_model,
        robot_spawn_delay=float(LaunchConfiguration("robot_spawn_delay").perform(context)),
        controller_spawn_delay=float(
            LaunchConfiguration("controller_spawn_delay").perform(context)
        ),
        planner_random_seed=int(LaunchConfiguration("planner_random_seed").perform(context)),
        moveit_clients=moveit_clients,
        fairino_ik_task_profile=LaunchConfiguration("fairino_ik_task_profile").perform(context),
        benchmark_goal_root_mode=LaunchConfiguration("benchmark_goal_root_mode").perform(context),
        benchmark_comparison=benchmark_comparison,
        extra_mappings=extra_mappings,
    )
    if as_bool(LaunchConfiguration("enable_camera_bridge").perform(context)):
        actions.extend(camera_bridge_nodes(use_sim_time))
    if as_bool(LaunchConfiguration("enable_servo").perform(context)):
        kinematics_kdl_config = load_yaml(
            profile.moveit_config_package, profile.kinematics_kdl_file
        )
        actions.append(servo_node(moveit_config, profile, kinematics_kdl_config, use_sim_time))
    return actions


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    return LaunchDescription(
        [
            *_declare_launch_arguments(gz_share),
            OpaqueFunction(function=_launch_setup),
        ]
    )
