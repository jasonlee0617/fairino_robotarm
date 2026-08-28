#!/usr/bin/env python3
import os
import yaml
import sys
import math
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.moveit_stack import build_moveit_config  # noqa: E402
from launch_utils.d435_profile import d435_mappings  # noqa: E402
from launch_utils.robot_profiles import load_robot_profile  # noqa: E402
from myrobot_common.launch_utils.yaml_loader import (  # noqa: E402
    launch_parameter_value,
    load_launch_parameters_yaml,
    load_yaml,
)
from visual_servo_bringup.position_servo_config import (  # noqa: E402
    aruco_detector_parameters,
    aruco_parameters,
    aruco_pose_source_parameters,
    sim_camera_defaults,
    sim_target_motion_parameters,
    visual_servo_parameters,
    yolo_kalman_parameters,
)


_VISUAL_SERVO_PARAMS = visual_servo_parameters("sim")
_SIM_TARGET_MOTION = sim_target_motion_parameters(_VISUAL_SERVO_PARAMS["perception_source"])
_PUBLIC_NODE_PARAMETER_NAMES = (
    "open_gripper_after_home", "ik_plugin", "planning_pipeline_id", "planner_id",
    "move_group_ready_timeout_sec", "allow_cross_client_fallback",
    "arm_max_velocity", "arm_max_acceleration", "allowed_planning_time",
    "position_tolerance", "orientation_tolerance", "allowed_start_tolerance",
    "v_xyz_max", "a_xyz_max", "twist_norm_max",
)
_PUBLIC_NODE_FALLBACKS = {
    name: _VISUAL_SERVO_PARAMS[name] for name in _PUBLIC_NODE_PARAMETER_NAMES
}
# 公开场景参数集中管理。固定 YAML、模型和 RViz 文件仍由包共享目录定位。
_LAUNCH_ARGUMENT_SPECS = (
    ("use_sim_time", "true", "是否使用 Gazebo 的 /clock。", None),
    ("camera_profile", "d435_color_640x480x60_depth_640x480x60", "命名 D435 配置。", None),
    ("camera_profile_file", "", "外部 D435 配置文件；使用时清空 camera_profile。", None),
    ("camera_noise_mode", "off", "相机噪声模型。", ("off", "d435_empirical")),
    ("camera_depth_far_m", "10.0", "D435 深度远裁剪距离（米）。", None),
    ("camera_fps", "60", "仿真相机帧率。", None),
    ("camera_image_width", "640", "仿真彩色图像宽度。", None),
    ("camera_image_height", "480", "仿真彩色图像高度。", None),
    ("robot_profile", "fairino_arm_gripper_onbase", "Gazebo 机器人配置。", ("fairino_arm_gripper_onbase", "fairino_arm_gripper_inhand", "fairino3_v6")),
    *(
        (
            name,
            str(default).lower() if isinstance(default, bool) else str(default),
            "位置伺服运行参数；默认值来自 visual_position_servo_params.yaml。",
            None,
        )
        for name, default in _PUBLIC_NODE_FALLBACKS.items()
    ),
)
_YAML_LAUNCH_DEFAULTS = {
    **{
        name: str(value).lower() if isinstance(value, bool) else str(value)
        for name, value in load_launch_parameters_yaml(
            "visual_servo_bringup", "config/visual_position_servo_params.yaml", "sim"
        ).items()
        if not isinstance(value, dict)
    },
    **{
        name: str(value).lower() if isinstance(value, bool) else str(value)
        for name, value in sim_camera_defaults().items()
    },
}
_LAUNCH_ARGUMENT_SPECS = tuple(
    (name, _YAML_LAUNCH_DEFAULTS.get(name, default), description, choices)
    for name, default, description, choices in _LAUNCH_ARGUMENT_SPECS
)


def _declare_launch_arguments():
    declarations = []
    for name, default_value, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": default_value, "description": description}
        if choices:
            kwargs["choices"] = list(choices)
        declarations.append(DeclareLaunchArgument(name, **kwargs))
    return declarations


def _sim_yolo_parameters() -> dict:
    params = yolo_kalman_parameters()
    models_dir = os.path.join(get_package_share_directory("visual_perception"), "models")
    for name in ("model_path", "engine_path"):
        params[name] = os.path.join(models_dir, params[name])
    return params


def _typed_launch_value(context, name: str, fallback):
    return launch_parameter_value(LaunchConfiguration(name).perform(context), fallback)


def _public_node_parameters(context) -> dict:
    return {
        name: _typed_launch_value(context, name, fallback)
        for name, fallback in _PUBLIC_NODE_FALLBACKS.items()
    }


def _launch_setup(context, *args, **kwargs):
    robot_profile = LaunchConfiguration("robot_profile").perform(context)
    camera_profile = LaunchConfiguration("camera_profile").perform(context)
    camera_profile_file = LaunchConfiguration("camera_profile_file").perform(context)
    camera_noise_mode = LaunchConfiguration("camera_noise_mode").perform(context)
    camera_depth_far_m = LaunchConfiguration("camera_depth_far_m").perform(context)
    visual_servo_params = dict(_VISUAL_SERVO_PARAMS)
    visual_servo_params.update(_public_node_parameters(context))
    aruco_params = aruco_parameters()
    camera_mappings = {
        "camera_fps": LaunchConfiguration("camera_fps").perform(context),
        "camera_image_width": LaunchConfiguration("camera_image_width").perform(context),
        "camera_image_height": LaunchConfiguration("camera_image_height").perform(context),
    }
    camera_mappings.update(
        d435_mappings(
            camera_profile,
            camera_profile_file,
            camera_noise_mode,
            camera_fps=camera_mappings["camera_fps"],
            camera_depth_far_m=camera_depth_far_m,
        )
    )
    camera_profile_summary = LogInfo(
        msg=(
            "D435 camera profile: "
            f"source={camera_profile or camera_profile_file}, "
            f"color={camera_mappings['camera_image_width']}x"
            f"{camera_mappings['camera_image_height']}@"
            f"{camera_mappings['camera_fps']}Hz, "
            f"fx={float(camera_mappings['camera_fx']):.3f}, "
            f"fy={float(camera_mappings['camera_fy']):.3f}, "
            f"cx={float(camera_mappings['camera_cx']):.3f}, "
            f"cy={float(camera_mappings['camera_cy']):.3f}, "
            f"h_fov={math.degrees(float(camera_mappings['camera_h_fov'])):.3f}deg, "
            f"v_fov={math.degrees(float(camera_mappings['camera_v_fov'])):.3f}deg"
        )
    )
    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('myrobot_simulation') + '/launch/gazebo.launch.py']),
        launch_arguments={
            "robot_profile": robot_profile,
            "world": "robotarm_world",
            "rviz_config": os.path.join(
                get_package_share_directory("visual_servo_bringup"),
                "rviz",
                "visual_position_servo.rviz",
            ),
            "publish_frequency": "100.0",
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "enable_servo": "true",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "camera_fps": camera_mappings["camera_fps"],
            "camera_image_width": camera_mappings["camera_image_width"],
            "camera_image_height": camera_mappings["camera_image_height"],
            "camera_profile": camera_profile,
            "camera_profile_file": camera_profile_file,
            "camera_noise_mode": camera_noise_mode,
            "camera_depth_far_m": camera_mappings["camera_depth_far_m"],
            "spawn_z": "1.02",
            "controller_spawn_delay": "5.0",
        }.items(),
    )

    profile = load_robot_profile(robot_profile)
    moveit_config = build_moveit_config(
        profile,
        enable_camera_model=True,
        extra_mappings=camera_mappings,
    )
    cartesian_path_planner_params = load_yaml(
        "myrobot_planning_core", "config/cartesian_path_planner_params.yaml"
    )

    retime_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        ),
        launch_arguments={
            "robot_description": moveit_config.robot_description["robot_description"],
            "robot_description_semantic": moveit_config.robot_description_semantic[
                "robot_description_semantic"
            ],
            "robot_description_kinematics": yaml.safe_dump(
                moveit_config.robot_description_kinematics[
                    "robot_description_kinematics"
                ]
            ),
        }.items(),
    )
    visual_position_servo_node = TimerAction(
        # gazebo.launch.py starts the controller spawners at 5 s.  Start only
        # after their sequential activation, otherwise HOME races the action servers.
        period=10.0,
        actions=[
            Node(
                package='visual_servo_bringup',
                executable='visual_position_servo',
                name='visual_position_servo_node',
                output='screen',
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                    visual_servo_params,
                    cartesian_path_planner_params,
                ],
            )
        ]
    )
    source_actions = []
    if _VISUAL_SERVO_PARAMS["perception_source"] == "yolo_kalman":
        source_actions.extend([
            Node(
                package='ros_gz_bridge', executable='parameter_bridge',
                arguments=[f"{_SIM_TARGET_MOTION['cmd_topic']}@geometry_msgs/msg/Twist@gz.msgs.Twist"],
                output='screen', parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            ),
            TimerAction(period=2.0, actions=[Node(
                package='myrobot_simulation', executable='target_motion_controller_node.py',
                name='cube_target_motion_controller', output='screen', parameters=[{
                    'use_sim_time': LaunchConfiguration("use_sim_time"), **_SIM_TARGET_MOTION,
                }],
            )]),
            TimerAction(period=3.0, actions=[Node(
                package='visual_perception', executable='yolo_kalman_detector_obb.py',
                name='yolo_kalman_detector_obb', output='screen',
                parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}, _sim_yolo_parameters()],
            )]),
        ])
    else:
        source_actions.extend([
            Node(
                package='ros_gz_bridge', executable='parameter_bridge',
                arguments=[f"{_SIM_TARGET_MOTION['cmd_topic']}@geometry_msgs/msg/Twist@gz.msgs.Twist"],
                output='screen', parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            ),
            TimerAction(period=4.0, actions=[Node(
                package='ros2_aruco', executable='aruco_node', name='aruco_node',
                output='screen', parameters=[
                    {'use_sim_time': LaunchConfiguration("use_sim_time")},
                    aruco_detector_parameters(aruco_params),
                ],
            )]),
            TimerAction(period=4.5, actions=[Node(
                package='hand_eye_calibration', executable='aruco_marker_pose_publisher.py',
                name='aruco_marker_pose_publisher', output='screen', parameters=[
                    {'use_sim_time': LaunchConfiguration("use_sim_time")},
                    aruco_pose_source_parameters(aruco_params),
                ],
            )]),
            TimerAction(period=4.0, actions=[Node(
                package='myrobot_simulation', executable='target_motion_controller_node.py',
                name='aruco_marker_target_motion_controller', output='screen', parameters=[{
                    'use_sim_time': LaunchConfiguration("use_sim_time"), **_SIM_TARGET_MOTION,
                }],
            )]),
        ])
    return [
        camera_profile_summary,
        myrobot_simulation,
        retime_server_launch,
        *source_actions,
        visual_position_servo_node,
    ]


def generate_launch_description():
    return LaunchDescription([*_declare_launch_arguments(), OpaqueFunction(function=_launch_setup)])
