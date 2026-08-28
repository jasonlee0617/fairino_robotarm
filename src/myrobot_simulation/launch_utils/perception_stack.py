"""Perception and servo helpers for Gazebo launch files."""

from launch_ros.actions import Node

from .robot_profiles import RobotProfile
from myrobot_common.launch_utils.yaml_loader import load_yaml, wrap_yaml_as_ros_params_file


def _camera_info_bridge(source: str, target: str, use_sim_time: bool) -> Node:
    return Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[f"{source}@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo"],
        remappings=[(source, target)],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )


def camera_bridge_nodes(use_sim_time: bool):
    color_info_source = "/camera/camera_info"
    color_info_target = "/camera/camera/color/camera_info"
    aligned_depth_info_source = "/camera/aligned_depth/camera_info"
    aligned_depth_info_target = "/camera/camera/aligned_depth_to_color/camera_info"
    return [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/image_raw@sensor_msgs/msg/Image@gz.msgs.Image"],
            output="screen",
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/camera/image@sensor_msgs/msg/Image@ignition.msgs.Image",
                "/camera/aligned_depth/image@sensor_msgs/msg/Image@ignition.msgs.Image",
                "/camera/native_depth/image@sensor_msgs/msg/Image@ignition.msgs.Image",
                "/camera/native_depth/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            ],
            remappings=[
                ("/camera/image", "/camera/camera/color/image_raw"),
                (
                    "/camera/aligned_depth/image",
                    "/camera/camera/aligned_depth_to_color/image_raw",
                ),
                ("/camera/native_depth/image", "/camera/camera/depth/image_rect_raw"),
                ("/camera/native_depth/camera_info", "/camera/camera/depth/camera_info"),
            ],
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        _camera_info_bridge(color_info_source, color_info_target, use_sim_time),
        _camera_info_bridge(aligned_depth_info_source, aligned_depth_info_target, use_sim_time),
    ]


def servo_node(moveit_config, profile: RobotProfile, kinematics_kdl_config, use_sim_time: bool):
    servo_yaml = load_yaml(profile.moveit_config_package, profile.servo_parameters_file)
    sensors_3d_params = wrap_yaml_as_ros_params_file(
        profile.moveit_config_package, "config/sensors_3d.yaml"
    )
    servo_yaml["move_group_name"] = profile.group_name
    servo_yaml["planning_frame"] = profile.planning_frame
    servo_yaml["ee_frame_name"] = profile.ee_frame_name
    servo_yaml["command_out_topic"] = f"{profile.arm_controller}/joint_trajectory"
    servo_yaml["command_out_type"] = "trajectory_msgs/JointTrajectory"
    servo_yaml["cartesian_command_in_topic"] = "/servo_node/delta_twist_cmds"
    servo_yaml["joint_command_in_topic"] = "/servo_node/delta_joint_cmds"
    # Reuse the fairino move_group planning scene so Servo sees Octomap updates
    # without maintaining a second occupancy map monitor instance.
    servo_yaml["is_primary_planning_scene_monitor"] = False
    servo_yaml["monitored_planning_scene_topic"] = "/move_group_fairino/monitored_planning_scene"

    return Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            kinematics_kdl_config,
            sensors_3d_params,
            {"moveit_servo": servo_yaml},
            {"use_sim_time": use_sim_time},
        ],
    )
