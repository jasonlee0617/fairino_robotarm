from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _validate_cuda(context):
    required = (
        LaunchConfiguration("require_cuda_for_visual_perception")
        .perform(context)
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    if required:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for LLM visual perception; "
                "repair NVIDIA device access first"
            )
    return []


def generate_launch_description():
    node = Node(
        package="visual_perception",
        executable="llm_visual_perception.py",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "require_cuda_for_visual_perception": LaunchConfiguration(
                "require_cuda_for_visual_perception"
            ),
            "sync_slop": LaunchConfiguration("sync_slop"),
            "sync_watchdog_sec": LaunchConfiguration("sync_watchdog_sec"),
            "expected_camera_rate_hz": LaunchConfiguration(
                "expected_camera_rate_hz"
            ),
        }],
        on_exit=EmitEvent(event=Shutdown(reason="LLM visual perception exited")),
    )
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "require_cuda_for_visual_perception", default_value="true"
        ),
        DeclareLaunchArgument("sync_slop", default_value="0.05"),
        DeclareLaunchArgument("sync_watchdog_sec", default_value="3.0"),
        DeclareLaunchArgument("expected_camera_rate_hz", default_value="30.0"),
        OpaqueFunction(function=_validate_cuda),
        node,
    ])
