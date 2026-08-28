"""Camera-driver launch selection shared by robot business entries."""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def camera_launch(camera_type: str, *, realsense_args=None, oak_args=None):
    if camera_type == "oak":
        arguments = {"rs_compat": "true", **(oak_args or {})}
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("depthai_ros_driver"), "launch", "camera.launch.py"
            )),
            launch_arguments=arguments.items(),
        )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"
        )]),
        launch_arguments=(realsense_args or {}).items(),
    )
