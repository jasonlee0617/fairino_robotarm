from dataclasses import dataclass
from typing import Dict

import numpy as np

from myrobot_common.utils.params import param
from visual_grasping_bringup.task.task_types import TargetType


@dataclass(frozen=True)
class GraspProfile:
    roll: float
    pitch: float
    yaw_offset: float

def load_grasp_profiles(node) -> Dict[TargetType, GraspProfile]:
    profiles = {}
    for target in TargetType:
        prefix = f"grasp.{target.value}"
        profiles[target] = GraspProfile(
            roll=float(param(node, f"{prefix}.roll", 0.0)),
            pitch=float(param(node, f"{prefix}.pitch", -180.0)),
            yaw_offset=float(param(node, f"{prefix}.yaw_offset", 0.0)),
        )
    return profiles


def _resolve_grasp_attitude(node, target: TargetType, obj_yaw_rad: float):
    obj_y_deg = float(np.degrees(obj_yaw_rad))
    profile = node.grasp_profiles[target]
    target_name = target.value

    node.get_logger().info(
        f"✓ Using {target_name} base yaw(deg): {obj_y_deg:.1f}"
    )
    return profile, target_name, profile.roll, profile.pitch, profile.yaw_offset + obj_y_deg


def build_target_poses(node, target: TargetType, obj_pos_base, obj_yaw_rad: float):
    profile, target_name, obj_roll, obj_pitch, obj_yaw = _resolve_grasp_attitude(node, target, obj_yaw_rad)

    poses = {
        "target_above": node.pose_tools.make_pose(
            obj_pos_base.x,
            obj_pos_base.y,
            obj_pos_base.z + node.grasp_above,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
        "target_grasp": node.pose_tools.make_pose(
            obj_pos_base.x,
            obj_pos_base.y,
            obj_pos_base.z + node.grasp_offset,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
        "target_lift": node.pose_tools.make_pose(
            obj_pos_base.x,
            obj_pos_base.y,
            obj_pos_base.z + node.place_offset,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
    }

    node.get_logger().info(f"✓ Target poses ready for {target_name}")
    for key, pose in poses.items():
        node.get_logger().info(
            f"  - {key}: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
        )
    return poses


def build_box_poses(node, target: TargetType, box_pos_base, obj_yaw_rad: float):
    _, target_name, obj_roll, obj_pitch, obj_yaw = _resolve_grasp_attitude(node, target, obj_yaw_rad)

    poses = {
        "box_above": node.pose_tools.make_pose(
            box_pos_base.x,
            box_pos_base.y,
            box_pos_base.z + node.place_offset,
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
        "descend_to_box": node.pose_tools.make_pose(
            box_pos_base.x,
            box_pos_base.y,
            box_pos_base.z + getattr(node, "descend_to_box", 0.07),
            obj_roll,
            obj_pitch,
            obj_yaw,
        ),
    }

    node.get_logger().info(f"✓ Box poses ready for {target_name}")
    for key, pose in poses.items():
        node.get_logger().info(
            f"  - {key}: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
        )
    return poses


def build_task_poses(node, target: TargetType, obj_pos_base, box_pos_base, obj_yaw_rad: float):
    poses = build_target_poses(node, target, obj_pos_base, obj_yaw_rad)
    poses.update(build_box_poses(node, target, box_pos_base, obj_yaw_rad))
    return poses
