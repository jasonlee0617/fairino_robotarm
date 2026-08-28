from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


_TARGET_POS_FIELDS = {
    "elongated_object": "elongated_object_pos",
    "cube": "cube_pos",
    "box": "box_pos",
    "stone": "stone_pos",
}
_TARGET_AXIS_FIELDS = {
    "elongated_object": "elongated_object_axis",
    "cube": "cube_axis",
    "stone": "stone_axis",
}


def _target_key(target) -> str:
    """Normalise a TargetType enum member / int / str to a lowercase key."""
    try:
        return str(target.name).lower()
    except AttributeError:
        pass
    try:
        return str(target).lower()
    except Exception:
        pass
    return ""


@dataclass
class DetectionCache:
    elongated_object_pos: Optional[PointStamped] = field(default=None)
    cube_pos: Optional[PointStamped] = field(default=None)
    box_pos: Optional[PointStamped] = field(default=None)
    stone_pos: Optional[PointStamped] = field(default=None)
    elongated_object_axis: Optional[Vector3Stamped] = field(default=None)
    cube_axis: Optional[Vector3Stamped] = field(default=None)
    stone_axis: Optional[Vector3Stamped] = field(default=None)

    # ── constructor accepts optional node (for visual_grasping_bringup compat) ──
    def __init__(self, node=None):
        self.elongated_object_pos = None
        self.cube_pos = None
        self.box_pos = None
        self.stone_pos = None
        self.elongated_object_axis = None
        self.cube_axis = None
        self.stone_axis = None

    # ── visual_grasping_bringup API ──
    def reset(self):
        self.elongated_object_pos = None
        self.cube_pos = None
        self.box_pos = None
        self.stone_pos = None
        self.elongated_object_axis = None
        self.cube_axis = None
        self.stone_axis = None

    def get_position(self, target) -> Optional[PointStamped]:
        key = _TARGET_POS_FIELDS.get(_target_key(target))
        return getattr(self, key, None) if key else None

    def get_axis(self, target) -> Optional[Vector3Stamped]:
        key = _TARGET_AXIS_FIELDS.get(_target_key(target))
        return getattr(self, key, None) if key else None

    @staticmethod
    def pair_valid(position: PointStamped, axis: Vector3Stamped) -> bool:
        if position is None or axis is None:
            return False
        return (
            position.header.frame_id == axis.header.frame_id
            and position.header.stamp.sec == axis.header.stamp.sec
            and position.header.stamp.nanosec == axis.header.stamp.nanosec
        )

    # ── public callbacks (used by both DetectionSubscribers and direct bindings) ──
    def on_elongated_object_pos(self, msg: PointStamped):
        self.elongated_object_pos = msg
        if not self.pair_valid(msg, self.elongated_object_axis):
            self.elongated_object_axis = None

    def on_cube_pos(self, msg: PointStamped):
        self.cube_pos = msg
        if not self.pair_valid(msg, self.cube_axis):
            self.cube_axis = None

    def on_box_pos(self, msg: PointStamped):
        self.box_pos = msg

    def on_stone_pos(self, msg: PointStamped):
        self.stone_pos = msg
        if not self.pair_valid(msg, self.stone_axis):
            self.stone_axis = None

    def on_elongated_object_axis(self, msg: Vector3Stamped):
        self.elongated_object_axis = msg

    def on_cube_axis(self, msg: Vector3Stamped):
        self.cube_axis = msg

    def on_stone_axis(self, msg: Vector3Stamped):
        self.stone_axis = msg


class DetectionSubscribers:
    def __init__(self, node, cache: DetectionCache):
        self.node = node
        self.cache = cache
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        node.create_subscription(
            PointStamped,
            "/elongated_object_position_3d",
            cache.on_elongated_object_pos,
            qos,
        )
        node.create_subscription(PointStamped, "/cube_position_3d", cache.on_cube_pos, qos)
        node.create_subscription(PointStamped, "/box_position_3d", cache.on_box_pos, qos)
        node.create_subscription(PointStamped, "/stone_position_3d", cache.on_stone_pos, qos)
        node.create_subscription(Vector3Stamped, "/elongated_object_axis_3d", cache.on_elongated_object_axis, qos)
        node.create_subscription(Vector3Stamped, "/cube_axis_3d", cache.on_cube_axis, qos)
        node.create_subscription(Vector3Stamped, "/stone_axis_3d", cache.on_stone_axis, qos)
        node.get_logger().info("Detection subscribers set")


__all__ = ["DetectionCache", "DetectionSubscribers"]
