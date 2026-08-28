from __future__ import annotations

from typing import List, Optional, Sequence, Union

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped


class TargetSelector:
    """Compatible superset for both visual_servo_bringup (explicit pos/axis args) and
    visual_grasping_bringup (cache-based) callers."""

    def __init__(
        self,
        node,
        detection_timeout: Union[float, Sequence[str]] = 3.0,
        preferred_target: str = "elongated_object",
    ):
        self.node = node
        if isinstance(detection_timeout, (list, tuple)):
            # visual_grasping_bringup: TargetSelector(node, ["elongated_object", "cube", "stone"])
            self.detection_timeout = 3.0
            self.target_priority = [str(x).lower().strip() for x in detection_timeout]
            self.preferred_target = str(preferred_target).lower().strip()
        else:
            # visual_servo_bringup: TargetSelector(node, 3.0, "elongated_object")
            #         or: TargetSelector(node=self, detection_timeout=3.0, preferred_target="cube")
            self.detection_timeout = float(detection_timeout)
            self.target_priority = None
            self.preferred_target = str(preferred_target).lower().strip()

    def set_preference(self, preferred_target: str):
        self.preferred_target = str(preferred_target).lower().strip()

    def set_timeout(self, detection_timeout: float):
        self.detection_timeout = float(detection_timeout)

    def msg_age_sec(self, stamp) -> float:
        now = self.node.get_clock().now()
        t = rclpy.time.Time.from_msg(stamp)
        return (now - t).nanoseconds / 1e9

    def pair_valid(self, obj_pos: PointStamped, obj_axis: Vector3Stamped) -> bool:
        if obj_pos is None or obj_axis is None:
            return False
        return (
            obj_pos.header.frame_id == obj_axis.header.frame_id
            and obj_pos.header.stamp.sec == obj_axis.header.stamp.sec
            and obj_pos.header.stamp.nanosec == obj_axis.header.stamp.nanosec
            and self.msg_age_sec(obj_pos.header.stamp) < self.detection_timeout
        )

    def _resolve_priority(self) -> List[str]:
        """Return the resolved priority list with preferred_target first."""
        if self.target_priority is None:
            return [self.preferred_target]
        if self.preferred_target in self.target_priority:
            ordered = [self.preferred_target]
            for name in self.target_priority:
                if name != self.preferred_target:
                    ordered.append(name)
            return ordered
        return list(self.target_priority)

    # ── unified entry point ────────────────────────────────────────────
    def select_target(self, TargetType, *args, **kwargs):
        if len(args) == 4:
            return self._select_from_explicit(TargetType, *args)
        if len(args) == 1:
            return self._select_from_cache(TargetType, args[0])
        cache = kwargs.get("cache")
        if cache is not None:
            return self._select_from_cache(TargetType, cache)
        elongated_object_pos = kwargs.get("elongated_object_pos")
        elongated_object_axis = kwargs.get("elongated_object_axis")
        cube_pos = kwargs.get("cube_pos")
        cube_axis = kwargs.get("cube_axis")
        if any(
            v is not None
            for v in (elongated_object_pos, elongated_object_axis, cube_pos, cube_axis)
        ):
            return self._select_from_explicit(
                TargetType,
                elongated_object_pos,
                elongated_object_axis,
                cube_pos,
                cube_axis,
            )
        return None

    def _select_from_explicit(
        self,
        TargetType,
        elongated_object_pos,
        elongated_object_axis,
        cube_pos,
        cube_axis,
    ):
        elongated_object_ok = self.pair_valid(
            elongated_object_pos, elongated_object_axis
        )
        cube_ok = self.pair_valid(cube_pos, cube_axis)
        if elongated_object_ok and not cube_ok:
            return TargetType.ELONGATED_OBJECT
        if cube_ok and not elongated_object_ok:
            return TargetType.CUBE
        if elongated_object_ok and cube_ok:
            return (
                TargetType.CUBE
                if self.preferred_target == "cube"
                else TargetType.ELONGATED_OBJECT
            )
        return None

    def _select_from_cache(self, TargetType, cache):
        priority = self._resolve_priority()

        for name in priority:
            pos = cache.get_position(name)
            axis = cache.get_axis(name)
            if not self.pair_valid(pos, axis):
                continue
            # must also have box_pos fresh for grasp logic downstream
            box_pos = cache.box_pos
            if box_pos is None:
                continue

            pos_age = self.msg_age_sec(pos.header.stamp)
            box_age = self.msg_age_sec(box_pos.header.stamp)
            if pos_age < self.detection_timeout and box_age < self.detection_timeout:
                return getattr(TargetType, name.upper(), None)

        return None


__all__ = ["TargetSelector"]
