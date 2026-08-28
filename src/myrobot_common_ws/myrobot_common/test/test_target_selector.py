#!/usr/bin/env python3
"""Regression tests for stamped position/axis target selection."""

import pytest
import rclpy
from geometry_msgs.msg import Point, PointStamped, Vector3, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Header

from myrobot_common.perception.detection_cache import DetectionCache, DetectionSubscribers
from myrobot_common.perception.target_selector import TargetSelector


class _FakeTargetType:
    ELONGATED_OBJECT = "ELONGATED_OBJECT"
    CUBE = "CUBE"
    BOX = "BOX"
    STONE = "STONE"


class _Logger:
    def info(self, _message):
        pass


class _SubscriptionNode:
    def __init__(self):
        self.topics = []

    def create_subscription(self, _msg_type, topic, _callback, _qos):
        self.topics.append(topic)

    def get_logger(self):
        return _Logger()


@pytest.fixture(scope="module")
def ros_node():
    rclpy.init(args=[])
    node = Node("_test_target_selector")
    yield node
    node.destroy_node()
    rclpy.shutdown()


def _header(stamp_sec_offset=0.0, frame="cam"):
    now = rclpy.clock.Clock().now()
    stamp = rclpy.time.Time(seconds=now.nanoseconds * 1e-9 + stamp_sec_offset).to_msg()
    return Header(stamp=stamp, frame_id=frame)


def _make_pos(x, y, z, stamp_sec_offset=0.0, header=None):
    return PointStamped(header=header or _header(stamp_sec_offset), point=Point(x=float(x), y=float(y), z=float(z)))


def _make_axis(header, x=1.0, y=0.0, z=0.0):
    return Vector3Stamped(header=header, vector=Vector3(x=float(x), y=float(y), z=float(z)))


class TestTargetSelector:
    def test_detection_cache_uses_stamped_axis_fields(self):
        cache = DetectionCache()
        position = _make_pos(0, 0, 1)
        axis = _make_axis(position.header)
        cache.on_elongated_object_pos(position)
        cache.on_elongated_object_axis(axis)
        assert cache.get_position("elongated_object") is position
        assert cache.get_axis("elongated_object") is axis

    def test_position_axis_header_mismatch_is_rejected(self):
        cache = DetectionCache()
        position = _make_pos(0, 0, 1)
        cache.on_elongated_object_pos(position)
        cache.on_elongated_object_axis(_make_axis(_header(frame="other")))
        assert cache.get_axis("elongated_object") is not None
        assert not cache.pair_valid(position, cache.get_axis("elongated_object"))

    def test_detection_subscribers_use_axis_topics(self):
        node = _SubscriptionNode()
        DetectionSubscribers(node, DetectionCache())
        assert set(node.topics) == {
            "/elongated_object_position_3d", "/cube_position_3d", "/box_position_3d", "/stone_position_3d",
            "/elongated_object_axis_3d", "/cube_axis_3d", "/stone_axis_3d",
        }

    def test_cache_mode_requires_matching_axis_and_fresh_box(self, ros_node):
        cache = DetectionCache()
        position = _make_pos(0, 0, 1)
        cache.on_elongated_object_pos(position)
        cache.on_elongated_object_axis(_make_axis(position.header))
        cache.box_pos = _make_pos(0, 0, 1)
        assert TargetSelector(ros_node, ["elongated_object"]).select_target(_FakeTargetType, cache) == _FakeTargetType.ELONGATED_OBJECT
        cache.box_pos = _make_pos(0, 0, 1, stamp_sec_offset=-100.0)
        assert TargetSelector(ros_node, ["elongated_object"]).select_target(_FakeTargetType, cache) is None

    def test_cache_mode_prefers_configured_target(self, ros_node):
        cache = DetectionCache()
        elongated = _make_pos(0, 0, 1)
        cube = _make_pos(1, 0, 1)
        cache.on_elongated_object_pos(elongated)
        cache.on_elongated_object_axis(_make_axis(elongated.header))
        cache.on_cube_pos(cube)
        cache.on_cube_axis(_make_axis(cube.header))
        cache.box_pos = _make_pos(0, 0, 1)
        selector = TargetSelector(ros_node, ["elongated_object", "cube", "stone"])
        selector.set_preference("cube")
        assert selector.select_target(_FakeTargetType, cache) == _FakeTargetType.CUBE
