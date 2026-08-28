import math

import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.duration import Duration
import tf2_geometry_msgs
import tf2_ros


class TfTools:
    """TF helper for camera->base transforms and readiness checks."""

    def __init__(self, node, base_frame: str, camera_frame: str, check_period_sec: float = 1.0):
        self._node = node
        self.base_frame = base_frame
        self.camera_frame = camera_frame

        self._buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buffer, self._node)

        self.ready = False
        self._timer = self._node.create_timer(check_period_sec, self._check_tf_ready)
        self._node.get_logger().info("TF initializing...")

    def _check_tf_ready(self):
        try:
            self._buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())
            if not self.ready:
                self.ready = True
                self._node.get_logger().info("✓ TF ready")
                self._timer.destroy()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def transform_point(self, point_stamped, target_frame: str, timeout_sec: float = 0.2):
        if not self.ready:
            return None
        try:
            ps = PoseStamped()
            ps.header = point_stamped.header
            ps.pose.position = point_stamped.point
            ps.pose.orientation.w = 1.0
            t = rclpy.time.Time.from_msg(point_stamped.header.stamp)
            tf = self._buffer.lookup_transform(
                target_frame,
                point_stamped.header.frame_id,
                t,
                timeout=Duration(seconds=float(timeout_sec)),
            )
            out = tf2_geometry_msgs.do_transform_pose_stamped(ps, tf)
            return out.pose.position
        except Exception as exc:
            self._node.get_logger().error(f"✗ TF transform failed: {exc}")
            return None

    def camera_point_to_base(self, point_stamped, timeout_sec: float = 0.2):
        return self.transform_point(point_stamped, target_frame=self.base_frame, timeout_sec=timeout_sec)

    def camera_axis_to_base(self, axis_stamped: Vector3Stamped, timeout_sec: float = 0.2):
        if not self.ready:
            return None
        try:
            t = rclpy.time.Time.from_msg(axis_stamped.header.stamp)
            transform = self._buffer.lookup_transform(
                self.base_frame,
                axis_stamped.header.frame_id,
                t,
                timeout=Duration(seconds=float(timeout_sec)),
            )
            return tf2_geometry_msgs.do_transform_vector3(axis_stamped, transform).vector
        except Exception as exc:
            self._node.get_logger().error(f"✗ TF axis transform failed: {exc}")
            return None

    def camera_axis_yaw_to_base(self, axis_stamped: Vector3Stamped, symmetry_period: float, previous_yaw=None, alpha: float = 1.0):
        axis = self.camera_axis_to_base(axis_stamped)
        if axis is None or math.hypot(axis.x, axis.y) < 1e-6:
            return None
        yaw = math.atan2(axis.y, axis.x)
        if previous_yaw is None:
            return yaw
        candidates = [yaw + k * symmetry_period for k in range(-4, 5)]
        target = min(candidates, key=lambda value: abs((value - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi))
        delta = (target - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return (previous_yaw + float(alpha) * delta + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["TfTools"]
