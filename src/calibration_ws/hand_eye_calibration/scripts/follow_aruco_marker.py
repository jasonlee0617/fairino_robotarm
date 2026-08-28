#!/usr/bin/env python3
"""Low-rate global MoveIt follower for the shared ArUco marker pose."""

import math
import threading
import time

import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from myrobot_common.planning.motion_executor import MoveItMotion
from myrobot_common.utils.params import param
from myrobot_common.utils.pose_tools import PoseTools
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose


_JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


class ArucoMarkerFollower(Node):
    """Track the latest ArUco target with one global MoveIt plan at a time."""

    def __init__(self):
        super().__init__("aruco_marker_follower")
        self.base_frame = self._string("base_frame", "base_link")
        self.ee_frame = self._string("ee_frame", "tool0")
        self.marker_pose_topic = self._string("marker_pose_topic", "/aruco_marker/pose")
        self.ik_plugin = self._string("ik_plugin", "fairino")
        self.above_offset = self._float("above_offset", 0.12)
        self.target_rpy_deg = self._float_list("target_rpy_deg", [-45.0, -180.0, 0.0], 3)
        self.min_replan_translation_m = self._float("min_replan_translation_m", 0.02)
        self.min_replan_interval_sec = self._float("min_replan_interval_sec", 0.5)
        self.marker_pose_timeout_sec = self._float("marker_pose_timeout_sec", 0.5)
        self.arm_max_velocity = self._float("arm_max_velocity", 0.2)
        self.arm_max_acceleration = self._float("arm_max_acceleration", 0.2)
        self.allowed_planning_time = self._float("allowed_planning_time", 15.0)
        self.position_tolerance = self._float("position_tolerance", 0.005)
        self.orientation_tolerance = self._float("orientation_tolerance", 0.005)
        self.allowed_start_tolerance = self._float("allowed_start_tolerance", 0.1)
        self.move_group_ready_timeout_sec = self._float("move_group_ready_timeout_sec", 10.0)
        self.motion_timeout_sec = self._float("motion_timeout_sec", 30.0)
        self.arm_group_name = self._string("arm_group_name", "robot_arm")

        callback_group = ReentrantCallbackGroup()
        self._arms = {
            "fairino": MoveIt2(
                node=self,
                joint_names=_JOINT_NAMES,
                base_link_name=self.base_frame,
                end_effector_name=self.ee_frame,
                group_name=self.arm_group_name,
                move_group_namespace=self._string("move_group_ns_fairino", "/move_group_fairino"),
                callback_group=callback_group,
            ),
            "kdl": MoveIt2(
                node=self,
                joint_names=_JOINT_NAMES,
                base_link_name=self.base_frame,
                end_effector_name=self.ee_frame,
                group_name=self.arm_group_name,
                move_group_namespace=self._string("move_group_ns_kdl", "/move_group_kdl"),
                callback_group=callback_group,
            ),
        }
        self.pose_tools = PoseTools(self, self.base_frame)
        self.motion = MoveItMotion(
            self,
            arm_clients=self._arms,
            default_client=self.ik_plugin,
            pose_tools=self.pose_tools,
            action_delay=0.0,
        )
        self.motion.set_ik(self.ik_plugin)
        self.motion.set_planner(
            self._string("planning_pipeline_id", "fairino"),
            self._string("planner_id", "tube_birrt*"),
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        latest_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose", latest_qos)
        self.target_pose_pub = self.create_publisher(PoseStamped, "/follow_aruco_target_pose", latest_qos)
        self.create_subscription(PoseStamped, self.marker_pose_topic, self._on_marker_pose, latest_qos)
        self._lock = threading.Lock()
        self._latest_target = None
        self._latest_target_at = 0.0
        self._last_goal_xyz = None
        self._last_submit_time = 0.0
        self._worker_running = False

    def _string(self, name, default):
        return str(param(self, name, default))

    def _float(self, name, default):
        value = float(param(self, name, default))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _float_list(self, name, default, length):
        values = [float(value) for value in param(self, name, default)]
        if len(values) != length or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} must contain {length} finite values")
        return values

    def _on_marker_pose(self, msg: PoseStamped) -> None:
        if not msg.header.frame_id:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp)
            )
            marker = PoseStamped()
            marker.header.frame_id = self.base_frame
            marker.header.stamp = msg.header.stamp
            marker.pose = do_transform_pose(msg.pose, transform)
        except Exception as exc:
            self.get_logger().warn(f"Marker TF unavailable: {exc}", throttle_duration_sec=2.0)
            return

        target = self._target_from_marker(marker)
        self.pose_pub.publish(marker)
        self.target_pose_pub.publish(target)
        with self._lock:
            self._latest_target = target
            self._latest_target_at = time.monotonic()
            if self._worker_running:
                return
            self._worker_running = True
        threading.Thread(target=self._follow_latest_targets, daemon=True).start()

    def _target_from_marker(self, marker: PoseStamped) -> PoseStamped:
        roll, pitch, yaw = self.target_rpy_deg
        target = self.pose_tools.to_pose_stamped(
            self.pose_tools.make_pose(
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z + self.above_offset,
                roll,
                pitch,
                yaw,
            ),
            self.base_frame,
        )
        return target

    @staticmethod
    def _xyz(target: PoseStamped):
        position = target.pose.position
        return position.x, position.y, position.z

    def _next_goal(self):
        with self._lock:
            target = self._latest_target
            age = time.monotonic() - self._latest_target_at
        if target is None or age > self.marker_pose_timeout_sec:
            return None
        xyz = self._xyz(target)
        if time.monotonic() - self._last_submit_time < self.min_replan_interval_sec:
            return None
        if self._last_goal_xyz is not None and math.dist(xyz, self._last_goal_xyz) < self.min_replan_translation_m:
            return None
        return target, xyz

    def _follow_latest_targets(self) -> None:
        try:
            if not self.motion.wait_client_ready(
                planning_client=self.ik_plugin,
                timeout_sec=self.move_group_ready_timeout_sec,
            ):
                return
            while rclpy.ok():
                next_goal = self._next_goal()
                if next_goal is None:
                    return
                target, xyz = next_goal
                self._last_goal_xyz = xyz
                self._last_submit_time = time.monotonic()
                self.motion.move_to_pose(
                    target,
                    planning_client=self.ik_plugin,
                    cartesian=False,
                    action_name="follow_aruco_move",
                    max_velocity=self.arm_max_velocity,
                    max_acceleration=self.arm_max_acceleration,
                    allowed_planning_time=self.allowed_planning_time,
                    position_tolerance=self.position_tolerance,
                    orientation_tolerance=self.orientation_tolerance,
                    allowed_start_tolerance=self.allowed_start_tolerance,
                    timeout_sec=self.motion_timeout_sec,
                )
        finally:
            with self._lock:
                self._worker_running = False
                has_fresh_target = (
                    self._latest_target is not None
                    and time.monotonic() - self._latest_target_at <= self.marker_pose_timeout_sec
                )
            if has_fresh_target and self._next_goal() is not None:
                with self._lock:
                    if not self._worker_running:
                        self._worker_running = True
                        threading.Thread(target=self._follow_latest_targets, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerFollower()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
