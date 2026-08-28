"""Node-neutral ROS sampling primitives shared by automatic and manual modes."""

from __future__ import annotations

from collections import deque
import math
import queue
import threading
import time

import numpy as np
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool
import tf2_ros

from myrobot_common.planning.motion_executor import MoveItMotion
from myrobot_common.task.abort_manager import AbortManager
from myrobot_common.utils.pose_tools import PoseTools

from .solver import TransformMatrix, robot_pose_for_calibration
from .vision import (
    CameraInfoState,
    VisionQualityGate,
    create_aruco_detector,
    estimate_marker_pose,
    make_observation,
    median_marker_corners,
)

try:
    import cv2
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover - runtime capability diagnostic
    cv2 = None
    CvBridge = None


_IMAGE_CHANNELS = {
    "bgr8": 3, "rgb8": 3, "mono8": 1, "bgra8": 4, "rgba8": 4,
    "8uc1": 1, "8uc3": 3, "8uc4": 4,
}


class _NoopGripper:
    def cancel_execution(self):
        return None


class SamplingRuntime:
    """Mixin for a ROS node with ``frames_config`` and ``sampling_config``."""

    def _initialize_sampling_runtime(self) -> None:
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._joint_lock = threading.Lock()
        self._joint_history = deque(maxlen=500)
        self._motion = self._abort = self._arm = self._controller_action_client = None
        self.vision_gate = VisionQualityGate(
            marker_distance_min_m=self.sampling_config.marker_distance_min_m,
            marker_distance_max_m=self.sampling_config.marker_distance_max_m,
            minimum_corner_margin_px=self.sampling_config.minimum_corner_margin_px,
            minimum_marker_side_px=self.sampling_config.minimum_marker_side_px,
            stable_frames=self.sampling_config.stable_frames,
            maximum_center_std_px=self.sampling_config.maximum_center_std_px,
            maximum_marker_depth_std_m=self.sampling_config.maximum_marker_depth_std_m,
            maximum_marker_angle_std_deg=self.sampling_config.maximum_marker_angle_std_deg,
            logger_warn=self.get_logger().warn,
        )
        self._bridge = CvBridge() if CvBridge is not None else None
        self._cv_ready = False
        self._aruco_queue = queue.Queue(maxsize=1)
        self._aruco_worker = threading.Thread(target=self._aruco_worker_loop, daemon=True)
        self._aruco_worker.start()

    @property
    def tf_buffer(self):
        return self._tf_buffer

    def tf_to_matrix(self, transform) -> TransformMatrix:
        value = transform.transform
        return TransformMatrix(
            R.from_quat((value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w)),
            (float(value.translation.x), float(value.translation.y), float(value.translation.z)),
        )

    def _setup_motion(self) -> None:
        from pymoveit2 import MoveIt2

        arm = MoveIt2(
            node=self,
            joint_names=list(self.motion_config.joint_names),
            base_link_name=self.frames_config.base_frame,
            end_effector_name=self.frames_config.ee_frame,
            group_name=self.motion_config.move_group_name,
            ignore_new_calls_while_executing=False,
            callback_group=self._callback_group,
            move_group_namespace=self.motion_config.move_group_ns_fairino,
        )
        arm.max_step_size = self.motion_config.max_step_size
        arm.max_velocity = self.motion_config.max_velocity
        arm.max_acceleration = self.motion_config.max_acceleration
        arm.allowed_planning_time = self.motion_config.allowed_planning_time
        arm.position_tolerance = self.motion_config.position_tolerance
        arm.orientation_tolerance = self.motion_config.orientation_tolerance
        arm.allowed_start_tolerance = self.motion_config.allowed_start_tolerance
        gripper = _NoopGripper()
        self._abort = AbortManager(self, arm=arm, gripper=gripper)
        self.create_subscription(Bool, "/manual_abort", self._abort.on_manual_abort, 10)
        self._motion = MoveItMotion(
            node=self, arm_clients={"fairino": arm}, default_client="fairino", gripper=gripper,
            pose_tools=PoseTools(self, base_frame=self.frames_config.base_frame), abort=self._abort,
            action_delay=self.motion_config.action_delay,
        )
        if not self._motion.set_planner(self.motion_config.planning_pipeline_id, self.motion_config.planner_id):
            raise RuntimeError("unsupported Fairino planner configuration")
        self._arm = arm
        self._controller_action_client = ActionClient(
            self, FollowJointTrajectory, "/robot_arm_controller/follow_joint_trajectory"
        )

    def _wait_for_moveit(self) -> bool:
        deadline = time.monotonic() + self.sampling_config.moveit_ready_timeout
        while time.monotonic() < deadline and not self._should_stop():
            plan = getattr(self._arm, "_plan_kinematic_path_service", None) or getattr(self._arm, "_plan_kinematic_path_client", None)
            execute = getattr(self._arm, "_execute_trajectory_action_client", None)
            if plan is not None and plan.service_is_ready() and execute is not None and execute.server_is_ready():
                return True
            time.sleep(self.sampling_config.moveit_ready_poll_interval)
        namespace = str(self.motion_config.move_group_ns_fairino or "/").strip("/")
        prefix = f"/{namespace}" if namespace else ""
        self.get_logger().error(
            "PRECHECK: MoveIt is not ready; expected "
            f"{prefix}/plan_kinematic_path and {prefix}/execute_trajectory."
        )
        return False

    def _wait_for_execution_controller(self) -> bool:
        """Confirm the root controller action has a server before moving W01."""
        deadline = time.monotonic() + self.sampling_config.moveit_ready_timeout
        while time.monotonic() < deadline and not self._should_stop():
            client = getattr(self, "_controller_action_client", None)
            if client is not None and client.server_is_ready():
                return True
            time.sleep(self.sampling_config.moveit_ready_poll_interval)
        self.get_logger().error(
            "PRECHECK: /robot_arm_controller/follow_joint_trajectory has no action server. "
            "Run: ros2 action info /robot_arm_controller/follow_joint_trajectory"
        )
        return False

    def _on_joint_state(self, message: JointState) -> None:
        values = dict(zip(message.name, message.position))
        try:
            positions = tuple(float(values[name]) for name in self.motion_config.joint_names)
        except (KeyError, TypeError, ValueError):
            return
        if all(math.isfinite(value) for value in positions):
            with self._joint_lock:
                self._joint_history.append((time.monotonic(), positions))

    def _clear_joint_history(self) -> None:
        with self._joint_lock:
            self._joint_history.clear()

    def _joint_state_ready(self) -> bool:
        with self._joint_lock:
            return bool(self._joint_history)

    def _joint_state_stream_ready(self) -> bool:
        """Require a recent, updating six-axis feedback stream rather than one stale message."""
        now = time.monotonic()
        with self._joint_lock:
            recent = tuple(item for item in self._joint_history if item[0] >= now - 1.0)
        return (
            len(recent) >= 2
            and recent[-1][0] - recent[0][0] >= 0.02
            and now - recent[-1][0] <= 0.25
        )

    def _wait_for_joint_state_stream(self) -> bool:
        deadline = time.monotonic() + self.sampling_config.moveit_ready_timeout
        while time.monotonic() < deadline and not self._should_stop():
            if self._joint_state_stream_ready():
                return True
            time.sleep(self.sampling_config.moveit_ready_poll_interval)
        self.get_logger().error(
            "PRECHECK: /joint_states is not a fresh updating j1..j6 stream. "
            "Run: ros2 topic hz /joint_states"
        )
        return False

    def _wait_for_joint_stationary(self):
        cfg = self.sampling_config
        deadline = time.monotonic() + cfg.joint_stationary_timeout_sec
        reason = "waiting for fresh joint position window"
        while time.monotonic() < deadline and not self._should_stop():
            now = time.monotonic()
            with self._joint_lock:
                window = tuple(item for item in self._joint_history if item[0] >= now - cfg.joint_stationary_window_sec)
            if len(window) >= 2 and window[-1][0] - window[0][0] >= cfg.joint_stationary_window_sec * 0.90:
                span = max(max(item[1][axis] for item in window) - min(item[1][axis] for item in window) for axis in range(6))
                if span <= cfg.joint_stationary_max_position_delta_rad:
                    return True, f"joint stationary span={span:.6f}rad"
                reason = f"joint span={span:.6f}rad exceeds {cfg.joint_stationary_max_position_delta_rad:.6f}rad"
            time.sleep(0.02)
        return False, "stop requested" if self._should_stop() else reason

    def _aruco_worker_loop(self) -> None:
        if cv2 is None or not hasattr(cv2, "aruco"):
            self.get_logger().error("OpenCV ArUco is unavailable")
            return
        try:
            dictionary, parameters = create_aruco_detector(self.frames_config.aruco_dictionary_id)
        except Exception as exc:
            self.get_logger().error(f"Cannot initialize ArUco detector: {exc}")
            return
        self._cv_ready = True
        while True:
            image, info, receipt_time = self._aruco_queue.get()
            if not self._collection_active.is_set():
                continue
            try:
                corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
                if ids is None:
                    self.vision_gate.record_failure("no markers detected", receipt_time=receipt_time)
                    continue
                matches = [index for index, value in enumerate(ids.flatten()) if int(value) == self.frames_config.marker_id]
                if not matches:
                    self.vision_gate.record_failure(f"marker id {self.frames_config.marker_id} not detected", receipt_time=receipt_time)
                    continue
                marker_corners = np.asarray(corners[matches[0]], dtype=float).reshape(4, 2)
                rvec, tvec = estimate_marker_pose(marker_corners, self.sampling_config.marker_size_m, info)
                self.vision_gate.record_success(make_observation(marker_corners, rvec, tvec, info, receipt_time=receipt_time))
            except Exception as exc:
                self.vision_gate.record_failure(f"aruco processing failed: {exc}", receipt_time=receipt_time)
                self.vision_gate.log_exception("worker", exc)

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.vision_gate.update_camera_info(CameraInfoState(
            width=int(message.width), height=int(message.height), p=tuple(float(value) for value in message.p),
            d=tuple(float(value) for value in message.d), frame_id=str(message.header.frame_id),
        ))

    def _enqueue_image(self, payload) -> None:
        try:
            self._aruco_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._aruco_queue.get_nowait()
                self._aruco_queue.put_nowait(payload)
            except (queue.Empty, queue.Full):
                pass

    def _on_image(self, message: Image) -> None:
        receipt_time = time.monotonic()
        if not self._cv_ready:
            return
        info = self.vision_gate.camera_info_snapshot()
        if not info.ready:
            self.vision_gate.record_failure("CameraInfo projection matrix is not ready", receipt_time=receipt_time)
            return
        if int(message.width) != info.width or int(message.height) != info.height or str(message.header.frame_id) != info.frame_id:
            self.vision_gate.record_failure("CameraInfo does not match image dimensions or frame", receipt_time=receipt_time)
            return
        try:
            image = self._image_to_bgr(message)
        except Exception as exc:
            self.vision_gate.record_failure(f"image conversion failed: {exc}", receipt_time=receipt_time)
            return
        self._enqueue_image((image, info, receipt_time))

    def _image_to_bgr(self, message: Image):
        if self._bridge is not None:
            return self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        encoding = message.encoding.lower()
        channels = _IMAGE_CHANNELS.get(encoding)
        if channels is None:
            raise ValueError(f"unsupported image encoding {message.encoding}")
        expected = int(message.width) * channels
        if int(message.step) < expected:
            raise ValueError("invalid image step")
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(int(message.height), int(message.step))
        packed = rows[:, :expected].reshape(int(message.height), int(message.width), channels)
        if encoding in ("bgr8", "8uc3"):
            return packed.copy()
        if encoding == "rgb8":
            return cv2.cvtColor(packed, cv2.COLOR_RGB2BGR)
        if encoding in ("mono8", "8uc1"):
            return cv2.cvtColor(packed[:, :, 0], cv2.COLOR_GRAY2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(packed, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(packed, cv2.COLOR_BGRA2BGR)

    def _latest_base_to_ee(self) -> TransformMatrix:
        return self.tf_to_matrix(self._tf_buffer.lookup_transform(
            self.frames_config.base_frame, self.frames_config.ee_frame, Time(),
        ))

    def _stable_sample(self):
        self.vision_gate.begin_window()
        deadline = time.monotonic() + self.sampling_config.stable_marker_timeout_sec
        reason = "waiting for marker"
        while time.monotonic() < deadline and not self._should_stop():
            frames, reason = self.vision_gate.stable_window()
            if frames is not None:
                try:
                    corners = median_marker_corners(frames)
                    rvec, tvec = estimate_marker_pose(corners, self.sampling_config.marker_size_m, self.vision_gate.camera_info_snapshot())
                    tracking = TransformMatrix(R.from_rotvec(rvec), tuple(float(value) for value in tvec))
                    time.sleep(self.sampling_config.stable_tf_settle_sec)
                    robot = robot_pose_for_calibration(self._latest_base_to_ee(), self.frames_config.calibration_type)
                    return robot, tracking, reason
                except Exception as exc:
                    return None, None, f"stable sample failed: {exc}"
            time.sleep(0.05)
        return None, None, "stop requested" if self._should_stop() else reason
