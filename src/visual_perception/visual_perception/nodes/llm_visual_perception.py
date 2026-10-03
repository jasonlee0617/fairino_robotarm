#!/usr/bin/env python3

from collections import deque
import copy
import gc
import math
import signal
import threading
import time
import cv2
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_srvs.srv import Trigger
import numpy as np
import tf2_geometry_msgs  # noqa: F401  Registers PointStamped transforms.
import tf2_ros

from visual_perception.msg import InferenceResult
from visual_perception.msg import Yolov8Inference
from visual_perception_utils.model_utils import (
    assign_obb_confidence,
    require_four_class_obb_model,
    resolve_yolo_model_path,
)
from visual_perception_utils.depth_estimation import (
    robust_box_placement_from_depth,
    robust_center3d_from_obb_depth,
)
from visual_perception_utils.visualization import (
    draw_detection_center,
    draw_detection_diagnostics,
    draw_obb_major_axis,
)


class LlmYoloPerceptionNode(Node):

    def __init__(self):
        super().__init__('llm_visual_perception')
        from ultralytics import YOLO

        self._declare_parameters()
        self._yolo_class = YOLO
        self._read_parameters()
        self._model_lock = threading.RLock()
        self._control_callback_group = ReentrantCallbackGroup()
        self._camera_callback_group = MutuallyExclusiveCallbackGroup()
        self._inference_callback_group = MutuallyExclusiveCallbackGroup()
        self.model = None
        self._visual_available = True
        self._visual_last_error = ""
        self._shutting_down = False
        self.class_names = None
        self.inference_device = self._select_inference_device()
        if not self._load_model():
            raise RuntimeError("LLM YOLO model failed to load")
        self._setup_ros_interfaces()
        self._reset_runtime_state()
        self._start_processing()
        self.get_logger().info(
            f"YOLO OBB ready: rgb={self.rgb_topic}, depth={self.depth_topic}, "
            f"slop={self.sync_slop:.3f}s, device={self.inference_device}"
        )

    def _declare_parameters(self):
        defaults = {
            "model_path": "yolo-obb-1280.pt",
            "imgsz": 1024,
            "conf": 0.50,
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": (
                "/camera/camera/aligned_depth_to_color/camera_info"
            ),
            "base_frame": "base_link",
            "sync_slop": 0.02,
            "sync_watchdog_sec": 3.0,
            "expected_camera_rate_hz": 30.0,
            "require_cuda_for_visual_perception": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        self.model_path = resolve_yolo_model_path(str(self.get_parameter('model_path').value))
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.sync_watchdog_sec = float(self.get_parameter('sync_watchdog_sec').value)
        self.expected_camera_rate_hz = float(
            self.get_parameter('expected_camera_rate_hz').value
        )
        self.require_cuda = bool(
            self.get_parameter('require_cuda_for_visual_perception').value
        )

    def _setup_ros_interfaces(self):
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._camera_intrinsics = None
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/yolo/detected_result", 1)
        visual_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.img_pub = self.create_publisher(Image, "/camera/detected_result", visual_qos)
        self.depth_pub = self.create_publisher(Image, "/yolo/detected_result/depth", 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.class_colors = {
            "box": (255, 0, 0),
            "elongated_object": (0, 255, 0),
            "cube": (0, 255, 255),
            "stone": (255, 0, 255),
        }

    def _reset_runtime_state(self):
        self.rgb_sub = None
        self.depth_sub = None
        self.sync = None
        self._frame_count = 0
        now = time.monotonic()
        self._rgb_times = deque(maxlen=120)
        self._depth_times = deque(maxlen=120)
        self._sync_times = deque(maxlen=120)
        self._result_times = deque(maxlen=120)
        self._last_rgb_activity = now
        self._last_depth_activity = now
        self._last_sync_activity = now
        self._last_result_activity = now
        self._last_sync_delta_sec = None
        self._latest_pair = None
        self._latest_pair_lock = threading.Lock()
        self._inference_active = threading.Event()
        self._last_sync_warning_at = -1e9
        self._rtf_wall_at = now
        self._rtf_ros_ns = self.get_clock().now().nanoseconds
        self._last_rtf = 1.0

    def _start_processing(self):
        self._start_sync()
        self.inference_timer = self.create_timer(
            0.01,
            self._process_latest_pair,
            callback_group=self._inference_callback_group,
        )
        self.sync_watchdog = self.create_timer(
            1.0,
            self._check_sync,
            callback_group=self._control_callback_group,
        )
        self.create_service(
            Trigger,
            '/llm_visual_perception/status',
            self._status,
            callback_group=self._control_callback_group,
        )

    def _select_inference_device(self):
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            if self.require_cuda:
                raise RuntimeError(f"CUDA readiness check failed: {exc}") from exc
            cuda_available = False
        if self.require_cuda and not cuda_available:
            raise RuntimeError(
                "CUDA is required for LLM visual perception; repair NVIDIA device access first"
            )
        if not cuda_available:
            self.get_logger().warning("LLM visual perception is explicitly using CPU")
        return "cuda:0" if cuda_available else "cpu"

    def _load_model(self):
        with self._model_lock:
            if self._shutting_down:
                return False
            if self.model is not None:
                return True
            try:
                model = self._yolo_class(self.model_path)
                class_names = require_four_class_obb_model(model.names)
            except Exception as exc:
                self.get_logger().error(f"Failed to load LLM YOLO model: {exc}")
                self._visual_available = False
                self._visual_last_error = str(exc)
                return False
            self.model = model
            self.class_names = class_names
            self._visual_available = True
            self._visual_last_error = ""
        self.get_logger().debug(
            f"Four-class YOLO-OBB contract accepted: {self.class_names}"
        )
        return True

    def _status(self, _request, response):
        if not self._visual_available:
            response.success = False
            response.message = str(self._visual_last_error or "VISION_MODEL_UNAVAILABLE")
            return response
        now = time.monotonic()
        checks = (
            (self.count_publishers(self.rgb_topic) <= 0, "VISION_NO_RGB_PUBLISHER"),
            (self.count_publishers(self.depth_topic) <= 0, "VISION_NO_DEPTH_PUBLISHER"),
            (
                now - self._last_rgb_activity > self.sync_watchdog_sec,
                "VISION_RGB_STALE",
            ),
            (
                now - self._last_depth_activity > self.sync_watchdog_sec,
                "VISION_DEPTH_STALE",
            ),
            (
                now - self._last_sync_activity > self.sync_watchdog_sec,
                "VISION_SYNC_STALE",
            ),
            (now - self._last_result_activity > self.sync_watchdog_sec,
             "VISION_RESULT_STALE"),
        )
        failure = next((code for failed, code in checks if failed), "")
        response.success = not failure
        response.message = failure
        return response

    def _unload_model(self):
        with self._model_lock:
            model, self.model = self.model, None
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            self.get_logger().warning(f"LLM YOLO CUDA cache cleanup skipped: {exc}")

    @staticmethod
    def _is_cuda_oom(exc):
        message = str(exc).lower()
        return "cuda" in message and "out of memory" in message

    def _start_sync(self):
        self.rgb_sub = Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
            callback_group=self._camera_callback_group,
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
            callback_group=self._camera_callback_group,
        )
        self.rgb_sub.registerCallback(self._note_rgb)
        self.depth_sub.registerCallback(self._note_depth)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=3,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.camera_callback)
        self._last_sync_activity = time.monotonic()

    @staticmethod
    def _stamp_ns(msg):
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _note_rgb(self, _msg):
        self._last_rgb_activity = time.monotonic()
        self._rgb_times.append(self._last_rgb_activity)

    def _note_depth(self, _msg):
        self._last_depth_activity = time.monotonic()
        self._depth_times.append(self._last_depth_activity)

    @staticmethod
    def _rate_hz(times):
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def _check_sync(self):
        now = time.monotonic()
        ros_ns = self.get_clock().now().nanoseconds
        wall_delta = now - self._rtf_wall_at
        if wall_delta > 0.0:
            self._last_rtf = max(0.0, (ros_ns - self._rtf_ros_ns) / 1e9 / wall_delta)
        self._rtf_wall_at, self._rtf_ros_ns = now, ros_ns
        if now - self._last_sync_activity <= self.sync_watchdog_sec:
            return
        if now - self._last_sync_warning_at >= 30.0:
            self._last_sync_warning_at = now
            delta_ms = (
                None
                if self._last_sync_delta_sec is None
                else round(self._last_sync_delta_sec * 1000.0, 1)
            )
            expected_rate = max(
                self.expected_camera_rate_hz * self._last_rtf, 1e-6
            )
            self.get_logger().warning(
                "VISION_SYNC "
                f"rgb_publishers={self.count_publishers(self.rgb_topic)} "
                f"depth_publishers={self.count_publishers(self.depth_topic)} "
                f"rgb_age_ms={(now - self._last_rgb_activity) * 1000.0:.1f} "
                f"depth_age_ms={(now - self._last_depth_activity) * 1000.0:.1f} "
                f"sync_age_ms={(now - self._last_sync_activity) * 1000.0:.1f} "
                f"last_stamp_delta_ms={delta_ms} "
                f"rgb_hz={self._rate_hz(self._rgb_times):.1f} "
                f"depth_hz={self._rate_hz(self._depth_times):.1f} "
                f"sync_hz={self._rate_hz(self._sync_times):.1f} "
                f"result_hz={self._rate_hz(self._result_times):.1f}"
                f" rtf={self._last_rtf:.2f} "
                f"expected_hz={self.expected_camera_rate_hz:.1f} "
                f"rgb_delivery={self._rate_hz(self._rgb_times) / expected_rate:.2f} "
                f"depth_delivery={self._rate_hz(self._depth_times) / expected_rate:.2f}"
            )

    def _camera_info_callback(self, msg):
        if msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self._camera_intrinsics = {
                "fx": float(msg.k[0]), "fy": float(msg.k[4]),
                "cx": float(msg.k[2]), "cy": float(msg.k[5]),
            }

    def _transform_point(self, xyz, header):
        point = PointStamped()
        point.header = header
        point.point.x, point.point.y, point.point.z = (float(value) for value in xyz)
        return self.tf_buffer.transform(
            point, self.base_frame, timeout=Duration(seconds=0.2)
        )

    def _diagnostic_lines(self, class_name, corners, center_uv, header, depth):
        intrinsics = self._camera_intrinsics
        if intrinsics is None:
            return ["3D unavailable: camera_info"]
        if class_name == "box":
            center3d, quality, center_uv, _free_cells = (
                robust_box_placement_from_depth(corners, depth, intrinsics)
            )
        else:
            center3d, quality = robust_center3d_from_obb_depth(
                poly_2d=corners,
                depth=depth,
                camera_intrinsics=intrinsics,
                stride=1,
                min_points=20,
                max_points=5000,
                depth_max_range=10.0,
                depth_inlier_m=0.08,
                depth_mad_scale=3.0,
                min_depth_inlier_ratio=0.6,
            )
        if center3d is None:
            return ["3D unavailable: depth"]
        edges = np.roll(corners, -1, axis=0) - corners
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm <= 1e-6:
            return ["3D unavailable: OBB axis"]
        axis_uv = edge / edge_norm * min(20.0, edge_norm / 2.0)
        z = float(center3d[2])
        axis3d = (
            (center_uv[0] + axis_uv[0] - intrinsics["cx"]) * z / intrinsics["fx"],
            (center_uv[1] + axis_uv[1] - intrinsics["cy"]) * z / intrinsics["fy"],
            z,
        )
        try:
            center_base = self._transform_point(center3d, header)
            axis_base = self._transform_point(axis3d, header)
        except Exception:
            return ["3D unavailable: TF"]
        direction = (
            axis_base.point.x - center_base.point.x,
            axis_base.point.y - center_base.point.y,
        )
        if math.hypot(*direction) <= 1e-6:
            return ["3D unavailable: axis TF"]
        yaw = (math.atan2(direction[1], direction[0]) + math.pi / 2.0) % math.pi - math.pi / 2.0
        return [
            f"base: {center_base.point.x:.3f}, {center_base.point.y:.3f}, "
            f"{center_base.point.z:.3f} m",
            f"yaw: {math.degrees(yaw):.1f} deg  depthQ: {quality:.2f}",
        ]

    def camera_callback(self, rgb_msg, depth_msg):
        if self._shutting_down:
            return
        now = time.monotonic()
        self._last_sync_activity = now
        self._sync_times.append(now)
        self._last_sync_delta_sec = abs(
            self._stamp_ns(rgb_msg) - self._stamp_ns(depth_msg)
        ) / 1e9
        with self._latest_pair_lock:
            self._latest_pair = (rgb_msg, depth_msg)

    def _process_latest_pair(self):
        if self._shutting_down or self._inference_active.is_set():
            return
        with self._latest_pair_lock:
            pair, self._latest_pair = self._latest_pair, None
        if pair is None:
            return
        self._inference_active.set()
        try:
            self._camera_callback_impl(*pair)
        finally:
            self._inference_active.clear()

    def _decode_depth(self, depth_msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            ).astype(np.float32)
            if depth_msg.encoding in ("16UC1", "mono16"):
                depth /= 1000.0
            return depth
        except Exception:
            return None

    def _render_results(self, results, image, header, depth):
        inference = Yolov8Inference()
        inference.header = header
        annotated = image.copy()
        for result in results:
            if result.obb is None:
                continue
            for box in result.obb:
                corners = (
                    box.xyxyxyxy[0].to("cpu").detach().numpy().copy().reshape(4, 2)
                )
                class_name = self.class_names[int(box.cls.item())]
                inference_result = InferenceResult()
                inference_result.class_name = class_name
                assign_obb_confidence(inference_result, box)
                inference_result.coordinates = copy.copy(corners.reshape(-1).tolist())
                inference.yolov8_inference.append(inference_result)

                center_pixel = tuple(np.clip(
                    np.mean(corners, axis=0),
                    [0, 0],
                    [image.shape[1] - 1, image.shape[0] - 1],
                ).astype(int))
                color = self.class_colors[class_name]
                cv2.polylines(
                    annotated,
                    [corners.reshape(-1, 1, 2).astype(np.int32)],
                    True,
                    color,
                    2,
                )
                draw_detection_center(annotated, center_pixel)
                draw_obb_major_axis(annotated, corners, color)
                lines = [
                    f"{class_name} conf={float(box.conf.item()):.2f}",
                    f"uv: {center_pixel[0]}, {center_pixel[1]}",
                ]
                if depth is None:
                    lines.append("3D unavailable: depth")
                else:
                    lines.extend(
                        self._diagnostic_lines(
                            class_name, corners, center_pixel, header, depth
                        )
                    )
                draw_detection_diagnostics(annotated, center_pixel, lines, color)
        return inference, annotated

    def _publish_results(self, inference, annotated, rgb_msg, depth_msg):
        if self._shutting_down or not rclpy.ok():
            return False
        try:
            self.yolov8_pub.publish(inference)
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            img_msg.header = rgb_msg.header
            self.img_pub.publish(img_msg)
            self.depth_pub.publish(depth_msg)
            return True
        except Exception:
            if self._shutting_down or not rclpy.ok():
                return False
            raise

    def _camera_callback_impl(self, rgb_msg, depth_msg):
        self._last_sync_activity = time.monotonic()
        try:
            with self._model_lock:
                if self.model is None or not self._visual_available:
                    return
                img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                inference_started_at = time.monotonic()
                results = self.model(
                    img,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    verbose=False,
                    device=self.inference_device,
                )
                inference_ms = (time.monotonic() - inference_started_at) * 1000.0
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            self._unload_model()
            self._visual_available = False
            self._visual_last_error = f"CUDA OOM: {exc}"
            self.get_logger().error(
                f"LLM YOLO inference disabled after CUDA OOM: {exc}"
            )
            return
        self._visual_available = True
        self._visual_last_error = ""
        depth = self._decode_depth(depth_msg)
        inference, annotated_frame = self._render_results(
            results, img, rgb_msg.header, depth
        )
        detection_count = len(inference.yolov8_inference)
        if not self._publish_results(inference, annotated_frame, rgb_msg, depth_msg):
            return
        self._last_result_activity = time.monotonic()
        self._result_times.append(self._last_result_activity)
        self._frame_count += 1
        if self._frame_count == 1 or self._frame_count % 30 == 0:
            self.get_logger().debug(
                f"YOLO inference frame={self._frame_count} device={self.inference_device} "
                f"inference_ms={inference_ms:.1f} detections={detection_count}"
            )

    def _begin_shutdown(self):
        self._shutting_down = True
        self.inference_timer.cancel()
        self.sync_watchdog.cancel()
        with self._latest_pair_lock:
            self._latest_pair = None


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = LlmYoloPerceptionNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node._begin_shutdown()
            executor.shutdown(timeout_sec=2.0)
            node._unload_model()
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
