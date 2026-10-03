"""Task-side RGB-D frame matching and candidate resolution."""

from collections import Counter, deque
from dataclasses import dataclass
import math
import threading
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import numpy as np
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs  # noqa: F401

from visual_perception.msg import Yolov8Inference
from visual_perception_utils.depth_estimation import (
    robust_box_placement_from_depth,
    robust_center3d_from_obb_depth,
)


class PerceptionUnavailable(ValueError):
    """The LLM task cannot safely plan from the current camera data."""

    def __init__(self, error_code, detail=None):
        if detail is None:
            detail, error_code = error_code, "VISION_UNAVAILABLE"
        super().__init__(detail)
        self.error_code = str(error_code)


@dataclass(frozen=True)
class ResolvedCandidate:
    index: int
    class_name: str
    confidence: float
    center_uv: tuple[float, float]
    xyz: tuple[float, float, float]
    yaw: float
    frame_stamp_ns: int
    depth_inlier_ratio: float
    result_seq: int = 0
    depth_m: float = 0.0
    obb_area_px: float = 0.0
    placement_uv: tuple[float, float] | None = None
    container_xyz: tuple[float, float, float] | None = None

    def public(self):
        value = {
            "index": self.index, "class_name": self.class_name,
            "confidence": self.confidence, "center_uv": list(self.center_uv),
            "base_xyz": list(self.xyz), "yaw": self.yaw,
            "frame_stamp_ns": self.frame_stamp_ns,
            "result_seq": self.result_seq,
            "depth_inlier_ratio": self.depth_inlier_ratio,
            "depth_m": self.depth_m,
            "obb_area_px": self.obb_area_px,
            "selectable": True,
        }
        if self.placement_uv is not None:
            value["placement_uv"] = list(self.placement_uv)
        if self.container_xyz is not None:
            value["container_base_xyz"] = list(self.container_xyz)
        return value


def xy_shift(left, right):
    return math.hypot(left.xyz[0] - right.xyz[0], left.xyz[1] - right.xyz[1])


class RgbdPerception:
    """Resolve all LLM YOLO OBB candidates against synchronized RGB-D data."""

    def __init__(self, node, tf_buffer, *, base_frame, yolo_topic, depth_topic,
                 camera_info_topic, rgb_depth_tolerance_sec, detection_max_age_sec,
                 vision_wait_timeout_sec, tf_wait_timeout_sec=0.6,
                 callback_group=None):
        self.node, self.tf_buffer, self.base_frame = node, tf_buffer, base_frame
        self.yolo_topic, self.depth_topic = yolo_topic, depth_topic
        self.rgb_depth_tolerance_sec = rgb_depth_tolerance_sec
        self.detection_max_age_sec = detection_max_age_sec
        self.vision_wait_timeout_sec = vision_wait_timeout_sec
        self.tf_wait_timeout_sec = max(0.0, float(tf_wait_timeout_sec))
        self._lock = threading.RLock()
        self._bridge = CvBridge()
        self._depth_frames, self._yolo_frames = deque(maxlen=20), deque(maxlen=20)
        self._active_frame = self._camera_intrinsics = None
        self._result_seq = 0
        self._camera_frame = ""
        self._last_depth_received_at = None
        self._last_yolo_received_at = None
        self._last_selectable_count = 0
        self._last_unavailable_reason_counts = {}
        self.yolo_subscription = node.create_subscription(
            Yolov8Inference, yolo_topic, self._yolo_callback, 10,
            callback_group=callback_group)
        self.depth_subscription = node.create_subscription(
            Image, depth_topic, self._depth_callback, 10,
            callback_group=callback_group)
        self.camera_info_subscription = node.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback,
            qos_profile_sensor_data, callback_group=callback_group)

    @staticmethod
    def _stamp_ns(header) -> int:
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def _camera_info_callback(self, msg):
        if msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            return
        with self._lock:
            self._camera_intrinsics = {
                "fx": float(msg.k[0]), "fy": float(msg.k[4]),
                "cx": float(msg.k[2]), "cy": float(msg.k[5]),
            }
            self._camera_frame = str(msg.header.frame_id)

    def _depth_callback(self, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough").astype(np.float32)
            if msg.encoding in ("16UC1", "mono16"):
                depth /= 1000.0
            depth = np.nan_to_num(depth, nan=0.0, posinf=20.0, neginf=0.0)
            depth[depth > 20.0] = 20.0
        except Exception as exc:
            self.node.get_logger().warning(f"Cannot decode YOLO depth: {exc}")
            return
        with self._lock:
            self._last_depth_received_at = time.monotonic()
            self._depth_frames.append((msg.header, depth))
            self._activate_frame_locked()

    def _yolo_callback(self, msg):
        with self._lock:
            self._last_yolo_received_at = time.monotonic()
            self._yolo_frames.append(msg)
            self._activate_frame_locked()

    def _activate_frame_locked(self):
        if not self._yolo_frames or not self._depth_frames:
            return
        tolerance_ns = int(self.rgb_depth_tolerance_sec * 1e9)
        matches = (
            (self._stamp_ns(yolo.header),
             abs(self._stamp_ns(yolo.header) - self._stamp_ns(header)),
             yolo, header, depth)
            for yolo in self._yolo_frames
            for header, depth in self._depth_frames
        )
        try:
            stamp_ns, delta_ns, yolo, header, depth = max(
                (item for item in matches if item[1] <= tolerance_ns),
                key=lambda item: (item[0], -item[1]))
        except ValueError:
            return
        pair_key = (stamp_ns, self._stamp_ns(header))
        if self._active_frame is not None and self._active_frame["pair_key"] == pair_key:
            return
        self._result_seq = int(getattr(self, "_result_seq", 0)) + 1
        self._active_frame = {
            "yolo": yolo, "depth_header": header, "depth": depth,
            "stamp_ns": stamp_ns, "sync_delta_sec": delta_ns / 1e9,
            "pair_key": pair_key, "result_seq": self._result_seq,
            "received_monotonic": time.monotonic(),
        }

    def current_frame(self):
        with self._lock:
            frame = self._active_frame
        if (
            frame is None
            or time.monotonic() - frame["received_monotonic"] > self.detection_max_age_sec
        ):
            return None
        return frame

    @staticmethod
    def _detections(frame):
        for index, item in enumerate(frame["yolo"].yolov8_inference):
            try:
                points = np.asarray(item.coordinates, dtype=np.float32).reshape(4, 2)
            except (TypeError, ValueError):
                continue
            yield index, item, points, np.mean(points, axis=0)

    def metadata(self, frame=None):
        frame = self.current_frame() if frame is None else frame
        if frame is None:
            return []
        return [
            {"index": index, "class_name": str(item.class_name),
             "confidence": float(getattr(item, "confidence", 0.0)),
             "center_uv": [float(center[0]), float(center[1])]}
            for index, item, _points, center in self._detections(frame)
        ]

    @staticmethod
    def _box_occupied_polys(index, points, detections):
        return tuple(
            other_points
            for other_index, other, other_points, other_center in detections
            if other_index != index
            and str(other.class_name) != "box"
            and cv2.pointPolygonTest(
                points,
                (float(other_center[0]), float(other_center[1])),
                False,
            ) >= 0
        )

    def planning_metadata(self, frame, *, include_unavailable=False):
        result = []
        reason_counts = Counter()
        selectable_count = 0
        shape = getattr(frame.get("depth"), "shape", ())
        image_size = [int(shape[1]), int(shape[0])] if len(shape) >= 2 else None
        detections = list(self._detections(frame))
        for index, item, points, center in detections:
            area_px = float(abs(np.cross(points[1] - points[0], points[3] - points[0])))
            occupied_polys = ()
            if str(item.class_name) == "box":
                occupied_polys = self._box_occupied_polys(
                    index, points, detections
                )
            if occupied_polys:
                resolved, unavailable_reason = self._resolve_detection_detailed(
                    index, item, points, center, frame,
                    occupied_polys=occupied_polys,
                )
            else:
                resolved, unavailable_reason = self._resolve_detection_detailed(
                    index, item, points, center, frame
                )
            if resolved is not None:
                selectable_count += 1
                public = resolved.public()
                if image_size is not None:
                    public["image_size"] = image_size
                result.append(public)
            elif include_unavailable:
                reason_counts[unavailable_reason] += 1
                raw = {
                    "index": index,
                    "class_name": str(item.class_name),
                    "confidence": float(getattr(item, "confidence", 0.0)),
                    "center_uv": [float(center[0]), float(center[1])],
                    "frame_stamp_ns": int(frame["stamp_ns"]),
                    "result_seq": int(frame.get("result_seq", 0)),
                    "selectable": False,
                    "unavailable_reason": unavailable_reason,
                    "obb_area_px": area_px,
                }
                if image_size is not None:
                    raw["image_size"] = image_size
                result.append(raw)
        self._last_selectable_count = selectable_count
        self._last_unavailable_reason_counts = dict(reason_counts)
        return result

    def wait_for_planning_metadata(self, *, after_result_seq=None):
        deadline = time.monotonic() + max(0.0, self.vision_wait_timeout_sec)
        frame_seen = False
        best_metadata = []
        raw_box_seen = False
        last_pair_key = None
        while True:
            frame = self.current_frame()
            if frame is not None:
                if (
                    after_result_seq is not None
                    and int(frame.get("result_seq", 0)) <= int(after_result_seq)
                ):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                    continue
                pair_key = (
                    frame.get("pair_key", frame.get("stamp_ns"))
                    if isinstance(frame, dict) else id(frame)
                )
                if pair_key == last_pair_key:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                    continue
                last_pair_key = pair_key
                frame_seen = True
                raw_classes = {
                    str(item.class_name) for _index, item, _points, _center
                    in self._detections(frame)
                }
                raw_box_seen = raw_box_seen or "box" in raw_classes
                metadata = self.planning_metadata(frame, include_unavailable=True)
                if metadata:
                    best_metadata = metadata
                    resolved_classes = {
                        item["class_name"] for item in metadata if item.get("selectable", True)
                    }
                    if not raw_box_seen or "box" in resolved_classes:
                        return metadata
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if best_metadata:
            if raw_box_seen and not any(
                item["class_name"] == "box" and item.get("selectable", True)
                for item in best_metadata
            ):
                reasons = Counter(
                    item.get("unavailable_reason", "VISION_UNAVAILABLE")
                    for item in best_metadata
                    if item.get("class_name") == "box"
                    and not item.get("selectable", True)
                )
                self.node.get_logger().warning(
                    "YOLO detected box in 2D, but no box candidate passed depth/TF "
                    f"validation before the vision timeout: {dict(reasons)}"
                )
            return best_metadata
        if frame_seen:
            raise PerceptionUnavailable(
                "VISION_NO_SELECTABLE_TARGET",
                "No detected target has usable 3D data; adjust the view and retry.",
            )
        code, detail = self.vision_unavailable_reason()
        raise PerceptionUnavailable(code, detail)

    def _publisher_count(self, topic):
        try:
            return int(self.node.count_publishers(topic))
        except Exception:
            return 0

    def vision_unavailable_reason(self):
        missing = [topic for topic in (self.yolo_topic, self.depth_topic)
                   if self._publisher_count(topic) == 0]
        if missing:
            code = (
                "VISION_DEPTH_STALE"
                if self.depth_topic in missing else "VISION_SYNC_STALE"
            )
            return code, (
                "Vision input unavailable: no publisher on "
                f"{', '.join(missing)}. Start "
                "`ros2 launch myrobot_simulation llm_robot_control_sim.launch.py` "
                "and wait for the first YOLO inference."
            )
        if self._camera_intrinsics is None:
            return (
                "VISION_CAMERA_INFO_MISSING",
                "Vision input unavailable: camera_info has not arrived yet.",
            )
        now = time.monotonic()
        depth_received = getattr(self, "_last_depth_received_at", None)
        if depth_received is None or now - depth_received > self.detection_max_age_sec:
            return (
                "VISION_DEPTH_STALE",
                "Depth results are stale; check the aligned-depth stream.",
            )
        return "VISION_SYNC_STALE", (
            "YOLO/depth publishers are connected but no fresh synchronized frame arrived; "
            "wait for the first inference or check the llm_visual_perception warning log."
        )

    def _transform_point(self, xyz, header, frame=None):
        point = PointStamped()
        point.header = header
        if not point.header.frame_id:
            point.header.frame_id = self._camera_frame
        point.point.x, point.point.y, point.point.z = (float(value) for value in xyz)
        transform = None if frame is None else frame.get("_base_transform")
        if transform is None:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                point.header.frame_id,
                Time.from_msg(point.header.stamp),
                timeout=Duration(seconds=self.tf_wait_timeout_sec),
            )
            if frame is not None:
                frame["_base_transform"] = transform
        return tf2_geometry_msgs.do_transform_point(point, transform)

    def _resolve_detection_detailed(
        self, index, item, points, center_uv, frame, *, occupied_polys=()
    ):
        if self._camera_intrinsics is None:
            return None, "VISION_CAMERA_INFO_MISSING"
        class_name = str(item.class_name)
        placement_uv = None
        if class_name == "box":
            center3d, quality, placement_uv, free_cells = (
                robust_box_placement_from_depth(
                    points,
                    frame["depth"],
                    self._camera_intrinsics,
                    occupied_polys=occupied_polys,
                )
            )
            if free_cells == 0:
                return None, "VISION_BOX_NO_FREE_REGION"
        else:
            center3d, quality = robust_center3d_from_obb_depth(
                poly_2d=points, depth=frame["depth"],
                camera_intrinsics=self._camera_intrinsics,
                stride=1, min_points=20, max_points=5000,
                depth_max_range=10.0, depth_inlier_m=0.08, depth_mad_scale=3.0,
                min_depth_inlier_ratio=0.6)
        if center3d is None:
            return None, "VISION_DEPTH_INVALID"
        edges = np.roll(points, -1, axis=0) - points
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm <= 1e-6:
            return None, "VISION_OBB_AXIS_INVALID"
        axis_uv = edge / edge_norm * min(20.0, edge_norm / 2.0)
        z = float(center3d[2])
        intrinsics = self._camera_intrinsics
        orientation_center3d = center3d
        if class_name == "box":
            orientation_center3d = (
                (center_uv[0] - intrinsics["cx"]) * z / intrinsics["fx"],
                (center_uv[1] - intrinsics["cy"]) * z / intrinsics["fy"],
                z,
            )
        axis3d = (
            (center_uv[0] + axis_uv[0] - intrinsics["cx"]) * z / intrinsics["fx"],
            (center_uv[1] + axis_uv[1] - intrinsics["cy"]) * z / intrinsics["fy"], z)
        try:
            center_base = self._transform_point(
                center3d, frame["yolo"].header, frame
            )
            orientation_center_base = center_base
            if class_name == "box":
                orientation_center_base = self._transform_point(
                    orientation_center3d, frame["yolo"].header, frame
                )
            axis_base = self._transform_point(
                axis3d, frame["yolo"].header, frame
            )
        except Exception as exc:
            self.node.get_logger().warning(f"camera-to-base TF unavailable: {exc}")
            return None, "VISION_TF_UNAVAILABLE"
        direction = (
            axis_base.point.x - orientation_center_base.point.x,
            axis_base.point.y - orientation_center_base.point.y,
        )
        if math.hypot(*direction) <= 1e-6:
            return None, "VISION_OBB_AXIS_INVALID"
        yaw = math.atan2(direction[1], direction[0])
        yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
        return ResolvedCandidate(
            index=int(index), class_name=class_name,
            confidence=float(getattr(item, "confidence", 0.0)),
            center_uv=(float(center_uv[0]), float(center_uv[1])),
            xyz=(float(center_base.point.x), float(center_base.point.y),
                 float(center_base.point.z)), yaw=float(yaw),
            frame_stamp_ns=int(frame["stamp_ns"]), depth_inlier_ratio=float(quality),
            result_seq=int(frame.get("result_seq", 0)), depth_m=float(center3d[2]),
            obb_area_px=float(abs(np.cross(points[1] - points[0], points[3] - points[0]))),
            placement_uv=(
                None if placement_uv is None
                else (float(placement_uv[0]), float(placement_uv[1]))
            ),
            container_xyz=(
                None if class_name != "box"
                else (
                    float(orientation_center_base.point.x),
                    float(orientation_center_base.point.y),
                    float(orientation_center_base.point.z),
                )
            ),
        ), ""

    def resolve_candidate_detailed(self, index, frame=None):
        frame = self.current_frame() if frame is None else frame
        if frame is None:
            return None, "VISION_SYNC_STALE"
        detections = list(self._detections(frame))
        for item_index, item, points, center in detections:
            if item_index == int(index):
                occupied = (
                    self._box_occupied_polys(item_index, points, detections)
                    if str(item.class_name) == "box" else ()
                )
                return self._resolve_detection_detailed(
                    item_index,
                    item,
                    points,
                    center,
                    frame,
                    occupied_polys=occupied,
                )
        return None, "TARGET_NOT_FOUND"

    def fresh_match(self, old):
        frame = self.current_frame()
        if frame is None:
            return None
        matches = [
            resolved
            for index, item, points, center in self._detections(frame)
            if str(item.class_name) == old.class_name
            for resolved in [self._resolve_detection_detailed(
                index, item, points, center, frame
            )[0]]
            if resolved is not None
        ]
        return min(matches, key=lambda candidate: xy_shift(old, candidate)) if matches else None

    def diagnostics(self, frame=None):
        frame = self.current_frame() if frame is None else frame
        with self._lock:
            now = time.monotonic()
            return {
                "fresh_detection": frame is not None,
                "candidate_count": len(self.metadata(frame)),
                "yolo_buffer_count": len(self._yolo_frames),
                "depth_buffer_count": len(self._depth_frames),
                "yolo_publisher_count": self._publisher_count(self.yolo_topic),
                "depth_publisher_count": self._publisher_count(self.depth_topic),
                "rgb_depth_delta_sec": None if frame is None else frame["sync_delta_sec"],
                "camera_info_ready": self._camera_intrinsics is not None,
                "result_seq": int(getattr(self, "_result_seq", 0)),
                "yolo_age_sec": None
                if getattr(self, "_last_yolo_received_at", None) is None
                else round(now - self._last_yolo_received_at, 3),
                "depth_age_sec": None
                if getattr(self, "_last_depth_received_at", None) is None
                else round(now - self._last_depth_received_at, 3),
                "selectable_candidate_count": getattr(
                    self, "_last_selectable_count", 0
                ),
                "unavailable_reason_counts": dict(
                    getattr(self, "_last_unavailable_reason_counts", {})
                ),
            }
