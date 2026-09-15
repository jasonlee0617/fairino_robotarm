#!/usr/bin/env python3
"""采集真实 D435 profile，供 Gazebo 相机模型使用。

使用前只启动真实相机，勿启动 Gazebo 或机械臂：

  ros2 launch realsense2_camera rs_launch.py \\
    enable_color:=true enable_depth:=true \\
    rgb_camera.color_profile:=640x480x60 \\
    depth_module.depth_profile:=640x480x60 \\
    align_depth.enable:=true

另开终端采集：

  ros2 run hand_eye_calibration capture_d435_profile.py --ros-args \\
    -p color_profile:=640x480x60 \\
    -p depth_profile:=640x480x60 \\
    -p sample_count:=60

默认原子写入 d435_color_640x480x60_depth_640x480x60.yaml。profile 名称
必须与驱动实际输出的分辨率和帧率一致。采集期间相机和约 0.5--1.0 m 处的
平整目标面必须静止，中心 16x16 ROI 必须持续有有效深度。
"""

from datetime import datetime
from pathlib import Path
import os
import re
import tempfile
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
import yaml


_PROFILE_RE = re.compile(r"^(\d+)x(\d+)x(\d+)$")


def _profile_spec(value, label):
    match = _PROFILE_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"{label} must be WIDTHxHEIGHTxFPS, got {value!r}")
    width, height, fps = (int(part) for part in match.groups())
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"{label} must contain positive dimensions and FPS")
    return width, height, fps


def _camera_info_dict(message):
    return {
        "width": int(message.width), "height": int(message.height),
        "distortion_model": str(message.distortion_model),
        "d": [float(value) for value in message.d],
        "k": [float(value) for value in message.k],
        "r": [float(value) for value in message.r],
        "p": [float(value) for value in message.p],
    }


def _consistent(messages):
    payloads = [_camera_info_dict(message) for message in messages]
    first = payloads[0]
    for payload in payloads[1:]:
        if payload["width"] != first["width"] or payload["height"] != first["height"]:
            return False, "CameraInfo dimensions changed during capture"
        for key in ("d", "k", "r", "p"):
            if not np.allclose(payload[key], first[key], rtol=0.0, atol=1.0e-9):
                return False, f"CameraInfo {key} changed during capture"
    return True, first


def _stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _validate_image_stream(label, frames, expected, tolerance_ratio):
    width, height, expected_fps = expected
    if len(frames) < 2:
        return False, f"{label} image stream has fewer than two frames"
    dimensions, stamps = zip(*frames)
    if any(value != (width, height) for value in dimensions):
        return False, f"{label} image dimensions do not match requested {width}x{height}"
    if any(stamp <= 0 for stamp in stamps):
        return False, f"{label} image has a zero timestamp"
    if any(later <= earlier for earlier, later in zip(stamps, stamps[1:])):
        return False, f"{label} image timestamps are duplicate or out of order"
    measured_fps = (len(stamps) - 1) * 1.0e9 / (stamps[-1] - stamps[0])
    if abs(measured_fps - expected_fps) > expected_fps * tolerance_ratio:
        return False, (
            f"{label} image rate {measured_fps:.2f}Hz does not match requested "
            f"{expected_fps}Hz within {tolerance_ratio * 100.0:.1f}%"
        )
    return True, measured_fps


def _temporal_depth_noise_stddev(patches):
    if len(patches) < 2:
        return None
    values = np.stack(patches, axis=0)
    valid = np.sum(np.isfinite(values), axis=0) >= 2
    if not np.any(valid):
        return None
    stddev = np.nanstd(values[:, valid], axis=0, ddof=1)
    stddev = stddev[np.isfinite(stddev)]
    return float(np.median(stddev)) if stddev.size else None


def _depth_patch_m(message):
    if message.encoding not in ("16UC1", "32FC1"):
        raise ValueError(f"unsupported depth encoding {message.encoding!r}")
    base_dtype = np.uint16 if message.encoding == "16UC1" else np.float32
    dtype = np.dtype(base_dtype).newbyteorder(">" if message.is_bigendian else "<")
    row_width = message.step // dtype.itemsize
    if row_width < message.width:
        raise ValueError("depth image step is smaller than its width")
    image = np.frombuffer(message.data, dtype=dtype).reshape(message.height, row_width)
    center_y, center_x = message.height // 2, message.width // 2
    # Only the 16x16 ROI is needed.  Converting an entire 640x480 frame to
    # float64 here can cause the Python subscriber itself to drop 60 Hz frames.
    patch = image[
        max(0, center_y - 8):center_y + 8,
        max(0, center_x - 8):center_x + 8,
    ].astype(np.float64, copy=True)
    if message.encoding == "16UC1":
        patch *= 0.001
    patch[~np.isfinite(patch) | (patch <= 0.0)] = np.nan
    return patch


class D435ProfileCapture(Node):
    def __init__(self):
        super().__init__("capture_d435_profile")
        self.declare_parameter("color_camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("depth_camera_info_topic", "/camera/camera/depth/camera_info")
        self.declare_parameter("color_image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_image_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("color_optical_frame", "camera_color_optical_frame")
        self.declare_parameter("depth_optical_frame", "camera_depth_optical_frame")
        self.declare_parameter("color_profile", "1280x720x30")
        self.declare_parameter("depth_profile", "848x480x30")
        self.declare_parameter("output_file", "")
        self.declare_parameter("sample_count", 60)
        self.declare_parameter("capture_noise", True)
        self.declare_parameter("frame_rate_tolerance_ratio", 0.10)
        self.declare_parameter("timeout_sec", 20.0)

        self._color_spec = _profile_spec(
            self.get_parameter("color_profile").value, "color_profile"
        )
        self._depth_spec = _profile_spec(
            self.get_parameter("depth_profile").value, "depth_profile"
        )
        self._count = int(self.get_parameter("sample_count").value)
        self._tolerance = float(self.get_parameter("frame_rate_tolerance_ratio").value)
        if self._count < 2:
            raise ValueError("sample_count must be at least 2")
        if not 0.0 < self._tolerance < 1.0:
            raise ValueError("frame_rate_tolerance_ratio must be in (0, 1)")

        self._started_monotonic = time.monotonic()
        self._color_info, self._depth_info = [], []
        self._color_frames, self._depth_frames, self._depth_patches = [], [], []
        self._depth_image_error = ""
        self._last_tf_warning_monotonic = 0.0
        self._tf = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf, self)
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            CameraInfo, self.get_parameter("color_camera_info_topic").value,
            self._on_color_info, qos,
        )
        self.create_subscription(
            CameraInfo, self.get_parameter("depth_camera_info_topic").value,
            self._on_depth_info, qos,
        )
        self.create_subscription(
            Image, self.get_parameter("color_image_topic").value,
            self._on_color_image, qos,
        )
        self.create_subscription(
            Image, self.get_parameter("depth_image_topic").value,
            self._on_depth_image, qos,
        )
        self.create_timer(0.1, self._finish_when_ready)

    def _on_color_info(self, message):
        if len(self._color_info) < self._count:
            self._color_info.append(message)

    def _on_depth_info(self, message):
        if len(self._depth_info) < self._count:
            self._depth_info.append(message)

    def _on_color_image(self, message):
        if len(self._color_frames) < self._count:
            self._color_frames.append(((int(message.width), int(message.height)), _stamp_ns(message)))

    def _on_depth_image(self, message):
        if len(self._depth_frames) >= self._count:
            return
        try:
            patch = _depth_patch_m(message)
        except ValueError as exc:
            self._depth_image_error = str(exc)
            return
        self._depth_frames.append(((int(message.width), int(message.height)), _stamp_ns(message)))
        if self.get_parameter("capture_noise").value:
            self._depth_patches.append(patch)

    def _abort(self, message):
        self.get_logger().error(message)
        rclpy.shutdown()

    def _finish_when_ready(self):
        enough_info = (
            len(self._color_info) >= self._count
            and len(self._depth_info) >= self._count
        )
        enough_frames = (
            len(self._color_frames) >= self._count
            and len(self._depth_frames) >= self._count
        )
        enough_noise = (
            not self.get_parameter("capture_noise").value
            or len(self._depth_patches) >= self._count
        )
        if enough_info and enough_frames and enough_noise:
            self._finish_capture()
            return
        if time.monotonic() - self._started_monotonic < float(
            self.get_parameter("timeout_sec").value
        ):
            return
        missing = []
        if not enough_info:
            missing.append("CameraInfo")
        if not enough_frames:
            missing.append("color/depth images")
        if not enough_noise:
            missing.append("valid depth ROI patches")
        detail = f"; depth image error: {self._depth_image_error}" if self._depth_image_error else ""
        self._abort(f"Timed out before receiving {' and '.join(missing)}.{detail}")

    def _finish_capture(self):
        color_ok, color = _consistent(self._color_info)
        depth_ok, depth = _consistent(self._depth_info)
        if not color_ok or not depth_ok:
            self._abort(color if not color_ok else depth)
            return
        if (color["width"], color["height"]) != self._color_spec[:2]:
            self._abort(
                f"Color CameraInfo is {color['width']}x{color['height']}, requested "
                f"{self._color_spec[0]}x{self._color_spec[1]}"
            )
            return
        if (depth["width"], depth["height"]) != self._depth_spec[:2]:
            self._abort(
                f"Depth CameraInfo is {depth['width']}x{depth['height']}, requested "
                f"{self._depth_spec[0]}x{self._depth_spec[1]}"
            )
            return
        color_rate_ok, color_rate = _validate_image_stream(
            "Color", self._color_frames, self._color_spec, self._tolerance
        )
        depth_rate_ok, depth_rate = _validate_image_stream(
            "Depth", self._depth_frames, self._depth_spec, self._tolerance
        )
        if not color_rate_ok or not depth_rate_ok:
            self._abort(color_rate if not color_rate_ok else depth_rate)
            return
        noise = None
        if self.get_parameter("capture_noise").value:
            noise = _temporal_depth_noise_stddev(self._depth_patches)
            if noise is None:
                self._abort("Depth ROI has insufficient valid temporal samples for noise capture")
                return
        try:
            extrinsic = self._tf.lookup_transform(
                str(self.get_parameter("depth_optical_frame").value),
                str(self.get_parameter("color_optical_frame").value),
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except Exception as exc:
            # RealSense publishes this as a static TF.  The streams can arrive
            # before /tf_static reaches a newly started listener, so keep the
            # validated sample window and retry until the ordinary capture
            # timeout rather than rejecting a valid profile immediately.
            elapsed = time.monotonic() - self._started_monotonic
            if elapsed < float(self.get_parameter("timeout_sec").value):
                if elapsed - self._last_tf_warning_monotonic >= 1.0:
                    self._last_tf_warning_monotonic = elapsed
                    self.get_logger().warn(
                        "Waiting for depth <- color TF "
                        f"({self.get_parameter('depth_optical_frame').value} <- "
                        f"{self.get_parameter('color_optical_frame').value}): {exc}"
                    )
                return
            self._abort(
                "Cannot capture depth <- color TF before timeout: "
                f"{exc}. Start rs_launch.py with publish_tf:=true, or pass "
                "-p depth_optical_frame:=<actual frame> and "
                "-p color_optical_frame:=<actual frame>."
            )
            return
        transform = extrinsic.transform
        color_profile = str(self.get_parameter("color_profile").value)
        depth_profile = str(self.get_parameter("depth_profile").value)
        output = str(self.get_parameter("output_file").value)
        if not output:
            output = str(
                Path(os.path.expandvars(
                    "$HOME/my-workspace/fairino_robotarm/src/camera_ws"
                ))
                / "realsense2_gz_description"
                / "config"
                / "d435_profiles"
                / f"d435_color_{color_profile}_depth_{depth_profile}.yaml"
            )
        profile = {
            "model": "D435",
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "color_profile": color_profile,
            "depth_profile": depth_profile,
            "color_camera_info": color,
            "depth_camera_info": depth,
            "depth_to_color": {
                "translation_m": [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ],
                "rotation_xyzw": [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ],
            },
            "empirical_depth_noise_stddev_m": noise,
        }
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(profile, stream, sort_keys=False)
        os.replace(temporary, destination)
        self.get_logger().info(
            f"D435 profile saved: {destination}; color_rate={color_rate:.2f}Hz; "
            f"depth_rate={depth_rate:.2f}Hz; empirical_depth_noise_stddev_m={noise}"
        )
        rclpy.shutdown()


def main():
    rclpy.init()
    node = D435ProfileCapture()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
