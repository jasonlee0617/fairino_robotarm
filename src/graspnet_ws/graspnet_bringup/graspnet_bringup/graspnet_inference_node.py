#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GraspNet 推理节点 (GraspnetInferenceNode)
#
# 功能：
#   - 订阅 RGB 图像、深度图像、相机内参
#   - 使用 GraspNet-baseline 模型生成抓取候选
#   - 支持自动/手动/服务触发推理
#   - 支持 Open3D 可视化确认（用户选择最优抓取或全部候选）
#   - 发布抓取姿态 PoseArray、得分 Float32MultiArray、元数据 Float32MultiArray
#   - 提供预览最佳抓取姿态的话题
#
# 依赖：
#   - ROS 2 (rclpy)
#   - cv_bridge（ROS 图像转 OpenCV）
#   - torch（PyTorch）
#   - open3d（可选，用于可视化）
#   - graspnet-baseline 代码库
# ---------------------------------------------------------------------------
import json
import signal
import threading
import time
from typing import List, Optional, Tuple
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension, String
from std_srvs.srv import Trigger
from graspnet_bringup import inference_processing
from graspnet_bringup import inference_runtime

# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════


def _graspnet_source_path(*parts: str) -> str:
    return inference_runtime.graspnet_source_path(*parts)


def _load_graspnet_modules(baseline_dir: str):
    return inference_runtime.load_graspnet_modules(baseline_dir)


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> Tuple[float, float, float, float]:
    return inference_processing.rotmat_to_quat_xyzw(rot)


def _vector(values, count: int, fallback: float) -> np.ndarray:
    return inference_processing.vector(values, count, fallback)


def _filter_grasp_group_by_width(grasp_group, min_width_m: float, max_width_m: float):
    return inference_processing.filter_grasp_group_by_width(
        grasp_group, min_width_m, max_width_m
    )


def _valid_camera_info(info: CameraInfo) -> bool:
    return inference_processing.valid_camera_info(info)


def _workspace_mask(
    points: np.ndarray,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
) -> np.ndarray:
    return inference_processing.workspace_mask(points, x_min_m, x_max_m, y_min_m, y_max_m)


def _support_plane_signed_distances(
    points: np.ndarray,
    plane_model: np.ndarray,
    base_from_camera_rotation: np.ndarray,
    max_tilt_deg: float,
) -> Optional[np.ndarray]:
    return inference_processing.support_plane_signed_distances(
        points, plane_model, base_from_camera_rotation, max_tilt_deg
    )


def _object_height_mask(
    signed_distances_m: np.ndarray,
    min_height_m: float,
    max_height_m: float,
) -> np.ndarray:
    return inference_processing.object_height_mask(
        signed_distances_m, min_height_m, max_height_m
    )


def _filter_collision_free_grasps(
    detector_class,
    scene_points: np.ndarray,
    grasp_group,
    voxel_size_m: float,
    approach_distance_m: float,
    collision_threshold: float,
):
    return inference_processing.filter_collision_free_grasps(
        detector_class,
        scene_points,
        grasp_group,
        voxel_size_m,
        approach_distance_m,
        collision_threshold,
    )


def _graspgroup_to_pose_metadata(grasp_group) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    return inference_processing.graspgroup_to_pose_metadata(grasp_group)


# ═══════════════════════════════════════════════════════════
#  GraspNet 推理节点
# ═══════════════════════════════════════════════════════════

class GraspnetInferenceNode(Node):
    """
    ROS2 节点：使用 GraspNet-baseline 进行 6-DOF 抓取姿态推理。

    订阅：
        - 彩色图像 (RGB)
        - 对齐后的深度图像
        - 相机内参 (CameraInfo)
    锁存一次 CameraInfo，并通过 RGB/深度近似时间同步器触发推理，
    或通过 /grasp/compute 服务手动触发。

    发布：
        - 抓取姿态 PoseArray（/grasp/poses）
        - 得分 Float32MultiArray（/grasp/scores）
        - 元数据 Float32MultiArray（/grasp/metadata）
        - 预览最佳抓取姿态 PoseStamped
        - 预览最佳抓取得分 Float32
    """

    def __init__(self):
        super().__init__("graspnet_inference", automatically_declare_parameters_from_overrides=True)

        # 声明所有参数的默认值（如果尚未声明）
        self._declare_defaults()

        # 读取 ROS 参数
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.info_topic = str(self.get_parameter("camera_info_topic").value)
        self.poses_topic = str(self.get_parameter("poses_topic").value)
        self.scores_topic = str(self.get_parameter("scores_topic").value)
        self.metadata_topic = str(self.get_parameter("metadata_topic").value)
        self.preview_best_pose_topic = str(self.get_parameter("preview_best_pose_topic").value)
        self.preview_best_score_topic = str(self.get_parameter("preview_best_score_topic").value)

        # GraspNet 模型路径与配置
        self.baseline_dir = str(self.get_parameter("baseline_dir").value)
        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.num_point = int(self.get_parameter("num_point").value)
        self.top_k_publish = max(1, int(self.get_parameter("top_k_publish").value))
        self.min_grasp_width_m = float(self.get_parameter("min_grasp_width_m").value)
        self.max_grasp_width_m = float(self.get_parameter("max_grasp_width_m").value)
        if self.min_grasp_width_m < 0.0 or self.max_grasp_width_m < self.min_grasp_width_m:
            raise ValueError("Require 0 <= min_grasp_width_m <= max_grasp_width_m.")
        self.support_plane_distance_m = float(self.get_parameter("support_plane_distance_m").value)
        self.support_plane_min_inlier_ratio = float(
            self.get_parameter("support_plane_min_inlier_ratio").value
        )
        self.support_plane_max_tilt_deg = float(
            self.get_parameter("support_plane_max_tilt_deg").value
        )
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.workspace_x_min_m = float(self.get_parameter("workspace_x_min_m").value)
        self.workspace_x_max_m = float(self.get_parameter("workspace_x_max_m").value)
        self.workspace_y_min_m = float(self.get_parameter("workspace_y_min_m").value)
        self.workspace_y_max_m = float(self.get_parameter("workspace_y_max_m").value)
        self.object_min_height_m = float(self.get_parameter("object_min_height_m").value)
        self.object_max_height_m = float(self.get_parameter("object_max_height_m").value)
        self.collision_detection_enabled = bool(
            self.get_parameter("collision_detection_enabled").value
        )
        self.collision_voxel_size_m = float(self.get_parameter("collision_voxel_size_m").value)
        self.collision_approach_distance_m = float(
            self.get_parameter("collision_approach_distance_m").value
        )
        self.collision_threshold = float(self.get_parameter("collision_threshold").value)
        if self.support_plane_distance_m <= 0.0:
            raise ValueError("support_plane_distance_m must be positive.")
        if not 0.0 < self.support_plane_min_inlier_ratio <= 1.0:
            raise ValueError("support_plane_min_inlier_ratio must be in (0, 1].")
        if not 0.0 <= self.support_plane_max_tilt_deg < 90.0:
            raise ValueError("support_plane_max_tilt_deg must be in [0, 90).")
        if self.workspace_x_min_m > self.workspace_x_max_m:
            raise ValueError("Require workspace_x_min_m <= workspace_x_max_m.")
        if self.workspace_y_min_m > self.workspace_y_max_m:
            raise ValueError("Require workspace_y_min_m <= workspace_y_max_m.")
        if self.object_min_height_m < 0.0 or self.object_max_height_m < self.object_min_height_m:
            raise ValueError("Require 0 <= object_min_height_m <= object_max_height_m.")
        if self.collision_voxel_size_m <= 0.0:
            raise ValueError("collision_voxel_size_m must be positive.")
        if self.collision_approach_distance_m < 0.0 or self.collision_threshold < 0.0:
            raise ValueError("collision approach distance and threshold must be non-negative.")
        self.min_valid_points = int(self.get_parameter("min_valid_points").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)

        # 发布前需人工确认。
        self.confirm_before_publish = bool(
            self.get_parameter("confirm_before_publish").value
        )
        # 确认时可视化的候选数。
        self.confirm_visual_top_k = max(
            1, int(self.get_parameter("confirm_visual_top_k").value)
        )
        self.confirm_window_name = str(self.get_parameter("confirm_window_name").value)

        # 时间同步参数
        self.sync_queue = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop_s").value)
        self.active_mode = str(self.get_parameter("initial_mode").value).strip().lower()
        self.rgbd_wait_timeout_sec = float(
            self.get_parameter("rgbd_wait_timeout_sec").value
        )

        # 随机数生成器
        self.rng = np.random.default_rng(int(self.get_parameter("random_seed").value))

        # 动态加载 GraspNet 模块
        modules = _load_graspnet_modules(self.baseline_dir)
        (
            self.torch,
            self.GraspNet,
            self.pred_decode,
            self.GNCameraInfo,
            self.create_point_cloud_from_depth_image,
            self.GraspGroup,
            self.ModelFreeCollisionDetector,
        ) = modules

        # 设置设备（优先 GPU）
        self.device = self.torch.device("cuda:0" if self.torch.cuda.is_available() else "cpu")
        self.net = self._load_net() if self.active_mode == "graspnet" else None

        # CV Bridge（ROS ↔ OpenCV 图像转换）
        self.bridge = CvBridge()

        # 线程锁：保护共享数据
        self._lock = threading.Lock()
        self._compute_lock = threading.Lock()  # 防止并发推理
        self._preview_cache = None
        self._preview_visualizer = None
        self._preview_id = ""
        self._latest: Optional[Tuple[Image, Image, CameraInfo]] = None  # 最近一次同步数据
        self._latest_received_at = 0.0
        self._mode_enabled_at = time.monotonic()
        self._camera_info: Optional[CameraInfo] = None
        self.camera_info_sub = None
        self.rgb_sub = self.depth_sub = self.ats = None
        self._callback_group = ReentrantCallbackGroup()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # 发布者 QoS 配置（可靠，保留最新 5 条）
        out_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pose_pub = self.create_publisher(PoseArray, self.poses_topic, out_qos)
        self.score_pub = self.create_publisher(Float32MultiArray, self.scores_topic, out_qos)
        self.metadata_pub = self.create_publisher(Float32MultiArray, self.metadata_topic, out_qos)
        self.preview_pose_pub = self.create_publisher(PoseStamped, self.preview_best_pose_topic, out_qos)
        self.preview_score_pub = self.create_publisher(Float32, self.preview_best_score_topic, out_qos)
        self.preview_state_pub = self.create_publisher(
            String, "/llm_control/graspnet_preview_state", 10
        )
        self.create_subscription(
            String,
            "/llm_control/graspnet_preview_control",
            self._on_external_preview_control,
            10,
        )
        self.create_timer(0.05, self._poll_external_preview)

        # 服务：/grasp/compute
        self.create_service(
            Trigger, "/grasp/compute", self.on_compute,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger, "/grasp/release_gpu", self.on_release_gpu,
            callback_group=self._callback_group,
        )
        mode_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/llm_control/active_mode", self.on_active_mode, mode_qos,
            callback_group=self._callback_group,
        )

        # CameraInfo 不是图像帧：锁存规范内参，避免低频旧时间戳阻塞 RGB-D 缓存。
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.on_camera_info,
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )

        if self.active_mode == "graspnet":
            self._start_rgbd_subscriptions()

        # 仅在 GraspNet 模式同步实际采集的 RGB-D 图像对。
        self.get_logger().info(f"Initial GraspNet mode: {self.active_mode}")
        self.get_logger().info(f"GraspNet checkpoint: {self.checkpoint_path}")
        self.get_logger().info(
            f"RGB/Depth/Info: {self.rgb_topic}, {self.depth_topic}, {self.info_topic}"
        )
        self.get_logger().info(
            "Support-plane/workspace filtering enabled, "
            f"top_k_publish={self.top_k_publish}, "
            f"collision_detection={self.collision_detection_enabled}"
        )
        self.get_logger().info(
            f"Confirm before publish={self.confirm_before_publish}, "
            f"confirm_visual_top_k={self.confirm_visual_top_k}"
        )

    def _start_rgbd_subscriptions(self):
        if self.rgb_sub is not None:
            return
        self._latest = None
        self._latest_received_at = 0.0
        self._mode_enabled_at = time.monotonic()
        self.rgb_sub = Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        self.depth_sub = Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        self.ats = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
        )
        self.ats.registerCallback(self.on_synced)

    def _stop_rgbd_subscriptions(self):
        for subscriber in (self.rgb_sub, self.depth_sub):
            if subscriber is not None:
                subscriber.unsubscribe()
        self.rgb_sub = self.depth_sub = self.ats = None
        with self._lock:
            self._latest = None
            self._latest_received_at = 0.0

    def on_active_mode(self, msg):
        mode = str(msg.data).strip().lower()
        if mode not in ("yolo", "graspnet") or mode == self.active_mode:
            return
        self.active_mode = mode
        if mode == "graspnet":
            self._start_rgbd_subscriptions()
        else:
            self._stop_rgbd_subscriptions()
            model, self.net = self.net, None
            inference_runtime.release_model(self.torch, model)
        self.get_logger().info(f"GraspNet input mode switched to {mode}")

    def _declare_defaults(self):
        """声明所有可配置参数的默认值（仅当参数尚未声明时）。"""
        defaults = {
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            "poses_topic": "/grasp/poses",
            "scores_topic": "/grasp/scores",
            "metadata_topic": "/grasp/metadata",
            "preview_best_pose_topic": "/graspnet_bringup/preview_best_pose",
            "preview_best_score_topic": "/graspnet_bringup/preview_best_score",
            "baseline_dir": _graspnet_source_path("graspnet_baseline"),
            "checkpoint_path": _graspnet_source_path("models", "checkpoint-rs.tar"),
            "num_point": 20000,
            "top_k_publish": 5,
            "min_grasp_width_m": 0.005,
            "max_grasp_width_m": 0.061,
            "support_plane_distance_m": 0.005,
            "support_plane_min_inlier_ratio": 0.20,
            "support_plane_max_tilt_deg": 15.0,
            "base_frame": "base_link",
            "workspace_x_min_m": -0.30,
            "workspace_x_max_m": 0.30,
            "workspace_y_min_m": 0.00,
            "workspace_y_max_m": 0.60,
            "object_min_height_m": 0.002,
            "object_max_height_m": 0.150,
            "collision_detection_enabled": True,
            "collision_voxel_size_m": 0.005,
            "collision_approach_distance_m": 0.08,
            "collision_threshold": 0.01,
            "min_valid_points": 2000,
            "depth_min_m": 0.05,
            "depth_max_m": 5.0,
            "sync_queue_size": 3,
            "sync_slop_s": 0.05,
            "initial_mode": "graspnet",
            "rgbd_wait_timeout_sec": 3.5,
            "confirm_before_publish": False,
            "confirm_visual_top_k": 50,
            "confirm_window_name": "GraspNet: E=execute, B=preview raw best, ESC/Q=cancel",
            "random_seed": 0,
        }
        for name, value in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

    def _load_net(self):
        """加载 GraspNet 模型权重并设置为 eval 模式。"""
        return inference_runtime.load_model(
            self.torch, self.GraspNet, self.checkpoint_path, self.device
        )

    def _ensure_net_loaded(self):
        if self.net is None:
            self.get_logger().info("Reloading GraspNet GPU model for /grasp/compute.")
            self.net = self._load_net()

    def on_release_gpu(self, _req: Trigger.Request, resp: Trigger.Response):
        """Release the GraspNet model and stop inactive RGB-D subscriptions."""
        if not self._compute_lock.acquire(blocking=False):
            resp.success = False
            resp.message = "GraspNet inference is already running."
            return resp
        try:
            model, self.net = self.net, None
            inference_runtime.release_model(self.torch, model)
            if self.active_mode != "graspnet":
                self._stop_rgbd_subscriptions()
            resp.success = True
            resp.message = "GraspNet GPU model released"
            self.get_logger().info(resp.message)
            return resp
        finally:
            self._compute_lock.release()

    def destroy_node(self):
        self._close_external_preview()
        model, self.net = self.net, None
        inference_runtime.release_model(self.torch, model)
        super().destroy_node()

    # ═══════════════════════════════════════════════════════
    #  消息同步回调
    # ═══════════════════════════════════════════════════════

    def on_camera_info(self, info_msg: CameraInfo):
        """锁存当前 profile 的规范内参后停止订阅，避免 CameraInfo 参与时间同步。"""
        if not _valid_camera_info(info_msg):
            self.get_logger().warn(
                f"Ignoring invalid CameraInfo from {self.info_topic}: "
                f"frame={info_msg.header.frame_id!r}, size={info_msg.width}x{info_msg.height}, "
                f"fx={info_msg.k[0]:.3f}, fy={info_msg.k[4]:.3f}"
            )
            return
        with self._lock:
            if self._camera_info is not None:
                return
            self._camera_info = info_msg
            subscription = self.camera_info_sub
            self.camera_info_sub = None
        if subscription is not None:
            self.destroy_subscription(subscription)
        self.get_logger().info(
            f"CameraInfo locked: topic={self.info_topic}, frame={info_msg.header.frame_id}, "
            f"size={info_msg.width}x{info_msg.height}, fx={info_msg.k[0]:.3f}, "
            f"fy={info_msg.k[4]:.3f}, cx={info_msg.k[2]:.3f}, cy={info_msg.k[5]:.3f}"
        )

    def on_synced(self, rgb_msg: Image, depth_msg: Image):
        """缓存最新 RGB-D 采集对；CameraInfo 必须已独立锁存。"""
        with self._lock:
            if self._camera_info is None:
                return
            info_msg = self._camera_info
            self._latest = (rgb_msg, depth_msg, info_msg)
            self._latest_received_at = time.monotonic()

    # ═══════════════════════════════════════════════════════
    #  服务回调
    # ═══════════════════════════════════════════════════════

    def on_compute(self, _req: Trigger.Request, resp: Trigger.Response):
        """
        服务 /grasp/compute 的回调：执行一次推理并发布结果。
        若已有推理在运行，返回失败。
        """
        if not self._compute_lock.acquire(blocking=False):
            resp.success = False
            resp.message = "GraspNet inference is already running."
            return resp
        try:
            mode_deadline = time.monotonic() + 0.5
            while self.active_mode != "graspnet" and time.monotonic() < mode_deadline:
                time.sleep(0.02)
            if self.active_mode != "graspnet":
                resp.success = False
                resp.message = "GraspNet mode is inactive."
                return resp
            GraspnetInferenceNode._ensure_net_loaded(self)
            deadline = time.monotonic() + max(0.0, self.rgbd_wait_timeout_sec)
            latest = None
            while True:
                with self._lock:
                    if self._latest_received_at >= self._mode_enabled_at:
                        latest = self._latest
                if latest is not None:
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            if latest is None:
                resp.success = False
                resp.message = "No CameraInfo latch and synchronized RGB/Depth received yet."
                return resp
            count = self._infer_and_publish(*latest)
            resp.success = True
            resp.message = f"Published {count} GraspNet grasp candidates."
            return resp
        except RuntimeError as exc:
            if str(exc) == "Grasp confirmation canceled by user.":
                resp.success = False
                resp.message = f"CANCELED: {exc}"
                self.get_logger().info(resp.message)
                return resp
            resp.success = False
            resp.message = f"Inference failed: {exc}"
            self.get_logger().error(resp.message)
            return resp
        except Exception as exc:
            resp.success = False
            resp.message = f"Inference failed: {exc}"
            self.get_logger().error(resp.message)
            return resp
        finally:
            torch = getattr(self, "torch", None)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._compute_lock.release()

    # ═══════════════════════════════════════════════════════
    #  核心推理与发布流程
    # ═══════════════════════════════════════════════════════

    def _infer_and_publish(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> int:
        """
        执行完整的抓取推理流水线：
            1. 将 ROS 图像转为 numpy
            2. 调用 _generate_grasps 获得所有抓取和采样点云
            3. 根据配置进行可视化确认
            4. 转换数据并发布
        返回发布的抓取数量。
        """
        stamp = rgb_msg.header.stamp
        self.get_logger().info(
            f"GraspNet compute RGB-D capture stamp={stamp.sec}.{stamp.nanosec:09d}"
        )
        rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        grasp_all, grasp_pub, debug_clouds = self._generate_grasps(
            np.asarray(rgb_bgr),
            np.asarray(depth_raw),
            info_msg,
            rgb_msg.header.stamp,
        )
        # 确定坐标系和时间戳
        frame_id = info_msg.header.frame_id

        if self.confirm_before_publish:
            # 需要用户确认（仅当确认通过才继续）
            if not self._confirm_grasps(grasp_all, debug_clouds, frame_id, stamp):
                raise RuntimeError("Grasp confirmation canceled by user.")

        # 将 GraspGroup 转换为姿态和元数据
        poses_np, metadata = _graspgroup_to_pose_metadata(grasp_pub)
        self._publish_results(poses_np, metadata, frame_id, stamp)
        self._preview_cache = {
            "grasp_group": grasp_pub,
            "debug_clouds": debug_clouds,
            "frame_id": frame_id,
            "stamp": stamp,
            "capture_stamp": f"{int(stamp.sec)}.{int(stamp.nanosec):09d}",
        }
        return int(poses_np.shape[0])

    def _generate_grasps(
        self,
        rgb_bgr: np.ndarray,
        depth_raw: np.ndarray,
        info: CameraInfo,
        stamp,
    ):
        """
        GraspNet 抓取生成核心逻辑：
            - 深度预处理与支撑平面过滤
            - 创建点云
            - 随机采样
            - 前向推理
            - 解码、NMS、排序
        返回: (全部抓取 GraspGroup, 发布的 top-k GraspGroup, Open3D 调试点云)
        """
        # 深度图转换为米（假设原始为毫米的 uint16）
        depth_m = (
            depth_raw.astype(np.float32) / 1000.0
            if depth_raw.dtype == np.uint16
            else depth_raw.astype(np.float32)
        )
        height, width = depth_m.shape[:2]
        # RGB 转为 [0,1] 浮点数
        color = rgb_bgr[..., ::-1].astype(np.float32) / 255.0

        # 解析相机内参
        fx, fy, cx, cy = float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5])
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError("Invalid camera intrinsics: fx/fy is zero.")
        camera = self.GNCameraInfo(width, height, fx, fy, cx, cy, 1.0)
        # 生成有组织的点云
        cloud_org = self.create_point_cloud_from_depth_image(depth_m, camera, organized=True)

        depth_mask = (depth_m > self.depth_min_m) & (depth_m < self.depth_max_m)
        mask, workspace_points, debug_clouds, stats = self._support_plane_mask(
            cloud_org, depth_mask, info, stamp
        )
        valid = int(mask.sum())
        self.get_logger().info(
            "RANSAC support plane: "
            f"inlier_ratio={stats['plane_ratio']:.3f}, "
            f"distance_threshold_m={self.support_plane_distance_m:.3f}"
        )
        self.get_logger().info(
            "Raw depth points:       {raw}\n"
            "Workspace points:       {workspace}\n"
            "Plane inliers:          {plane}\n"
            "Above-plane points:     {above}\n"
            "Below-plane rejected:   {below}\n"
            "Final GraspNet points:  {final}".format(**stats)
        )
        if valid < self.min_valid_points:
            raise RuntimeError(f"Too few valid points after support-plane filtering: {valid}")

        # 提取支撑平面之外的点云和颜色
        cloud_masked = cloud_org[mask]
        color_masked = color[mask]
        # 随机采样固定数量的点
        if len(cloud_masked) >= self.num_point:
            indices = self.rng.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            base = np.arange(len(cloud_masked))
            extra = self.rng.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            indices = np.concatenate([base, extra], axis=0)

        cloud_sampled = cloud_masked[indices]
        color_sampled = color_masked[indices]

        # 构建输入 batch
        end_points = {
            "point_clouds": self.torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device),
            "cloud_colors": color_sampled,
        }
        # 前向推理
        with self.torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = self.pred_decode(end_points)

        grasp_array = grasp_preds[0].detach().cpu().numpy()
        if grasp_array.size == 0:
            raise RuntimeError("pred_decode produced no grasps.")
        # 创建 GraspGroup，NMS 过滤并按得分排序
        grasp_group = self.GraspGroup(grasp_array)
        raw_count = len(grasp_group)
        grasp_group.nms()
        grasp_group.sort_by_score()
        nms_count = len(grasp_group)
        nms_widths = np.asarray(grasp_group.widths, dtype=np.float32)
        nms_finite_widths = nms_widths[np.isfinite(nms_widths)]
        nms_width_range = (
            "n/a"
            if len(nms_finite_widths) == 0
            else f"[{nms_finite_widths.min():.4f}, {nms_finite_widths.max():.4f}] m"
        )
        grasp_group, feasible_count = _filter_grasp_group_by_width(
            grasp_group,
            self.min_grasp_width_m,
            self.max_grasp_width_m,
        )
        if feasible_count == 0:
            raise RuntimeError(
                'no width-feasible grasp candidates: '
                f'allowed=[{self.min_grasp_width_m:.3f}, {self.max_grasp_width_m:.3f}] m, '
                f'nms_width_range={nms_width_range}'
            )
        collision_rejected = 0
        if self.collision_detection_enabled:
            grasp_group, collision_rejected = _filter_collision_free_grasps(
                self.ModelFreeCollisionDetector,
                workspace_points,
                grasp_group,
                self.collision_voxel_size_m,
                self.collision_approach_distance_m,
                self.collision_threshold,
            )
            if len(grasp_group) == 0:
                raise RuntimeError("no collision-free grasp candidates after workspace collision filtering")
        # 确认和发布均只使用物理可执行的候选。
        grasp_pub = grasp_group[: self.top_k_publish]
        self.get_logger().info(
            'GraspNet candidates: '
            f'raw={raw_count}, nms={nms_count}, nms_width_range={nms_width_range}, '
            f'width_feasible={feasible_count}, '
            f'collision_rejected={collision_rejected}, collision_free={len(grasp_group)}, '
            f'published={len(grasp_pub)}'
        )
        return grasp_group, grasp_pub, debug_clouds

    def _support_plane_mask(
        self,
        cloud_org: np.ndarray,
        depth_mask: np.ndarray,
        info: CameraInfo,
        stamp,
    ):
        """Keep only the configured above-table object-height band in the base-frame workspace."""
        import open3d as o3d

        valid_indices = np.flatnonzero(depth_mask.reshape(-1))
        points = cloud_org.reshape(-1, 3)[valid_indices]
        frame_id = info.header.frame_id
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_frame,
                frame_id,
                rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=0.5),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Support-plane TF lookup failed ({self.base_frame} <- {frame_id}): {exc}"
            ) from exc
        rotation = transform.transform.rotation
        rotation_matrix = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()
        translation = transform.transform.translation
        points_in_base = points @ rotation_matrix.T + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )
        workspace_keep = _workspace_mask(
            points_in_base,
            self.workspace_x_min_m,
            self.workspace_x_max_m,
            self.workspace_y_min_m,
            self.workspace_y_max_m,
        )
        workspace_indices = valid_indices[workspace_keep]
        workspace_points = points[workspace_keep]
        if len(workspace_points) < 3:
            raise RuntimeError("Too few workspace points to estimate support plane.")
        plane_cloud = o3d.geometry.PointCloud()
        plane_cloud.points = o3d.utility.Vector3dVector(workspace_points.astype(np.float64))
        plane_cloud = plane_cloud.voxel_down_sample(voxel_size=0.005)
        if len(plane_cloud.points) < 3:
            raise RuntimeError("Too few valid points to estimate support plane.")
        plane_model, inliers = plane_cloud.segment_plane(
            distance_threshold=self.support_plane_distance_m,
            ransac_n=3,
            num_iterations=1000,
        )
        plane_ratio = len(inliers) / len(plane_cloud.points)
        if plane_ratio < self.support_plane_min_inlier_ratio:
            raise RuntimeError(
                f"Support plane rejected: inlier_ratio={plane_ratio:.3f} < "
                f"{self.support_plane_min_inlier_ratio:.3f}."
            )
        signed_distances = _support_plane_signed_distances(
            workspace_points,
            np.asarray(plane_model),
            rotation_matrix,
            self.support_plane_max_tilt_deg,
        )
        if signed_distances is None:
            raise RuntimeError(
                "Support plane rejected: tilt exceeds "
                f"{self.support_plane_max_tilt_deg:.1f} degrees."
            )
        plane_inlier_mask = np.abs(signed_distances) <= self.support_plane_distance_m
        above_plane_mask = signed_distances > self.object_min_height_m
        keep = _object_height_mask(
            signed_distances,
            self.object_min_height_m,
            self.object_max_height_m,
        )
        mask_flat = np.zeros(depth_mask.size, dtype=bool)
        mask_flat[workspace_indices[keep]] = True
        stats = {
            "raw": int(len(points)),
            "workspace": int(len(workspace_points)),
            "plane": int(plane_inlier_mask.sum()),
            "above": int(above_plane_mask.sum()),
            "below": int((~above_plane_mask).sum()),
            "final": int(keep.sum()),
            "plane_ratio": plane_ratio,
        }
        debug_clouds = {
            "raw": points,
            "plane": workspace_points[plane_inlier_mask],
            "final": workspace_points[keep],
            "rejected": workspace_points[~above_plane_mask],
        }
        return mask_flat.reshape(depth_mask.shape), workspace_points, debug_clouds, stats

    def _publish_external_preview_state(self, state: str, error: str = ""):
        self.preview_state_pub.publish(String(data=json.dumps({
            "preview_id": self._preview_id,
            "state": str(state),
            "error": str(error),
        }, ensure_ascii=False)))

    def _close_external_preview(self, state="closed", error=""):
        visualizer = self._preview_visualizer
        if visualizer is not None:
            try:
                visualizer.destroy_window()
            except Exception as exc:
                self.get_logger().warning(f"Open3D preview close failed: {exc}")
        self._preview_visualizer = None
        if self._preview_id:
            self._publish_external_preview_state(state, error)
        self._preview_id = ""

    def _on_external_preview_control(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        command = str(payload.get("command", ""))
        if command == "close":
            if not self._preview_id or payload.get("preview_id") == self._preview_id:
                self._close_external_preview()
            return
        if command != "show":
            return
        preview_id = str(payload.get("preview_id", ""))
        if not preview_id:
            return
        self._close_external_preview()
        self._preview_id = preview_id
        self._publish_external_preview_state("opening")
        cache = self._preview_cache
        if cache is None:
            self._publish_external_preview_state("failed", "preview cache unavailable")
            self._preview_id = ""
            return
        if str(payload.get("capture_stamp", "")) != cache["capture_stamp"]:
            self._publish_external_preview_state("failed", "capture stamp mismatch")
            self._preview_id = ""
            return
        index = int(payload.get("candidate_index", -1))
        if index < 0 or index >= len(cache["grasp_group"]):
            self._publish_external_preview_state("failed", "candidate index unavailable")
            self._preview_id = ""
            return
        try:
            import open3d as o3d

            def cloud(points, color):
                result = o3d.geometry.PointCloud()
                result.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
                result.paint_uniform_color(color)
                return result.voxel_down_sample(voxel_size=0.005)

            vis = o3d.visualization.Visualizer()
            if not vis.create_window(
                window_name="GraspNet selected executable grasp",
                width=1280,
                height=720,
            ):
                raise RuntimeError("failed to create Open3D window")
            clouds = cache["debug_clouds"]
            for geometry in (
                cloud(clouds["raw"], (0.55, 0.55, 0.55)),
                cloud(clouds["rejected"], (0.12, 0.12, 0.12)),
                cloud(clouds["plane"], (1.0, 0.0, 0.0)),
                cloud(clouds["final"], (0.0, 1.0, 0.0)),
                o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1),
            ):
                vis.add_geometry(geometry)
            gripper = cache["grasp_group"][index:index + 1].to_open3d_geometry_list()[0]
            gripper.paint_uniform_color((0.0, 0.0, 1.0))
            vis.add_geometry(gripper)
            self._preview_visualizer = vis
            self._publish_preview_best_pose(
                cache["grasp_group"][index:index + 1], cache["frame_id"], cache["stamp"]
            )
            self._publish_external_preview_state("open")
        except Exception as exc:
            self.get_logger().error(f"Open3D external preview failed: {exc}")
            self._close_external_preview("failed", str(exc))

    def _poll_external_preview(self):
        vis = self._preview_visualizer
        if vis is None:
            return
        try:
            if not vis.poll_events():
                self._close_external_preview("closed")
                return
            vis.update_renderer()
        except Exception as exc:
            self._close_external_preview("failed", str(exc))

    def _confirm_grasps(self, grasp_group, debug_clouds, frame_id: str, stamp) -> bool:
        """
        交互式确认窗口：
            - 显示点云、坐标系和若干抓取候选
            - 按键操作：
                E: 确认当前所有候选并发布
                B: 只显示最佳抓取，并发布预览姿态
                ESC/Q/关闭窗口: 取消
        返回 True 表示用户确认发布，False 表示取消。
        """
        if len(grasp_group) == 0:
            raise RuntimeError("No GraspNet grasps available for confirmation.")

        try:
            import open3d as o3d
        except Exception as exc:
            raise RuntimeError(f"Open3D is required for grasp confirmation: {exc}") from exc

        def cloud(points, color):
            result = o3d.geometry.PointCloud()
            result.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
            result.paint_uniform_color(color)
            return result.voxel_down_sample(voxel_size=0.005)

        raw_cloud = cloud(debug_clouds["raw"], (0.55, 0.55, 0.55))
        plane_cloud = cloud(debug_clouds["plane"], (1.0, 0.0, 0.0))
        final_cloud = cloud(debug_clouds["final"], (0.0, 1.0, 0.0))
        rejected_cloud = cloud(debug_clouds["rejected"], (0.12, 0.12, 0.12))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # 显示前 confirm_visual_top_k 个抓取，均以蓝色表示。
        grasp_vis = grasp_group[: self.confirm_visual_top_k]
        candidate_grippers = grasp_vis.to_open3d_geometry_list()
        for gripper in candidate_grippers:
            gripper.paint_uniform_color((0.0, 0.0, 1.0))

        accepted = {"value": None}  # 用字典存储用户决定（True/False/None）

        # 按键回调
        def accept(vis):
            accepted["value"] = True
            vis.close()
            return False

        def cancel(vis):
            accepted["value"] = False
            vis.close()
            return False

        def show_best_only(vis):
            # 移除除最佳以外的所有抓取模型
            for gripper in candidate_grippers[1:]:
                vis.remove_geometry(gripper, reset_bounding_box=False)
            vis.update_renderer()
            # 发布最佳抓取的预览姿态
            self._publish_preview_best_pose(grasp_group[:1], frame_id, stamp)
            self.get_logger().info(
                "Showing best GraspNet grasp only and published preview pose. "
                "Press E to execute or ESC/Q to cancel."
            )
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window(window_name=self.confirm_window_name, width=1280, height=720):
            raise RuntimeError("Failed to create Open3D confirmation window. Check DISPLAY/GUI access.")

        try:
            vis.add_geometry(raw_cloud)
            vis.add_geometry(rejected_cloud)
            vis.add_geometry(plane_cloud)
            vis.add_geometry(final_cloud)
            vis.add_geometry(frame)
            for gripper in candidate_grippers:
                vis.add_geometry(gripper)
            # 注册按键
            vis.register_key_callback(ord("E"), accept)
            vis.register_key_callback(ord("e"), accept)
            vis.register_key_callback(ord("B"), show_best_only)
            vis.register_key_callback(ord("b"), show_best_only)
            vis.register_key_callback(ord("Q"), cancel)
            vis.register_key_callback(ord("q"), cancel)
            vis.register_key_callback(256, cancel)        # ESC
            self.get_logger().info(
                "Grasp confirmation window opened. Press B to preview the raw best grasp; "
                "press E to publish candidates and execute the first grasp that passes "
                "safety and planning checks; press ESC/Q or close window to cancel."
            )
            vis.run()
        finally:
            vis.destroy_window()

        return bool(accepted["value"])

    # ═══════════════════════════════════════════════════════
    #  发布函数
    # ═══════════════════════════════════════════════════════

    def _publish_preview_best_pose(self, grasp_group, frame_id: str, stamp):
        """发布最佳抓取姿态的预览（单个 PoseStamped 和得分）。"""
        poses_np, metadata = _graspgroup_to_pose_metadata(grasp_group)
        if poses_np.shape[0] == 0:
            return
        score = float(metadata[0][0]) if metadata else float("nan")
        score_msg = Float32()
        score_msg.data = score
        self.preview_score_pub.publish(score_msg)

        x, y, z, qx, qy, qz, qw = [float(value) for value in poses_np[0].tolist()]
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.preview_pose_pub.publish(pose_msg)
        self.get_logger().info(f"Published best GraspNet preview pose score={score:.4f} frame={frame_id}")

    def _publish_results(
        self,
        poses_np: np.ndarray,
        metadata: List[Tuple[float, float, float]],
        frame_id: str,
        stamp,
    ):
        """发布抓取姿态、得分和元数据（宽度、深度）。"""
        # 构建 PoseArray
        poses_msg = PoseArray()
        poses_msg.header.frame_id = frame_id
        poses_msg.header.stamp = stamp
        for row in poses_np:
            x, y, z, qx, qy, qz, qw = [float(value) for value in row.tolist()]
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            poses_msg.poses.append(pose)

        count = len(metadata)
        # 得分
        scores_msg = Float32MultiArray()
        scores_msg.data = [float(row[0]) for row in metadata]

        # 元数据（每行 3 个值：score, width, depth）
        metadata_msg = Float32MultiArray()
        metadata_msg.layout.dim = [
            MultiArrayDimension(label="grasp", size=count, stride=count * 3),
            MultiArrayDimension(label="field", size=3, stride=3),
        ]
        metadata_msg.data = [float(value) for row in metadata for value in row]

        self.score_pub.publish(scores_msg)
        self.metadata_pub.publish(metadata_msg)
        self.pose_pub.publish(poses_msg)


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = GraspnetInferenceNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            executor.shutdown(timeout_sec=2.0)
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
