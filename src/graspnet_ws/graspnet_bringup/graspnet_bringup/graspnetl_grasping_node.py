#!/usr/bin/env python3
"""
视觉抓取流水线节点 (GraspnetVisualGraspingNode)

功能：
    - 通过 GraspNet 生成抓取候选，接收 PoseArray 与评分/元数据话题。
    - 状态机控制：pre-grasp pose -> 开爪 -> 计算抓取 -> 选择候选 -> 移动到抓取位姿 -> 闭合 -> 提升 -> 回到 pre-grasp pose。
    - 双 MoveIt 客户端（Fairino / KDL）支持动态切换与交叉回退。
    - 使用运动执行模块 (MoveItMotion) 规划与执行轨迹。
    - 处理 TF 变换、姿态修正、轨迹预检等。
"""

import threading
import time
from typing import Optional, Sequence

import rclpy
import tf2_geometry_msgs
import tf2_ros
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from pymoveit2 import MoveIt2
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import Trigger

# 项目内部模块
from myrobot_common.planning.motion_executor import MoveItMotion, PlanScoreConfig, PlannerSwitch
from myrobot_common.planning.trajectory_scoring import select_best_path
from myrobot_common.task.abort_manager import AbortManager
from myrobot_common.utils.params import param
from myrobot_common.utils.pose_tools import PoseTools
from graspnet_bringup.task.graspnet_state_machine import GraspnetStateMachine
from graspnet_bringup.task.task_types import GraspCandidate, GraspState
from graspnet_bringup.task import graspnet_candidate_utils as _candidate_utils
from graspnet_bringup.task.graspnet_candidate_utils import (
    build_candidates,
    candidate_geometry_rejection,
    prepare_candidate,
)

# Kept for source-level compatibility with existing tests and local tools.
_apply_orientation_correction = _candidate_utils._apply_orientation_correction
_make_lift_pose = _candidate_utils._make_lift_pose
_offset_pose_along_axis = _candidate_utils._offset_pose_along_axis
_pose_axis = _candidate_utils._pose_axis
_preopen_positions_from_width = _candidate_utils._preopen_positions_from_width


_CAPTURE_TIME_TF_HISTORY_SEC = 120.0


# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

def _float_list(value, fallback: Sequence[float]) -> list[float]:
    """
    安全解析逗号分号分隔的浮点数列表。
    若解析结果为空，返回 fallback。
    """
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(";", ",").split(",")]
        parsed = [float(item) for item in parts if item]
    else:
        parsed = [float(item) for item in value]
    return parsed if parsed else list(fallback)


# ═══════════════════════════════════════════════════════════
#  主节点类
# ═══════════════════════════════════════════════════════════

class GraspnetVisualGraspingNode(Node):
    """ROS2 节点：基于 GraspNet 的视觉抓取流水线。"""

    def __init__(self):
        super().__init__(
            "graspnet_visual_grasping",
            automatically_declare_parameters_from_overrides=True,
        )

        # 回调组：Reentrant 用于并发处理服务/订阅，MutuallyExclusive 用于控制循环和应急中断
        self.callback_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.abort_cb_group = MutuallyExclusiveCallbackGroup()

        self._startup_ready_logged = False      # 是否已打印就绪日志
        self.current_state = GraspState.WAIT_READY
        self.active_mode = "graspnet"

        # 抓取相关缓存
        self._start_seq = 0
        self._grasp_msg: Optional[PoseArray] = None
        self._grasp_scores: list[float] = []
        self._grasp_metadata: list[float] = []
        self._candidates: list[GraspCandidate] = []
        self._active_candidate: Optional[GraspCandidate] = None
        self._g_requested = False
        self._last_compute_error = ""
        self._last_mode_exit_log_time = 0.0

        # controller_manager 服务客户端
        self._controller_manager_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
            callback_group=self.callback_group,
        )
        # GraspNet 推理服务客户端
        self.compute_client = self.create_client(
            Trigger,
            "/grasp/compute",
            callback_group=self.callback_group,
        )

        # 加载参数并初始化工具
        self._load_params()
        self.pose_tools = PoseTools(self, base_frame=self.base_frame)
        self.pregrasp_pose = self._build_pregrasp_pose()

        # 保留人工确认期间的采集时刻 TF；仍不允许退回到 latest TF。
        self._tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=_CAPTURE_TIME_TF_HISTORY_SEC)
        )
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.get_logger().info(
            f"Capture-time TF history cache={_CAPTURE_TIME_TF_HISTORY_SEC:.0f}s"
        )

        # 抓取结果缓存（带锁）
        self._latest_poses: Optional[PoseArray] = None
        self._latest_scores: list[float] = []
        self._latest_metadata: list[float] = []
        self._result_seq = 0
        self._result_lock = threading.Lock()

        # 预览姿态与评分缓存（带锁）
        self._latest_preview_pose: Optional[PoseStamped] = None
        self._latest_preview_score: Optional[float] = None
        self._last_preview_plan_key: Optional[tuple[int, int, str]] = None
        self._preview_lock = threading.Lock()

        # 初始化 MoveIt 客户端、中止管理器、运动执行器
        self._setup_moveit()
        self.abort = AbortManager(self, arm=self.moveit2_arm, gripper=self.moveit2_gripper)
        self.abort.set_command_enabled(lambda: self.active_mode == "graspnet")
        self.motion = MoveItMotion(
            node=self,
            arm_clients={"fairino": self.moveit2_arm_fairino, "kdl": self.moveit2_arm_kdl},
            default_client=self.ik_plugin,
            gripper=self.moveit2_gripper,
            pose_tools=self.pose_tools,
            abort=self.abort,
            select_best_path=select_best_path,
            score_cfg=PlanScoreConfig(
                num_candidates=self.num_candidate_plans,
                wrist_weight=self.wrist_weight,
                wrist_joint_indices=self.wrist_joint_indices,
            ),
            action_delay=self.action_delay,
            joint_constraint=self.j2_constraint,
            open_positions=self.gripper_open_positions,
            close_positions=self.gripper_close_positions,
        )
        self.motion.set_ik(self.ik_plugin)
        self.ik_plugin = self.motion.current_client
        self.abort.set_recovery_hooks(
            open_gripper_fn=lambda: self.motion.control_gripper(open_gripper=True, timeout_sec=30.0),
            close_gripper_fn=self._close_gripper_at_pregrasp,
            go_home_fn=self._move_to_pregrasp_pose,
            reset_fn=self._reset_task_cache,
            recovery_complete_fn=self._recovery_complete,
        )

        # QoS 配置（可靠、保留最近一条）
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 订阅抓取结果
        self.create_subscription(PoseArray, self.poses_topic, self._on_poses, qos)
        self.create_subscription(Float32MultiArray, self.scores_topic, self._on_scores, qos)
        self.create_subscription(Float32MultiArray, self.metadata_topic, self._on_metadata, qos)
        # 订阅预览抓取
        self.create_subscription(PoseStamped, self.preview_best_pose_topic, self._on_preview_pose, qos)
        self.create_subscription(Float32, self.preview_best_score_topic, self._on_preview_score, qos)
        self.create_subscription(
            String,
            "/motion_control/command",
            self._on_motion_command,
            10,
            callback_group=self.callback_group,
        )
        # 手动中止
        self.create_subscription(
            Bool,
            "/manual_abort",
            self._on_manual_abort,
            10,
            callback_group=self.abort_cb_group,
        )
        # 发布者
        self.state_pub = self.create_publisher(String, "/graspnet_bringup/state", 10)
        mode_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/llm_control/active_mode", self._on_active_mode, mode_qos,
            callback_group=self.callback_group,
        )
        self.target_pub = self.create_publisher(PoseStamped, "/robot/target_pose", 10)
        self.selected_grasp_6d_pub = self.create_publisher(String, "/graspnet_bringup/selected_grasp_6d", 10)
        self.grasp_plan_6d_pub = self.create_publisher(String, "/graspnet_bringup/grasp_plan_6d", 10)
        self.rejected_grasp_pub = self.create_publisher(String, "/graspnet_bringup/rejected_grasp", 10)

        # 控制循环定时器
        self.state_machine = GraspnetStateMachine(self)
        self.create_timer(
            self.control_period_sec,
            self.state_machine.tick,
            callback_group=self.control_cb_group,
        )

        self.get_logger().info("GraspnetVisualGraspingNode initialized")

    # ═══════════════════════════════════════════════════════
    #  参数加载
    # ═══════════════════════════════════════════════════════

    def _load_params(self):
        """从 ROS 参数服务器加载所有配置参数。"""
        self.arm_group_name = str(param(self, "arm_group_name", "robot_arm"))
        self.hand_group_name = str(param(self, "hand_group_name", "hand"))
        self.base_frame = str(param(self, "base_frame", "base_link"))
        self.camera_frame = str(param(self, "camera_frame", "camera_color_optical_frame"))
        self.ee_frame = str(param(self, "ee_frame", "tool0"))

        # 话题名称
        self.poses_topic = str(param(self, "poses_topic", "/grasp/poses"))
        self.scores_topic = str(param(self, "scores_topic", "/grasp/scores"))
        self.metadata_topic = str(param(self, "metadata_topic", "/grasp/metadata"))
        self.preview_best_pose_topic = str(
            param(self, "preview_best_pose_topic", "/graspnet_bringup/preview_best_pose")
        )
        self.preview_best_score_topic = str(
            param(self, "preview_best_score_topic", "/graspnet_bringup/preview_best_score")
        )

        # MoveIt 客户端命名空间与规划器配置
        self.move_group_ns_fairino = str(param(self, "move_group_ns_fairino", "/move_group_fairino"))
        self.move_group_ns_kdl = str(param(self, "move_group_ns_kdl", "/move_group_kdl"))
        self.move_group_ready_timeout_sec = float(param(self, "move_group_ready_timeout_sec", 10.0))
        self.ik_plugin = PlannerSwitch.normalize_ik(str(param(self, "ik_plugin", "kdl")))
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(
            str(param(self, "planning_pipeline_id", "fairino"))
        )
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            str(param(self, "planner_id", "tube_birrt*")),
        )
        if not PlannerSwitch.is_valid(self.planning_pipeline_id, self.planner_id):
            raise ValueError(
                f"Unsupported planner config: pipeline={self.planning_pipeline_id}, "
                f"planner={self.planner_id}"
            )

        # 运动参数
        self.max_step_size = float(param(self, "max_step_size", 0.05))
        self.arm_max_velocity = float(param(self, "arm_max_velocity", 0.6))
        self.arm_max_acceleration = float(param(self, "arm_max_acceleration", 0.6))
        self.allowed_planning_time = float(param(self, "allowed_planning_time", 15.0))
        self.position_tolerance = float(param(self, "position_tolerance", 0.005))
        self.orientation_tolerance = float(param(self, "orientation_tolerance", 0.005))
        self.allowed_start_tolerance = float(param(self, "allowed_start_tolerance", 0.1))

        # 轨迹评分配置
        self.num_candidate_plans = int(param(self, "num_candidate_plans", 5))
        self.wrist_weight = float(param(self, "wrist_weight", 50.0))
        self.wrist_joint_indices = tuple(int(v) for v in param(self, "wrist_joint_indices", [2, 3, 4]))

        # 夹爪张开位置
        self.gripper_open_positions = tuple(
            float(v) for v in param(self, "gripper_open_positions", [0.0305, -0.0305])
        )
        self.gripper_close_positions = tuple(
            float(v) for v in param(self, "gripper_close_positions", [0.001, -0.001])
        )
        self.use_graspnet_width = bool(param(self, "use_graspnet_width", False))

        # 抓取几何参数
        self.lift_distance = float(param(self, "lift_distance", 0.08))
        self.approach_distance_m = float(param(self, "approach_distance_m", 0.08))
        self.grasp_offset_m = float(param(self, "grasp_offset_m", 0.0))
        self.max_grasp_candidates = int(param(self, "max_grasp_candidates", 5))
        self.max_approach_tilt_deg = float(param(self, "max_approach_tilt_deg", 35.0))
        self.max_jaw_z_abs = float(param(self, "max_jaw_z_abs", 0.35))
        self.min_grasp_width_m = float(param(self, "min_grasp_width_m", 0.005))
        self.max_grasp_width_m = float(param(self, "max_grasp_width_m", 0.061))

        # 超时与状态周期
        self.result_timeout_sec = float(param(self, "result_timeout_sec", 8.0))
        self.compute_timeout_sec = float(param(self, "compute_timeout_sec", 120.0))
        self.control_period_sec = float(param(self, "control_period_sec", 0.5))
        self.action_delay = float(param(self, "action_delay", 0.5))

        # GraspNet 姿态到末端执行器姿态的修正角度（RPY 度）
        self.graspnet_to_ee_rpy_deg = _float_list(
            param(self, "graspnet_to_ee_rpy_deg", [90.0, 0.0, 90.0]),
            [90.0, 0.0, 90.0],
        )

        # Pre-grasp pose 配置。
        self.pregrasp_pose_cfg = {
            "x": float(param(self, "pregrasp_pose.x", 0.150)),
            "y": float(param(self, "pregrasp_pose.y", 0.3)),
            "z": float(param(self, "pregrasp_pose.z", 0.35)),
            "roll": float(param(self, "pregrasp_pose.roll", 0.0)),
            "pitch": float(param(self, "pregrasp_pose.pitch", -180.0)),
            "yaw": float(param(self, "pregrasp_pose.yaw", 0.0)),
        }

        # J2 关节约束（通常用于保护姿态）
        self.j2_constraint = {
            "joint_positions": [float(param(self, "j2_constraint_position", -1.5708))],
            "joint_names": ["j2"],
            "tolerance": float(param(self, "j2_constraint_tolerance", 1.5708)),
            "weight": 1.0,
        }

    # ═══════════════════════════════════════════════════════
    #  MoveIt 客户端初始化
    # ═══════════════════════════════════════════════════════

    def _setup_moveit(self):
        """创建 Fairino 和 KDL 两个手臂 MoveIt 客户端以及夹爪客户端。"""
        self.moveit2_arm_fairino = self._make_arm_client(self.move_group_ns_fairino)
        self.moveit2_arm_kdl = self._make_arm_client(self.move_group_ns_kdl)

        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            self._configure_arm_planner(arm)

        # 统一设置运动参数
        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            arm.max_step_size = self.max_step_size
            arm.max_velocity = self.arm_max_velocity
            arm.max_acceleration = self.arm_max_acceleration
            arm.allowed_planning_time = self.allowed_planning_time
            arm.position_tolerance = self.position_tolerance
            arm.orientation_tolerance = self.orientation_tolerance
            arm.allowed_start_tolerance = self.allowed_start_tolerance

        # 默认使用的客户端
        self.moveit2_arm = self.moveit2_arm_fairino if self.ik_plugin == "fairino" else self.moveit2_arm_kdl

        # 关节轨迹执行动作客户端
        self.arm_execute_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/robot_arm_controller/follow_joint_trajectory",
            callback_group=self.callback_group,
        )
        self.gripper_execute_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/hand_controller/follow_joint_trajectory",
            callback_group=self.callback_group,
        )

        # 夹爪 MoveIt 客户端（仅用于运动学规划夹爪开合，实际执行使用 FollowJointTrajectory）
        self.moveit2_gripper = MoveIt2(
            node=self,
            joint_names=["finger1_joint", "finger2_joint"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.hand_group_name,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns_fairino,
        )
        self.moveit2_gripper.pipeline_id = "ompl"
        self.moveit2_gripper.planner_id = ""

    def _configure_arm_planner(self, arm):
        arm.pipeline_id = self.planning_pipeline_id
        arm.planner_id = self.planner_id

    def _motion_limits_kwargs(self) -> dict:
        return {
            "max_step_size": self.max_step_size,
            "allowed_planning_time": self.allowed_planning_time,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "allowed_start_tolerance": self.allowed_start_tolerance,
        }

    def _make_arm_client(self, namespace: str):
        """创建一个手臂 MoveIt 客户端。"""
        return MoveIt2(
            node=self,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.arm_group_name,
            ignore_new_calls_while_executing=False,
            callback_group=self.callback_group,
            move_group_namespace=namespace,
        )

    # ═══════════════════════════════════════════════════════
    #  话题回调
    # ═══════════════════════════════════════════════════════

    def _on_poses(self, msg: PoseArray):
        with self._result_lock:
            self._latest_poses = msg
            self._result_seq += 1

    def _on_scores(self, msg: Float32MultiArray):
        with self._result_lock:
            self._latest_scores = [float(v) for v in msg.data]

    def _on_metadata(self, msg: Float32MultiArray):
        with self._result_lock:
            self._latest_metadata = [float(v) for v in msg.data]

    def _on_preview_pose(self, msg: PoseStamped):
        with self._preview_lock:
            self._latest_preview_pose = msg
        self._maybe_publish_preview_plan_6d()

    def _on_preview_score(self, msg: Float32):
        with self._preview_lock:
            self._latest_preview_score = float(msg.data)
        self._maybe_publish_preview_plan_6d()

    def _on_motion_command(self, msg: String):
        if (
            self.active_mode == "graspnet"
            and str(msg.data).strip().lower() == "g"
            and self.current_state == GraspState.WAIT_G
        ):
            self._g_requested = True

    def _on_manual_abort(self, msg: Bool):
        if self.active_mode == "graspnet":
            self.abort.on_manual_abort(msg)

    def _on_active_mode(self, msg: String):
        mode = str(msg.data).strip().lower()
        if mode not in ("yolo", "graspnet") or mode == self.active_mode:
            return
        if mode == "yolo" and self.current_state != GraspState.WAIT_G:
            now = time.monotonic()
            if now - self._last_mode_exit_log_time >= 5.0:
                self.get_logger().info("Ignoring GraspNet mode exit outside WAIT_G.")
                self._last_mode_exit_log_time = now
            return
        self.active_mode = mode
        self._reset_task_cache()
        if mode == "graspnet":
            self._set_state(GraspState.WAIT_READY)

    def _maybe_publish_preview_plan_6d(self):
        """
        当收到新的预览姿态和评分后，生成并发布对应的 6D 规划文本。
        通过对比 header 时间与帧 ID 的键值避免重复处理。
        """
        with self._preview_lock:
            pose_msg = self._latest_preview_pose
            score = self._latest_preview_score
            if pose_msg is None or score is None:
                return
            key = (
                int(pose_msg.header.stamp.sec),
                int(pose_msg.header.stamp.nanosec),
                pose_msg.header.frame_id,
            )
            if key == self._last_preview_plan_key:
                return

        candidate = GraspCandidate(idx=-1, camera_pose=pose_msg.pose, score=score)
        if not self.transform_candidate(candidate, pose_msg.header):
            return
        self.prepare_grasp_pose(candidate)
        self._publish_grasp_plan_6d(candidate, label="Preview Grasp plan")

        with self._preview_lock:
            self._last_preview_plan_key = key

    # ═══════════════════════════════════════════════════════
    #  候选构建与选择
    # ═══════════════════════════════════════════════════════

    def _build_candidates(
        self,
        msg: PoseArray,
        scores: list[float],
        metadata: list[float],
    ) -> list[GraspCandidate]:
        """根据收到的抓取数组构建候选列表，最多 max_grasp_candidates 个。"""
        return build_candidates(msg.poses, scores, metadata, self.max_grasp_candidates)

    def _select_executable_candidate(self) -> Optional[GraspCandidate]:
        """遍历候选列表，返回第一个经过变换、准备、验证和规划预检成功的候选。"""
        if self._grasp_msg is None:
            return None
        rejected: dict[str, int] = {}

        def record_rejection(candidate: GraspCandidate):
            reason = candidate.reject_reason or "unknown"
            category = reason.split(":", 1)[0]
            rejected[category] = rejected.get(category, 0) + 1

        for candidate in self._candidates:
            if candidate.reject_reason:
                record_rejection(candidate)
                continue
            if not self.transform_candidate(candidate, self._grasp_msg.header):
                record_rejection(candidate)
                continue
            self.prepare_grasp_pose(candidate)
            if not self.validate_candidate(candidate):
                record_rejection(candidate)
                continue
            if not self.plan_candidate(candidate):
                record_rejection(candidate)
                continue
            self.get_logger().info(
                "GraspNet candidate selection: "
                f"received={len(self._candidates)} selected_idx={candidate.idx} "
                f"rejected={rejected}"
            )
            return candidate
        self.get_logger().warn(
            "GraspNet candidate selection: "
            f"received={len(self._candidates)} selected_idx=none rejected={rejected}"
        )
        return None

    def transform_candidate(self, candidate: GraspCandidate, header) -> bool:
        """将候选的相机坐标系姿态变换到基座坐标系。"""
        base_pose = self._camera_pose_to_base(header, candidate.camera_pose)
        if base_pose is None:
            self._reject_candidate(candidate, "tf_at_capture_time_unavailable")
            return False
        candidate.base_pose = base_pose
        return True

    def prepare_grasp_pose(self, candidate: GraspCandidate):
        """
        根据 GraspNet depth 和 approach 轴生成最终抓取姿态、提升姿态和夹爪位置。
        """
        return prepare_candidate(
            candidate,
            grasp_offset_m=self.grasp_offset_m,
            orientation_rpy_deg=self.graspnet_to_ee_rpy_deg,
            approach_distance_m=self.approach_distance_m,
            lift_distance_m=self.lift_distance,
        )

    def validate_candidate(self, candidate: GraspCandidate) -> bool:
        """拒绝危险姿态；不修改 GraspNet 输出的抓取姿态。"""
        reason = candidate_geometry_rejection(
            candidate,
            min_width_m=self.min_grasp_width_m,
            max_width_m=self.max_grasp_width_m,
            max_approach_tilt_deg=self.max_approach_tilt_deg,
            max_jaw_z_abs=self.max_jaw_z_abs,
        )
        if reason:
            self._reject_candidate(candidate, reason)
            return False
        return True

    def plan_candidate(self, candidate: GraspCandidate) -> bool:
        """预检三个关键阶段的运动规划是否可行。"""
        checks = [
            (candidate.approach, False, "Plan GraspNet approach", 0.20, 0.20),
            (candidate.grasp, False, "Plan GraspNet grasp", 0.20, 0.20),
            (candidate.lift, False, "Plan GraspNet lift", 0.12, 0.12),
        ]
        for pose, cartesian, action_name, velocity, acceleration in checks:
            if not self._can_plan_pose(pose, cartesian, action_name, velocity, acceleration):
                self._reject_candidate(candidate, f"plan_failed:{action_name}")
                return False
        return True

    def _require_candidate(self) -> Optional[GraspCandidate]:
        """获取当前活跃候选，若为空则强制进入 PLAN 状态。"""
        if self._active_candidate is None:
            self._set_state(GraspState.PLAN)
        return self._active_candidate

    def _can_plan_pose(
        self,
        pose: Pose,
        cartesian: bool,
        action_name: str,
        max_velocity: float,
        max_acceleration: float,
    ) -> bool:
        """
        尝试规划指定位姿，仅检查可行性不执行。
        针对 Fairino 的笛卡尔规划使用专用方法。
        """
        arm = self.motion._select_arm(self.ik_plugin)
        target = self.pose_tools.to_pose_stamped(pose)
        try:
            arm.max_velocity = float(max_velocity)
            arm.max_acceleration = float(max_acceleration)
            arm.max_step_size = self.max_step_size
            arm.allowed_planning_time = self.allowed_planning_time
            arm.position_tolerance = self.position_tolerance
            arm.orientation_tolerance = self.orientation_tolerance
            arm.allowed_start_tolerance = self.allowed_start_tolerance
            arm.clear_path_constraints()
            arm.set_path_joint_constraint(
                joint_positions=self.j2_constraint["joint_positions"],
                joint_names=self.j2_constraint["joint_names"],
                tolerance=self.j2_constraint.get("tolerance", 0.0),
                weight=self.j2_constraint.get("weight", 1.0),
            )
            pipeline_id = PlannerSwitch.normalize_pipeline(getattr(arm, "pipeline_id", "ompl"))
            if cartesian and pipeline_id == "fairino":
                plan = self.motion._plan_fairino_cartesian(
                    arm=arm,
                    target_pose=target,
                    action_name=action_name,
                    fraction_threshold=0.98,
                )
            else:
                plan = arm.plan(
                    target,
                    cartesian=cartesian,
                    cartesian_fraction_threshold=0.98 if cartesian else 0.0,
                )
            return bool(plan)
        except Exception as exc:
            self.get_logger().warn(f"{action_name}: plan precheck failed: {exc}")
            return False

    # ═══════════════════════════════════════════════════════
    #  服务与结果等待
    # ═══════════════════════════════════════════════════════

    def _call_compute_service(self) -> bool:
        req = Trigger.Request()
        future = self.compute_client.call_async(req)
        deadline = time.time() + self.compute_timeout_sec
        while rclpy.ok() and not future.done():
            if self.abort.is_set() or time.time() >= deadline:
                self._last_compute_error = "GraspNet compute cancelled or timed out"
                return False
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            self._last_compute_error = "GraspNet compute service returned no response"
            return False
        result = future.result()
        if not result.success:
            self._last_compute_error = str(result.message)
            log = (
                self.get_logger().info
                if self._last_compute_error.startswith("CANCELED:")
                else self.get_logger().error
            )
            log(f"/grasp/compute failed: {self._last_compute_error}")
            return False
        self._last_compute_error = ""
        return bool(result.success)

    def _wait_for_result(self, start_seq: int) -> tuple[Optional[PoseArray], list[float], list[float]]:
        deadline = time.time() + self.result_timeout_sec
        while rclpy.ok() and time.time() < deadline:
            with self._result_lock:
                if self._latest_poses is not None and self._result_seq > start_seq:
                    return self._latest_poses, list(self._latest_scores), list(self._latest_metadata)
            time.sleep(0.05)
        return None, [], []

    # ═══════════════════════════════════════════════════════
    #  TF 与姿态变换
    # ═══════════════════════════════════════════════════════

    def _camera_pose_to_base(self, header, pose: Pose) -> Optional[Pose]:
        frame_id = header.frame_id or self.camera_frame
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = header.stamp
        ps.pose = pose
        stamped_time = rclpy.time.Time.from_msg(header.stamp)
        try:
            tf = self._tf_buffer.lookup_transform(
                self.base_frame,
                frame_id,
                stamped_time,
                timeout=Duration(seconds=0.5),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"tf_at_capture_time_unavailable: {self.base_frame} <- {frame_id} at "
                f"{header.stamp.sec}.{header.stamp.nanosec:09d}: {exc}"
            )
            return None
        return tf2_geometry_msgs.do_transform_pose_stamped(ps, tf).pose

    # ═══════════════════════════════════════════════════════
    #  消息发布
    # ═══════════════════════════════════════════════════════

    def _publish_target(self, pose: Pose):
        msg = self.pose_tools.to_pose_stamped(pose)
        self.target_pub.publish(msg)

    def _publish_selected_grasp_6d(self, candidate: GraspCandidate):
        text = (
            f"Selected GraspNet grasp idx={candidate.idx} {self._candidate_meta_text(candidate)} "
            f"frame={self.base_frame} raw_graspnet {self._pose_6d_text(candidate.base_pose)} "
            f"target_grasp {self._pose_6d_text(candidate.grasp)}"
        )
        msg = String()
        msg.data = text
        self.selected_grasp_6d_pub.publish(msg)
        self.get_logger().info(text)

    def _publish_grasp_plan_6d(
        self,
        candidate: GraspCandidate,
        label: str = "Grasp plan",
    ):
        idx_text = "" if candidate.idx < 0 else f" idx={candidate.idx}"
        lines = [f"{label}{idx_text} {self._candidate_meta_text(candidate)} frame={self.base_frame}"]
        if candidate.base_pose is not None:
            lines.append(f"  raw_graspnet {self._pose_6d_text(candidate.base_pose)}")
        if candidate.approach is not None:
            lines.append(f"  target_approach {self._pose_6d_text(candidate.approach)}")
        lines.append(f"  target_grasp {self._pose_6d_text(candidate.grasp)}")
        lines.append(f"  target_lift  {self._pose_6d_text(candidate.lift)}")
        if candidate.preopen_positions is not None:
            lines.append(
                "  gripper_preopen=("
                f"{candidate.preopen_positions[0]:.4f},{candidate.preopen_positions[1]:.4f})"
            )
        lines.append(
            "  gripper_close=("
            f"{self.gripper_close_positions[0]:.4f},{self.gripper_close_positions[1]:.4f})"
        )
        text = "\n".join(lines)
        msg = String()
        msg.data = text
        self.grasp_plan_6d_pub.publish(msg)
        self.get_logger().info(text)

    def _reject_candidate(self, candidate: GraspCandidate, reason: str):
        candidate.reject_reason = reason
        text = f"Reject GraspNet grasp idx={candidate.idx} {self._candidate_meta_text(candidate)} reason={reason}"
        msg = String()
        msg.data = text
        self.rejected_grasp_pub.publish(msg)
        self.get_logger().warn(text)

    # ═══════════════════════════════════════════════════════
    #  辅助文本格式化
    # ═══════════════════════════════════════════════════════

    def _candidate_meta_text(self, candidate: GraspCandidate) -> str:
        score = "n/a" if candidate.score is None else f"{candidate.score:.4f}"
        width = "n/a" if candidate.width_m is None else f"{candidate.width_m:.4f}m"
        depth = "n/a" if candidate.depth_m is None else f"{candidate.depth_m:.4f}m"
        return f"score={score} width={width} depth={depth}"

    def _pose_6d_text(self, pose: Pose) -> str:
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rpy = R.from_quat(quat).as_euler("xyz", degrees=True)
        return (
            f"xyz=({pose.position.x:.4f},{pose.position.y:.4f},{pose.position.z:.4f}) "
            f"rpy_deg=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}) "
            "quat_xyzw=("
            f"{pose.orientation.x:.6f},{pose.orientation.y:.6f},"
            f"{pose.orientation.z:.6f},{pose.orientation.w:.6f})"
        )

    def _build_pregrasp_pose(self):
        return self.pose_tools.make_pose(
            self.pregrasp_pose_cfg["x"],
            self.pregrasp_pose_cfg["y"],
            self.pregrasp_pose_cfg["z"],
            self.pregrasp_pose_cfg["roll"],
            self.pregrasp_pose_cfg["pitch"],
            self.pregrasp_pose_cfg["yaw"],
        )

    # ═══════════════════════════════════════════════════════
    #  系统就绪检查
    # ═══════════════════════════════════════════════════════

    def _move_to_pregrasp_pose(self) -> bool:
        planning_client = self.startup_client()
        return self.motion.move_to_pose(
            self.pregrasp_pose,
            action_name=f"Move to pre-grasp pose [client={planning_client}]",
            max_velocity=0.30,
            max_acceleration=0.30,
            planning_client=planning_client,
            cartesian=False,
            joint_constraint=False,
            timeout_sec=180.0,
            **self._motion_limits_kwargs(),
        )

    def _close_gripper_at_pregrasp(self) -> bool:
        return self.motion.control_gripper(
            open_gripper=False,
            positions=self.gripper_close_positions,
            action_name="Close gripper at pre-grasp",
            timeout_sec=90.0,
        )

    def startup_motion_ready(self, timeout_sec: Optional[float] = None, planning_client: str | None = None) -> bool:
        timeout = self.move_group_ready_timeout_sec if timeout_sec is None else float(timeout_sec)
        arm = self.motion._select_arm(planning_client)
        ready = (
            self._moveit_service_ready(arm, timeout)
            and self._moveit_service_ready(self.moveit2_gripper, timeout)
            and self._action_ready(self.arm_execute_action, timeout)
            and self._action_ready(self.gripper_execute_action, timeout)
            and self._controllers_active(("robot_arm_controller", "hand_controller"), timeout)
        )
        if ready and not self._startup_ready_logged:
            self.get_logger().info("MoveIt services ready for GraspNet visual grasping")
            self._startup_ready_logged = True
        return ready

    def startup_client(self) -> str:
        requested = PlannerSwitch.normalize_ik(self.ik_plugin)
        if self.startup_motion_ready(planning_client=requested):
            return requested
        for candidate in ("fairino", "kdl"):
            if candidate != requested and self.startup_motion_ready(planning_client=candidate):
                return candidate
        return requested

    def _tf_ready(self) -> bool:
        try:
            self._tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())
            return True
        except Exception:
            self._publish_state("waiting_tf")
            return False

    def _moveit_service_ready(self, moveit_obj, timeout_sec: float) -> bool:
        client = getattr(moveit_obj, "_plan_kinematic_path_service", None)
        if client is None:
            return True
        try:
            return bool(client.wait_for_service(timeout_sec=float(timeout_sec)))
        except Exception:
            return False

    def _action_ready(self, action_client, timeout_sec: float) -> bool:
        try:
            return bool(action_client.wait_for_server(timeout_sec=float(timeout_sec)))
        except Exception:
            return False

    def _controllers_active(self, names: tuple[str, ...], timeout_sec: float) -> bool:
        try:
            if not self._controller_manager_client.wait_for_service(timeout_sec=float(timeout_sec)):
                return False
            future = self._controller_manager_client.call_async(ListControllers.Request())
            deadline = time.time() + float(timeout_sec)
            while rclpy.ok() and not future.done():
                if time.time() >= deadline:
                    return False
                time.sleep(0.02)
            if not future.done() or future.result() is None:
                return False
            states = {controller.name: controller.state for controller in future.result().controller}
            return all(states.get(name) == "active" for name in names)
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════
    #  状态与恢复
    # ═══════════════════════════════════════════════════════

    def _publish_state(self, state: GraspState | str):
        msg = String()
        msg.data = state.value if isinstance(state, GraspState) else str(state)
        self.state_pub.publish(msg)

    def _set_state(self, state: GraspState | str):
        previous_state = self.current_state
        try:
            self.current_state = state if isinstance(state, GraspState) else GraspState(state)
        except ValueError:
            self.current_state = str(state)
        self._publish_state(self.current_state)
        if self.current_state == GraspState.WAIT_G and previous_state != GraspState.WAIT_G:
            self.get_logger().info(
                "GraspNet ready: 已到达 pregrasp，夹爪已闭合。在控制终端输入 g 开始计算并执行一次 "
                "GraspNet 抓取；空格立即停止，h 回到 pregrasp，r 在安全后解除停止。"
            )

    def _reset_task_cache(self):
        self._start_seq = 0
        self._g_requested = False
        self._grasp_msg = None
        self._grasp_scores = []
        self._grasp_metadata = []
        self._candidates = []
        self._active_candidate = None

    def _recovery_complete(self, recovered: bool):
        if recovered:
            self._set_state(GraspState.WAIT_READY)
        else:
            self._fail(GraspState.RECOVERY_PREGRASP_FAILED)

    def _fail(self, state: GraspState | str):
        self.get_logger().error(f"GraspNet visual grasping failed: {state}")
        self._set_state(state)


def main(args=None):
    rclpy.init(args=args)
    node = GraspnetVisualGraspingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
