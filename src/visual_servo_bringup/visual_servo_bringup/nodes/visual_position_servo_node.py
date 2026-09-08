#!/usr/bin/env python3
import time
import threading
import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String, Bool
from pymoveit2 import MoveIt2
from myrobot_common.perception.detection_cache import DetectionCache, DetectionSubscribers
from myrobot_common.planning.motion_executor import MoveItMotion, PlanScoreConfig, PlannerSwitch
from myrobot_common.planning.trajectory_scoring import select_best_path
from myrobot_common.task.abort_manager import AbortManager
from myrobot_common.utils.params import param, param_b, param_f
from myrobot_common.utils.pose_tools import PoseTools
from myrobot_common.utils.tf_tools import TfTools
from visual_servo_bringup.utils.debug_publishers import Publishers
from visual_servo_bringup.task.position_servo_state_machine import PositionServoStateMachine
from visual_servo_bringup.task.task_types import TargetType, TaskState

from visual_servo_bringup.servo.servo_controller import ServoController
from visual_servo_bringup.controllers.pid_controller import ServoControlConfig
from visual_servo_bringup.servo.servo_io import ServoIO
from visual_servo_bringup.servo.visual_servo_params import ServoRuntimeConfig
    
class VisualPositionServoNode(Node):
    # ↑ 主节点：继承自 rclpy.node.Node
    def __init__(self):
        # ↑ 构造函数：节点启动时执行
        super().__init__("visual_position_servo_node")
        # ↑ 初始化通用视觉伺服抓取节点；当前实现为位置伺服。

        self.callback_group = ReentrantCallbackGroup()# ↑ 可重入回调组：允许并发执行
        self.control_cb_group = MutuallyExclusiveCallbackGroup() # ↑ 控制回调组：control_loop 不希望并发重入，避免状态机被同时执行两次
        self.abort_cb_group = MutuallyExclusiveCallbackGroup()#  ↑ abort 回调组：abort 的处理也希望互斥，避免并发更新状态

        self.state_lock = threading.Lock()# ↑ 状态锁：用于保护 current_state 等共享变量，避免多线程竞争
        
        # ===== DEBUG switches =====
        self.declare_parameter("dbg", True)#是否开启调试输出
        self.declare_parameter("dbg_throttle_sec", 1.0)#默认 1 秒 最多打印一次
        self.dbg = bool(self.get_parameter("dbg").value)
        self.dbg_throttle_sec = float(self.get_parameter("dbg_throttle_sec").value)
        self._dbg_last = {}

        # --- Target preference ---
        self.declare_parameter("preferred_target", "cube")
        self.preferred_target = str(self.get_parameter("preferred_target").value).lower().strip()
        # --- Target preference ---

        # --- Frames ---
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("ee_frame", "tool0")
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        # --- Frames ---

        # --- tf树和位姿态工具 ---
        self.tf_tools = TfTools(self, base_frame=self.base_frame, camera_frame=self.camera_frame) 
         # ↑ TF 工具：负责相机点转 base 点
        self.pose_tools = PoseTools(self, base_frame=self.base_frame)
         # ↑ Pose 工具：生成姿态 Pose/PoseStamped 
        # --- tf树和位姿态工具 ---

        # --- Init subsystems ---
        self.det_cache = DetectionCache()
        # ↑ 创建检测缓存对象：只负责存变量，不做订阅
        self.det_subs = DetectionSubscribers(self, self.det_cache)
        # ↑ 创建订阅器：订阅 elongated_object/cube/box 的位置和 RPY topic
        self.setup_moveit()
        # ↑ 初始化 MoveIt2 arm + gripper + 约束参数
        self.setup_params()
        # ↑ 初始化“业务参数”（home_pose、profiles、selector 等）
        self.aruco_position = None
        self._aruco_pose_received_monotonic = None
        aruco_latest_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PoseStamped,
            self.aruco_marker_pose_topic,
            self._on_aruco_marker_pose,
            aruco_latest_qos,
            callback_group=self.callback_group,
        )
        self.aruco_motion_start_pub = self.create_publisher(
            Bool, self.aruco_motion_auto_start_topic, 10
        )
        self.moveit2_arm = self.moveit2_arm_kdl if self.ik_plugin == "kdl" else self.moveit2_arm_fairino
        self.setup_servo()
        # ↑ 初始化伺服参数 + ServoIO（topic/service/tf）
        # --- Init subsystems ---

        self.messages_publishers =Publishers(self, servo_controller_type=self.servo_controller_type)
        # 创建调试发布器

        # --- Abort manager ---
        self.abort = AbortManager(self, arm=self.moveit2_arm, gripper=self.moveit2_gripper)
        # ↑ AbortManager：统一中止/恢复（停止动作、开夹爪、回 home 等）
        self.create_subscription(Bool, "/manual_abort", self.abort.on_manual_abort, 10, callback_group=self.abort_cb_group)
        # ↑ 订阅手动中止 topic：/manual_abort True 时触发 abort
        # --- Abort manager ---

        self.TaskState = TaskState

        self.motion = MoveItMotion(
            node=self,
            arm=self.moveit2_arm_fairino,
            arm_clients={"fairino": self.moveit2_arm_fairino,"kdl": self.moveit2_arm_kdl,},
            default_client=self.ik_plugin,
            gripper=self.moveit2_gripper,
            pose_tools=self.pose_tools,
            abort=self.abort,
            select_best_path=select_best_path,
            score_cfg=PlanScoreConfig(
                num_candidates=self.NUM_CANDIDATE_PLANS,
                wrist_weight=self.WRIST_WEIGHT,
                wrist_joint_indices=self.WRIST_JOINT_INDICES,
            ),
            action_delay=self.action_delay,
        )
        # ↑ MoveItMotion 封装：统一的 move_to_pose / move_to_joints / control_gripper

        self.servo_controller = ServoController(self, self.servo_io)
        # ↑ 创建伺服控制器：传入 node
        
        # --- Detection caches ---
      
        self.target_above_pose = None  # ↑ 存储“目标上方 Pose”：SEARCHING 时算出来，MOVING_TO_TARGET_ABOVE 使用

        # --- Task state ---
        self.current_state = TaskState.IDLE  # ↑ 初始状态：IDLE

        # ↑ 当前锁定的跟踪目标类型
        self.active_target: TargetType | None = None
        self.state_machine = PositionServoStateMachine(self)

        self.state_publisher = self.create_publisher(String, "/task_state", 10)
        # ↑ 发布当前任务状态到 /task_state

        self.create_timer(0.2, self.control_loop, callback_group=self.control_cb_group)  # state machine (5 Hz)
        # ↑ 主状态机 timer：5Hz，处理 SEARCHING/MOVING/... 状态跳转
        self.position_servo_rate_hz = param_f(self, "position_servo_rate_hz", 250.0)
        if self.position_servo_rate_hz <= 0.0:
            raise ValueError("position_servo_rate_hz must be positive.")
        self.create_timer(
            1.0 / self.position_servo_rate_hz,
            self.servo_tick,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            "✓ VisualPositionServoNode initialized"
        )  # ↑ 节点初始化完成日志

    def dbg_throttle(self, key: str, sec: float | None = None) -> bool:
        # ↑ 调试节流函数：防止日志刷屏
        if not self.dbg:
            # ↑ dbg=false 时完全不打印
            return False
        if sec is None:
            # ↑ 如果没传节流时间，用默认参数 dbg_throttle_sec
            sec = self.dbg_throttle_sec
        now = time.time()
        # ↑ 当前 wall-time 秒
        last = self._dbg_last.get(key, 0.0)
        if (now - last) >= sec:
            # ↑ 如果距离上次打印已经超过阈值，允许打印
            self._dbg_last[key] = now
            return True
        return False
    # ---------------- 初始化 MoveIt2 的 arm 和 gripper ----------------
    def setup_moveit(self):
        arm_group_name = str(param(self, "arm_group_name", "robot_arm"))
        hand_group_name = str(param(self, "hand_group_name", "hand"))
        self.move_group_ns_fairino = str(param(self, "move_group_ns_fairino", "/move_group_fairino"))
        self.move_group_ns_kdl = str(param(self, "move_group_ns_kdl", "/move_group_kdl"))
        self.moveit2_arm_fairino = MoveIt2(
            node=self,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=arm_group_name,
            # use_move_group_action=True,  # 使用Move Group Action
            ignore_new_calls_while_executing=False,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns_fairino,
        )
        self.moveit2_arm_kdl = MoveIt2(
            node=self,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=arm_group_name,
            ignore_new_calls_while_executing=False,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns_kdl,
        )
        # 关键：禁用重定时
        self.moveit2_arm_fairino.retime_cartesian = True
        self.moveit2_arm_kdl.retime_cartesian = True

        self.allowed_start_tolerance = param_f(self, "allowed_start_tolerance", 0.1)

        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(
            str(param(self, "planning_pipeline_id", "fairino"))
        )
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            str(param(self, "planner_id", "birrt*")),
        )
        if not PlannerSwitch.is_valid(self.planning_pipeline_id, self.planner_id):
            raise ValueError(
                f"Unsupported planner config: pipeline={self.planning_pipeline_id}, "
                f"planner={self.planner_id}"
            )

        self.max_step_size = param_f(self, "max_step_size", 0.05)
        self.arm_max_velocity = param_f(self, "arm_max_velocity", 0.2)
        self.arm_max_acceleration = param_f(self, "arm_max_acceleration", 0.2)
        self.allowed_planning_time = param_f(self, "allowed_planning_time", 30.0)
        self.position_tolerance = param_f(self, "position_tolerance", 0.005)
        self.orientation_tolerance = param_f(self, "orientation_tolerance", 0.005)

        # self.moveit2_arm.planner_id = "RRTConnectFast"
        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            arm.pipeline_id = self.planning_pipeline_id
            arm.planner_id = self.planner_id
            arm.max_step_size = self.max_step_size
            arm.max_velocity = self.arm_max_velocity
            arm.max_acceleration = self.arm_max_acceleration
            arm.allowed_planning_time = self.allowed_planning_time
            arm.position_tolerance = self.position_tolerance
            arm.orientation_tolerance = self.orientation_tolerance
            arm.allowed_start_tolerance = self.allowed_start_tolerance
        # Backward compatibility for modules that still reference moveit2_arm.
        self.moveit2_arm = self.moveit2_arm_fairino

        self.moveit2_gripper = MoveIt2(
            node=self,
            joint_names=["finger1_joint", "finger2_joint"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=hand_group_name,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns_fairino,
        )

        self.moveit2_gripper.pipeline_id = "ompl"
        self.moveit2_gripper.planner_id = ""
        
        self.j2_constraint = {
            "joint_positions": [param_f(self, "j2_constraint_position", -1.5708)],
            "joint_names": ["j2"],
            "tolerance": param_f(self, "j2_constraint_tolerance", 1.5708),
            "weight": 1.0,
        }
        #↑ 关节约束：限制 j2
        self.get_logger().info(
            f"✓ MoveIt2 initialized: pipeline={self.planning_pipeline_id}, planner={self.planner_id}"
        )

    def motion_limits_kwargs(self) -> dict:
        return {
            "max_step_size": self.max_step_size,
            "allowed_planning_time": self.allowed_planning_time,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "allowed_start_tolerance": self.allowed_start_tolerance,
        }
    # ---------------- 初始化 MoveIt2 的 arm 和 gripper ----------------

    # ---------------- 初始化伺服：参数 + 创建 ServoIO ----------------
    def setup_servo(self):     
        self.declare_parameter("servo_ns", "/servo_node") # ↑ servo 命名空间参数
        servo_ns = str(self.get_parameter("servo_ns").value).rstrip("/")

        self.servo_runtime_cfg = ServoRuntimeConfig.from_node(self)
        self.servo_controller_type = self.servo_runtime_cfg.servo_controller_type
        self.servo_controller_family = self.servo_runtime_cfg.servo_controller_family
        self.pid_variant = self.servo_runtime_cfg.pid_variant
        self.align_xyz_tol = self.servo_runtime_cfg.servo_align_xyz_tol
        self.control_config = ServoControlConfig.from_runtime(self.servo_runtime_cfg)

        # ===== 创建 ServoIO（搬迁 I/O）=====
        self.servo_io = ServoIO(self, base_frame=self.base_frame, ee_frame=self.ee_frame, servo_ns=servo_ns)
        # ↑ 创建 ServoIO：内部创建 publisher/subscriber/service client/tf listener 等
    # ---------------- 初始化伺服：参数 + 创建 ServoIO ----------------

    # ---------------- 初始化任务相关参数 ----------------
    def setup_params(self):
        self.above_offset = param_f(self, "above_offset", 0.12)
        def pose_values(name: str, default: list[float]) -> list[float]:
            try:
                values = [float(value) for value in param(self, name, default)]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain three finite values.") from exc
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values.")
            return values

        home_xyz = pose_values("home_pose.xyz", [0.0, 0.0, 0.0])
        home_rpy_deg = pose_values("home_pose.rpy_deg", [0.0, 0.0, 0.0])
        self.home_pose = self.pose_tools.make_pose(*home_xyz, *home_rpy_deg)
        self.open_gripper_after_home = param_b(self, "open_gripper_after_home", False)
        self.action_delay = param_f(self, "action_delay", 0.5)
        self.detection_timeout = param_f(self, "detection_timeout", 3.0)
        self.perception_source = str(param(self, "perception_source", "yolo_kalman")).strip().lower()
        if self.perception_source not in {"yolo_kalman", "aruco"}:
            raise ValueError(f"Unsupported perception_source: {self.perception_source}")
        self.aruco_marker_pose_topic = str(param(self, "aruco_marker_pose_topic", "/aruco_marker/pose"))
        self.aruco_prediction_hold_sec = param_f(self, "aruco_prediction_hold_sec", 0.25)
        if self.aruco_prediction_hold_sec < 0.0:
            raise ValueError("aruco_prediction_hold_sec must be non-negative.")
        self.aruco_motion_auto_start_topic = str(
            param(self, "aruco_motion_auto_start_topic", "/aruco_marker_auto_start")
        )
        self.target_above_rpy_deg = [float(value) for value in param(
            self, "target_above_rpy_deg", [-45.0, -180.0, 0.0]
        )]
        if len(self.target_above_rpy_deg) != 3:
            raise ValueError("target_above_rpy_deg must contain [roll, pitch, yaw].")
        self.NUM_CANDIDATE_PLANS = int(param(self, "num_candidate_plans", 5))
        self.WRIST_WEIGHT = param_f(self, "wrist_weight", 50.0)
        self.WRIST_JOINT_INDICES = tuple(int(value) for value in param(self, "wrist_joint_indices", [2, 3, 4]))
        priority = [str(value).lower().strip() for value in param(
            self, "target_priority", ["cube", "elongated_object", "box", "stone"]
        )]
        valid_targets = {target.value for target in TargetType}
        self.target_priority = [name for name in priority if name in valid_targets]
        if self.preferred_target in valid_targets:
            self.target_priority = [self.preferred_target] + [
                name for name in self.target_priority if name != self.preferred_target
            ]
        if not self.target_priority:
            self.target_priority = [target.value for target in TargetType]
        self.get_logger().info("✓ Params set: global plan -> target_above -> servo")
        self.ik_plugin = self._normalize_planning_client(str(param(self, "ik_plugin", "fairino")))
        self.allow_cross_client_fallback = param_b(self, "allow_cross_client_fallback", True)
        self.move_group_ready_timeout_sec = param_f(self, "move_group_ready_timeout_sec", 5.0)
    # ---------------- 初始化任务相关参数 ----------------

    # ---------- 重置运行时缓存：用于任务完成/错误恢复/abort 恢复 ----------
    def _reset_task_cache(self):
        self.active_target = None
        self.aruco_position = None
        self._aruco_pose_received_monotonic = None
        # ↑ 当前目标置空
        self.det_cache.reset()
        # ↑ 清空检测缓存（避免用旧消息）
        if hasattr(self, "servo_controller") and self.servo_controller is not None:
            self.servo_controller.reset()
            # ↑ 如果 servo_controller 已创建，调用 reset 清空其内部滤波器/缓存
    # ---------- 重置运行时缓存：用于任务完成/错误恢复/abort 恢复 ----------

    # ---------- 恢复 MoveIt arm 的速度/加速度限制 ----------
    def _restore_arm_limits(self):
        try:
            self.moveit2_arm.max_velocity = 0.2
            self.moveit2_arm.max_acceleration = 0.2
        except Exception:
            pass
    # ---------- 恢复 MoveIt arm 的速度/加速度限制 ----------
    
    
    def _normalize_planning_client(self, client: str) -> str:
        primary_norm = str(client).strip().lower()
        if primary_norm in ("fairino", "kdl"):
            return primary_norm
        self.get_logger().warn(
            f"Invalid ik_plugin/planning client '{client}', fallback to 'fairino'."
        )
        return "fairino"

    def _planning_client_order(self) -> list[str]:
        primary_norm = self._normalize_planning_client(self.ik_plugin)
        if not self.allow_cross_client_fallback:
            return [primary_norm]
        secondary = "kdl" if primary_norm == "fairino" else "fairino"
        return [primary_norm, secondary]

    def go_home(self, phase: str = "go_home") -> bool:
        for client in self._planning_client_order():
            self.get_logger().info(f"phase={phase} client={client}")
            if not self.motion.wait_client_ready(
                planning_client=client,
                timeout_sec=self.move_group_ready_timeout_sec,
            ):
                self.get_logger().warn(
                    f"move_group client not ready: client={client}, timeout={self.move_group_ready_timeout_sec:.1f}s"
                )
                continue
            if self.motion.move_to_pose(
                self.home_pose,
                action_name=f"Go HOME [client={client}]",
                timeout_sec=30.0,
                planning_client=client,
            ):
                return True
            self.get_logger().warn(f"Go HOME failed on client={client}")
        return False

    def open_gripper_after_home_action(self) -> bool:
        if not self.open_gripper_after_home:
            return True
        return self.motion.control_gripper(
            open_gripper=True,
            action_name="Open gripper after HOME",
            timeout_sec=10.0,
        )
    
    
    def _set_state(self, st: TaskState):
        with self.state_lock:
            previous = self.current_state
            self.current_state = st
        if previous != st:
            self.get_logger().info(f"State: {previous.value} -> {st.value}")

    def _get_state(self) -> TaskState:
        with self.state_lock:
            return self.current_state    
        
    def _target_is_fresh(self, msg) -> bool:
        if msg is None:
            return False
        if self.perception_source == "aruco":
            return self.aruco_pose_receipt_age_sec() <= self.detection_timeout
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if stamp_ns <= 0:
            return False
        age_sec = (self.get_clock().now().nanoseconds - stamp_ns) * 1e-9
        return 0.0 <= age_sec <= self.detection_timeout

    def _on_aruco_marker_pose(self, msg: PoseStamped) -> None:
        point = PointStamped()
        point.header = msg.header
        point.point = msg.pose.position
        self.aruco_position = point
        self._aruco_pose_received_monotonic = time.monotonic()

    def aruco_pose_receipt_age_sec(self) -> float:
        if self._aruco_pose_received_monotonic is None:
            return float("inf")
        return max(0.0, time.monotonic() - self._aruco_pose_received_monotonic)

    def start_target_motion(self) -> None:
        if self.active_target == TargetType.CUBE:
            self.messages_publishers.publish_cube_auto_start(True)
        elif self.active_target == TargetType.ARUCO:
            message = Bool()
            message.data = True
            self.aruco_motion_start_pub.publish(message)

    def select_tracking_target(self, keep_active: bool):
        if getattr(self, "perception_source", "yolo_kalman") == "aruco":
            if self._target_is_fresh(self.aruco_position):
                return TargetType.ARUCO, self.aruco_position
            return None, None
        if keep_active and self.active_target is not None:
            active_msg = self.det_cache.get_position(self.active_target)
            if self._target_is_fresh(active_msg):
                return self.active_target, active_msg
            return None, None

        for name in self.target_priority:
            target = TargetType(name)
            msg = self.det_cache.get_position(target)
            if self._target_is_fresh(msg):
                return target, msg
        return None, None

    # ----------------  伺服 timer 回调：高频运行 ----------------
    def servo_tick(self):
        if self.servo_controller is not None:
            self.servo_controller.tick()
    # ----------------  伺服 timer 回调：高频运行 ----------------

    # ---------------- 主状态机 timer 回调（低频 5Hz） ----------------
    def control_loop(self):
        return self.state_machine.tick()

    def _control_loop_impl(self):
        # Legacy compatibility hook; state logic now lives in task/position_servo_state_machine.py
        return self.state_machine.tick()

def main(args=None):
    rclpy.init(args=args)
    node = VisualPositionServoNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted.")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
