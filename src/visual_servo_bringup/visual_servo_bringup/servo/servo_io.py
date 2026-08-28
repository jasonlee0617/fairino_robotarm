from __future__ import annotations

import time
from collections import deque

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TwistStamped
from myrobot_common.utils.params import param
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64, Int8
from std_srvs.srv import Empty, Trigger
from trajectory_msgs.msg import JointTrajectory


class ServoIO:
    """MoveIt Servo topic/service/status boundary plus EE telemetry.

    The FK/Jacobian here is retained only for runtime telemetry and latency
    tracing. Servo singularity decisions remain owned by MoveIt Servo status.
    """

    _DH_D = np.array([0.140, 0.0, 0.0, 0.102, 0.102, 0.100])
    _DH_A = np.array([0.0, -0.280, -0.240, 0.0, 0.0, 0.0])
    _DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
    _TOOL_OFFSET = np.array([0.0, 0.0, 0.1168])
    _JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]

    def __init__(self, node, base_frame: str, ee_frame: str, servo_ns: str):
        self.node = node
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.servo_ns = str(param(self.node, "servo_ns", servo_ns or "/servo_node")).rstrip("/")

        self._joint_positions = None
        self._joint_velocities = None
        self._ee_twist = np.zeros(6, dtype=np.float64)
        self._ee_linear_speed = 0.0
        self._ee_angular_speed = 0.0
        self._ee_pose_position = np.zeros(3, dtype=np.float64)
        self._ee_pose_rotation = np.eye(3, dtype=np.float64)
        self._use_tool_offset = ee_frame == "tool0"

        self._last_collision_scale = None
        self._last_servo_out = None
        self._last_joint_state_time = None
        self._last_cmd_norm = 0.0
        self._last_cmd_vec = np.zeros(4, dtype=np.float64)
        self._last_ee_lin = np.zeros(3, dtype=np.float64)
        self._cmd_events = deque(maxlen=256)
        self._cmd_event_threshold = 0.008
        self._cmd_delta_threshold = 0.006
        self._ee_response_threshold = 0.003

        self._setup_ros_io()
        node.get_logger().info(
            f"✓ Servo I/O set: twist={self.servo_twist_topic}, status={self.servo_status_topic}"
        )

    def _setup_ros_io(self):
        node = self.node
        # QoS is selected here because ServoIO owns both Twist command output
        # and MoveIt Servo status input; controller code should not know DDS details.
        qos_mode = str(param(node, "servo_qos_mode", "modern")).lower()
        if qos_mode in ("legacy", "old", "v1"):
            # Legacy profile (old behavior in this project):
            # cmd: BEST_EFFORT + VOLATILE
            # status: RELIABLE + TRANSIENT_LOCAL
            cmd_reliability = ReliabilityPolicy.BEST_EFFORT
            cmd_durability = DurabilityPolicy.VOLATILE
            status_reliability = ReliabilityPolicy.RELIABLE
            status_durability = DurabilityPolicy.TRANSIENT_LOCAL
            cmd_depth = int(param(node, "servo_qos_cmd_depth", 1))
            status_depth = int(param(node, "servo_qos_status_depth", 3))
        elif qos_mode in ("modern", "new", "v2", "auto"):
            # Modern profile (recommended / MoveIt Servo compatible):
            # cmd: RELIABLE + VOLATILE
            # status: RELIABLE + VOLATILE
            cmd_reliability = ReliabilityPolicy.RELIABLE
            cmd_durability = DurabilityPolicy.VOLATILE
            status_reliability = ReliabilityPolicy.RELIABLE
            status_durability = DurabilityPolicy.VOLATILE
            cmd_depth = int(param(node, "servo_qos_cmd_depth", 1))
            status_depth = int(param(node, "servo_qos_status_depth", 3))
        else:
            # Custom profile for advanced tuning / special endpoints.
            cmd_reliability = self._qos_reliability(
                str(param(node, "servo_qos_cmd_reliability", "reliable")).lower(),
                ReliabilityPolicy.RELIABLE,
            )
            cmd_durability = self._qos_durability(
                str(param(node, "servo_qos_cmd_durability", "volatile")).lower(),
                DurabilityPolicy.VOLATILE,
            )
            status_reliability = self._qos_reliability(
                str(param(node, "servo_qos_status_reliability", "reliable")).lower(),
                ReliabilityPolicy.RELIABLE,
            )
            status_durability = self._qos_durability(
                str(param(node, "servo_qos_status_durability", "volatile")).lower(),
                DurabilityPolicy.VOLATILE,
            )
            cmd_depth = int(param(node, "servo_qos_cmd_depth", 1))
            status_depth = int(param(node, "servo_qos_status_depth", 3))

        qos_status = QoSProfile(
            reliability=status_reliability,
            durability=status_durability,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, status_depth),
        )
        qos_cmd = QoSProfile(
            reliability=cmd_reliability,
            durability=cmd_durability,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, cmd_depth),
        )
        node.get_logger().info(
            f"Servo QoS mode={qos_mode}: "
            f"cmd(reliability={self._qos_reliability_name(cmd_reliability)}, durability={self._qos_durability_name(cmd_durability)}, depth={max(1, cmd_depth)}), "
            f"status(reliability={self._qos_reliability_name(status_reliability)}, durability={self._qos_durability_name(status_durability)}, depth={max(1, status_depth)})"
        )

        self._joint_sub = node.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self._ee_vel_pub = node.create_publisher(TwistStamped, "/ee_velocity", 10)
        self._act_latency_pub = node.create_publisher(Float32MultiArray, "/servo_act_latency_trace", 10)
        self._servo_out = {
            "last_time": None,
            "previous_endpoint": None,
            "endpoint_delta_rad": float("nan"),
            "endpoint_speed_rad_s": float("nan"),
            "count": 0,
            "window_start": time.monotonic(),
        }

        self.servo_twist_topic = f"{self.servo_ns}/delta_twist_cmds"
        self.servo_status_topic = f"{self.servo_ns}/status"
        self.servo_twist_pub = node.create_publisher(TwistStamped, self.servo_twist_topic, qos_cmd)

        self.last_servo_status_code = 0
        node.create_subscription(Int8, self.servo_status_topic, self._on_servo_status_int8, qos_status)

        self.start_servo_cli = node.create_client(Trigger, f"{self.servo_ns}/start_servo")
        self.stop_servo_cli = node.create_client(Trigger, f"{self.servo_ns}/stop_servo")
        self.pause_servo_cli = node.create_client(Trigger, f"{self.servo_ns}/pause_servo")
        self.unpause_servo_cli = node.create_client(Trigger, f"{self.servo_ns}/unpause_servo")
        self.reset_servo_status_cli = node.create_client(Empty, f"{self.servo_ns}/reset_servo_status")
        self.servo_started = False
        self._start_servo_future = None
        self._start_servo_deadline = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

        try:
            node.create_subscription(Float64, f"{self.servo_ns}/collision_velocity_scale", self._on_collision_scale, 10)
        except Exception:
            node.get_logger().warn("Cannot subscribe collision_velocity_scale (Float64).")
        try:
            node.create_subscription(JointTrajectory, "/robot_arm_controller/joint_trajectory", self._on_servo_out, 10)
        except Exception:
            node.get_logger().warn("Cannot subscribe /robot_arm_controller/joint_trajectory (JointTrajectory).")

    @staticmethod
    def _qos_reliability(value: str, default: ReliabilityPolicy) -> ReliabilityPolicy:
        if value in ("reliable", "rel"):
            return ReliabilityPolicy.RELIABLE
        if value in ("best_effort", "besteffort", "best"):
            return ReliabilityPolicy.BEST_EFFORT
        return default

    @staticmethod
    def _qos_durability(value: str, default: DurabilityPolicy) -> DurabilityPolicy:
        if value in ("volatile", "vol"):
            return DurabilityPolicy.VOLATILE
        if value in ("transient_local", "transientlocal", "transient"):
            return DurabilityPolicy.TRANSIENT_LOCAL
        return default

    @staticmethod
    def _qos_reliability_name(policy: ReliabilityPolicy) -> str:
        if policy == ReliabilityPolicy.RELIABLE:
            return "reliable"
        if policy == ReliabilityPolicy.BEST_EFFORT:
            return "best_effort"
        return "unknown"

    @staticmethod
    def _qos_durability_name(policy: DurabilityPolicy) -> str:
        if policy == DurabilityPolicy.VOLATILE:
            return "volatile"
        if policy == DurabilityPolicy.TRANSIENT_LOCAL:
            return "transient_local"
        return "unknown"

    @staticmethod
    def _dh_matrix(theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _fkine_all(self, q):
        transforms = [np.eye(4, dtype=np.float64)]
        for i in range(6):
            transforms.append(
                transforms[-1] @ self._dh_matrix(q[i], self._DH_D[i], self._DH_A[i], self._DH_ALPHA[i])
            )
        return transforms

    def _fkine_to_ee(self, q):
        transforms = self._fkine_all(q)
        t06 = transforms[6]
        if not self._use_tool_offset:
            return t06, transforms
        t_tool = np.eye(4, dtype=np.float64)
        t_tool[:3, 3] = self._TOOL_OFFSET
        return t06 @ t_tool, transforms

    def _compute_jacobian(self, q):
        t_ee, transforms = self._fkine_to_ee(q)
        p_ee = t_ee[:3, 3]
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            z_i = transforms[i][:3, 2]
            p_i = transforms[i][:3, 3]
            jacobian[:3, i] = np.cross(z_i, p_ee - p_i)
            jacobian[3:, i] = z_i
        return jacobian, t_ee

    def _on_joint_state(self, msg: JointState):
        positions = []
        velocities = []
        for name in self._JOINT_NAMES:
            if name not in msg.name:
                return
            idx = msg.name.index(name)
            positions.append(msg.position[idx])
            velocities.append(msg.velocity[idx] if idx < len(msg.velocity) else 0.0)
        self._joint_positions = np.array(positions, dtype=np.float64)
        self._joint_velocities = np.array(velocities, dtype=np.float64)
        self._last_joint_state_time = time.monotonic()
        self._compute_ee_velocity()

    def _compute_ee_velocity(self):
        if self._joint_positions is None or self._joint_velocities is None:
            return
        jacobian, t_ee = self._compute_jacobian(self._joint_positions)
        self._ee_twist = jacobian @ self._joint_velocities
        self._ee_pose_position = t_ee[:3, 3].copy()
        self._ee_pose_rotation = t_ee[:3, :3].copy()
        self._ee_linear_speed = float(np.linalg.norm(self._ee_twist[:3]))
        self._ee_angular_speed = float(np.linalg.norm(self._ee_twist[3:]))
        self._publish_ee_velocity()

    def _publish_ee_velocity(self):
        now = self.node.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        msg = TwistStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = float(self._ee_twist[0])
        msg.twist.linear.y = float(self._ee_twist[1])
        msg.twist.linear.z = float(self._ee_twist[2])
        msg.twist.angular.x = float(self._ee_twist[3])
        msg.twist.angular.y = float(self._ee_twist[4])
        msg.twist.angular.z = float(self._ee_twist[5])
        self._ee_vel_pub.publish(msg)

        ee_lin = self._ee_twist[:3].copy()
        delta_ee = float(np.linalg.norm(ee_lin - self._last_ee_lin))
        self._last_ee_lin = ee_lin
        while self._cmd_events:
            event = self._cmd_events[0]
            lag = float(now_sec - event["t_cmd"])
            if lag < 0.0:
                break
            if lag > 0.5:
                self._cmd_events.popleft()
                continue
            if delta_ee >= self._ee_response_threshold:
                try:
                    trace = Float32MultiArray()
                    trace.data = [
                        float(event["t_cmd"]),
                        float(now_sec),
                        float(lag),
                        float(event["cmd_vec"][0]),
                        float(event["cmd_vec"][1]),
                        float(ee_lin[0]),
                        float(ee_lin[1]),
                    ]
                    self._act_latency_pub.publish(trace)
                except Exception:
                    pass
                self._cmd_events.popleft()
            break

    @property
    def ee_twist(self):
        return self._ee_twist.copy()

    @property
    def ee_linear_speed(self) -> float:
        return self._ee_linear_speed

    @property
    def ee_angular_speed(self) -> float:
        return self._ee_angular_speed

    @property
    def ee_linear_velocity(self):
        return self._ee_twist[:3].copy()

    @property
    def ee_angular_velocity(self):
        return self._ee_twist[3:].copy()

    @property
    def ee_position_dh(self):
        return self._ee_pose_position.copy()

    @property
    def ee_rotation_dh(self):
        return self._ee_pose_rotation.copy()

    @property
    def joint_positions(self):
        return self._joint_positions.copy() if self._joint_positions is not None else None

    @property
    def joint_velocities(self):
        return self._joint_velocities.copy() if self._joint_velocities is not None else None

    def get_ee_velocity_dict(self) -> dict:
        return {
            "vx": float(self._ee_twist[0]),
            "vy": float(self._ee_twist[1]),
            "vz": float(self._ee_twist[2]),
            "wx": float(self._ee_twist[3]),
            "wy": float(self._ee_twist[4]),
            "wz": float(self._ee_twist[5]),
            "linear_speed": self._ee_linear_speed,
            "angular_speed": self._ee_angular_speed,
            "ee_x": float(self._ee_pose_position[0]),
            "ee_y": float(self._ee_pose_position[1]),
            "ee_z": float(self._ee_pose_position[2]),
        }

    def get_ee_fk_pose(self):
        if self._joint_positions is None:
            return None, None
        t_ee, _ = self._fkine_to_ee(self._joint_positions)
        return t_ee[:3, 3], R.from_matrix(t_ee[:3, :3]).as_quat()

    def _on_collision_scale(self, msg: Float64):
        self._last_collision_scale = float(msg.data)

    def _on_servo_out(self, msg: JointTrajectory):
        now = time.monotonic()
        self._last_servo_out = (now, len(msg.points))
        self._servo_out["count"] += 1
        if not msg.points or not msg.joint_names:
            return

        point = msg.points[-1]
        if len(point.positions) != len(msg.joint_names):
            return
        positions_by_name = dict(zip(msg.joint_names, point.positions))
        if not all(name in positions_by_name for name in self._JOINT_NAMES):
            return
        endpoint = np.array([positions_by_name[name] for name in self._JOINT_NAMES], dtype=np.float64)
        previous = self._servo_out["previous_endpoint"]
        self._servo_out["endpoint_delta_rad"] = (
            float(np.max(np.abs(endpoint - previous)))
            if previous is not None
            else 0.0
        )
        self._servo_out["previous_endpoint"] = endpoint

        if len(point.velocities) == len(msg.joint_names):
            velocities_by_name = dict(zip(msg.joint_names, point.velocities))
            self._servo_out["endpoint_speed_rad_s"] = float(
                max(abs(velocities_by_name[name]) for name in self._JOINT_NAMES)
            )

    def servo_chain_summary(self) -> dict:
        now = time.monotonic()
        elapsed = max(now - self._servo_out["window_start"], 1e-6)
        rate_hz = self._servo_out["count"] / elapsed
        self._servo_out["count"] = 0
        self._servo_out["window_start"] = now

        tracking_error = float("nan")
        endpoint = self._servo_out["previous_endpoint"]
        if endpoint is not None and self._joint_positions is not None:
            tracking_error = float(np.max(np.abs(endpoint - self._joint_positions)))

        return {
            "rate_hz": float(rate_hz),
            "endpoint_delta_rad": float(self._servo_out["endpoint_delta_rad"]),
            "endpoint_speed_rad_s": float(self._servo_out["endpoint_speed_rad_s"]),
            "tracking_error_rad": tracking_error,
            "actual_speed_rad_s": float(np.max(np.abs(self._joint_velocities)))
            if self._joint_velocities is not None
            else float("nan"),
        }

    def servo_output_age_sec(self) -> float:
        if self._last_servo_out is None:
            return float("inf")
        return max(0.0, time.monotonic() - self._last_servo_out[0])

    def joint_state_age_sec(self) -> float:
        if self._last_joint_state_time is None:
            return float("inf")
        return max(0.0, time.monotonic() - self._last_joint_state_time)

    def _on_servo_status_int8(self, msg: Int8):
        self.last_servo_status_code = int(msg.data)

    def reset_servo_status(self):
        if self.reset_servo_status_cli.wait_for_service(timeout_sec=0.2):
            self.reset_servo_status_cli.call_async(Empty.Request())

    def pause_servo(self):
        if self.pause_servo_cli.wait_for_service(timeout_sec=0.2):
            self.pause_servo_cli.call_async(Trigger.Request())

    def unpause_servo(self):
        if self.unpause_servo_cli.wait_for_service(timeout_sec=0.2):
            self.unpause_servo_cli.call_async(Trigger.Request())

    def start_servo(self, timeout_sec: float = 2.0) -> bool:
        if not self.start_servo_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().error("Servo start service not available. Is /servo_node running?")
            return False
        future = self.start_servo_cli.call_async(Trigger.Request())
        t0 = time.time()
        while rclpy.ok() and not future.done() and time.time() - t0 < timeout_sec:
            time.sleep(0.01)
        if not future.done():
            self.node.get_logger().error("start_servo timeout")
            return False
        response = future.result()
        if response is None or not response.success:
            self.node.get_logger().error(f"start_servo failed: {response.message if response else 'no response'}")
            return False
        self.servo_started = True
        self.node.get_logger().info("✓ Servo started")
        self.reset_servo_status()
        self.unpause_servo()
        self.publish_zero_twist(n=5, dt=0.01)
        return True

    def start_servo_async(self, timeout_sec: float = 2.0) -> bool | None:
        """Advance a non-blocking MoveIt Servo start request.

        Returns True after a successful start, None while waiting for the
        service or its response, and False after a failed or timed-out call.
        """
        if self.servo_started:
            return True
        if self._start_servo_future is None:
            if not self.start_servo_cli.service_is_ready():
                return None
            self._start_servo_future = self.start_servo_cli.call_async(Trigger.Request())
            self._start_servo_deadline = time.monotonic() + timeout_sec
            return None
        if not self._start_servo_future.done():
            if time.monotonic() < self._start_servo_deadline:
                return None
            self._start_servo_future = None
            self.node.get_logger().error("start_servo timeout")
            return False
        future = self._start_servo_future
        self._start_servo_future = None
        try:
            response = future.result()
        except Exception as error:
            self.node.get_logger().error(f"start_servo failed: {error}")
            return False
        if response is None or not response.success:
            self.node.get_logger().error(
                f"start_servo failed: {response.message if response else 'no response'}"
            )
            return False
        self.servo_started = True
        self.node.get_logger().info("✓ Servo started")
        if self.reset_servo_status_cli.service_is_ready():
            self.reset_servo_status_cli.call_async(Empty.Request())
        if self.unpause_servo_cli.service_is_ready():
            self.unpause_servo_cli.call_async(Trigger.Request())
        self.publish_zero_twist(n=1, dt=0.0)
        return True

    def stop_servo(self):
        self.publish_zero_twist(n=5, dt=0.01)
        if self.stop_servo_cli.wait_for_service(timeout_sec=0.5):
            self.stop_servo_cli.call_async(Trigger.Request())
        self.servo_started = False
        self.clear_last_command()

    def publish_twist(self, vx: float, vy: float, vz: float, wz: float):
        return self.publish_twist_6d(vx, vy, vz, 0.0, 0.0, wz)

    def publish_twist_6d(self, vx: float, vy: float, vz: float, wx: float, wy: float, wz: float):
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        msg.twist.angular.x = float(wx)
        msg.twist.angular.y = float(wy)
        msg.twist.angular.z = float(wz)
        self.servo_twist_pub.publish(msg)

        t_pub_sec = self.node.get_clock().now().nanoseconds * 1e-9
        cmd_vec = np.array([vx, vy, vz, wz], dtype=np.float64)
        cmd_norm = float(np.linalg.norm(cmd_vec))
        delta_cmd = float(np.linalg.norm(cmd_vec - self._last_cmd_vec))
        if cmd_norm >= self._cmd_event_threshold and delta_cmd >= self._cmd_delta_threshold:
            self._cmd_events.append({"t_cmd": t_pub_sec, "cmd_vec": cmd_vec.copy()})
        self._last_cmd_vec = cmd_vec
        self._last_cmd_norm = cmd_norm
        return t_pub_sec

    def clear_last_command(self):
        self._last_cmd_vec[:] = 0.0
        self._last_cmd_norm = 0.0

    def publish_zero_twist(self, n: int = 1, dt: float = 0.0):
        for _ in range(max(1, n)):
            self.publish_twist(0.0, 0.0, 0.0, 0.0)
            if dt > 0:
                time.sleep(dt)

    def quiesce_before_global_motion(
        self,
        zero_twist_count: int = 10,
        zero_twist_dt: float = 0.01,
        ee_speed_tol: float = 0.003,
        joint_speed_tol: float = 0.01,
        stable_sec: float = 0.08,
        timeout_sec: float = 0.5,
    ) -> bool:
        self.publish_zero_twist(n=int(zero_twist_count), dt=float(zero_twist_dt))
        self.node.get_logger().info(
            f"Servo handoff: zero twist sent n={int(zero_twist_count)}, dt={float(zero_twist_dt):.3f}s"
        )

        if self.servo_started:
            self.stop_servo()
        else:
            self.clear_last_command()

        t0 = time.monotonic()
        stable_t0 = None
        last_log_t = 0.0
        last_ee_speed = 0.0
        last_joint_speed = 0.0

        while rclpy.ok() and (time.monotonic() - t0) <= float(timeout_sec):
            joint_vel = self.joint_velocities
            last_ee_speed = float(self.ee_linear_speed)
            last_joint_speed = 0.0
            if joint_vel is not None and len(joint_vel) > 0:
                last_joint_speed = float(np.max(np.abs(joint_vel)))

            is_still = last_ee_speed <= float(ee_speed_tol) and last_joint_speed <= float(joint_speed_tol)
            now = time.monotonic()
            if is_still:
                if stable_t0 is None:
                    stable_t0 = now
                stable_ms = (now - stable_t0) * 1000.0
                if stable_ms >= float(stable_sec) * 1000.0:
                    self.node.get_logger().info(
                        "Servo handoff complete -> RETURNING_HOME "
                        f"ee_speed={last_ee_speed:.5f}, joint_speed={last_joint_speed:.5f}, "
                        f"stable_ms={stable_ms:.1f}"
                    )
                    self.clear_last_command()
                    return True
            else:
                stable_t0 = None

            if now - last_log_t >= 0.15:
                stable_ms = 0.0 if stable_t0 is None else (now - stable_t0) * 1000.0
                self.node.get_logger().info(
                    "Servo handoff: "
                    f"ee_speed={last_ee_speed:.5f}, joint_speed={last_joint_speed:.5f}, "
                    f"stable_ms={stable_ms:.1f}"
                )
                last_log_t = now
            time.sleep(0.01)

        self.node.get_logger().warn(
            "Servo handoff timeout: "
            f"ee_speed={last_ee_speed:.5f}, joint_speed={last_joint_speed:.5f}"
        )
        self.clear_last_command()
        return False

    def get_current_ee_pose_base(self):
        try:
            transform = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, rclpy.time.Time())
            p = transform.transform.translation
            q = transform.transform.rotation
            return np.array([p.x, p.y, p.z], dtype=float), np.array([q.x, q.y, q.z, q.w], dtype=float)
        except Exception:
            return None, None


__all__ = ["ServoIO"]
