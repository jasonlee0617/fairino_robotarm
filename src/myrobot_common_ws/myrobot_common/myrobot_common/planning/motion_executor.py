import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Any

from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath


@dataclass
class PlanScoreConfig:
    num_candidates: int = 5
    wrist_weight: float = 50.0
    wrist_joint_indices: Tuple[int, int, int] = (2, 3, 4)


class PlannerSwitch:
    FAIRINO_PLANNERS = {"aapf_birrt*", "tube_birrt*", "birrt*", "rrt*"}

    @staticmethod
    def normalize_pipeline(value: str) -> str:
        token = str(value or "").strip().lower()
        if token in ("fairino", "fairino_planning"):
            return "fairino"
        if token in ("kdl", "ompl"):
            return "ompl"
        return token

    @staticmethod
    def normalize_ik(value: str) -> str:
        token = str(value or "").strip().lower()
        if token in ("fairino", "fr", "fairino_ik"):
            return "fairino"
        if token in ("kdl", "ompl"):
            return "kdl"
        return token

    @staticmethod
    def normalize_planner(pipeline: str, planner: str) -> str:
        raw = str(planner or "").strip()
        key = raw.lower()
        if PlannerSwitch.normalize_pipeline(pipeline) == "fairino":
            fairino_aliases = {
                "aapf": "aapf_birrt*",
                "aapf_birrt": "aapf_birrt*",
                "aapf-birrt": "aapf_birrt*",
                "aapf_birrt*": "aapf_birrt*",
                "aapf-birrt*": "aapf_birrt*",
                "tube": "tube_birrt*",
                "tube_birrt": "tube_birrt*",
                "tube-birrt": "tube_birrt*",
                "tube_birrt*": "tube_birrt*",
                "tube-birrt*": "tube_birrt*",
                "birrt": "birrt*",
                "birrt*": "birrt*",
                "rrt": "rrt*",
                "rrt*": "rrt*",
            }
            return fairino_aliases.get(key, raw)
        if key in ("rrtconnect", "rrtconnectfast", "rrtconnectkconfigdefault"):
            return "RRTConnectFast"
        return raw

    @staticmethod
    def is_valid(pipeline: str, planner: str) -> bool:
        normalized_pipeline = PlannerSwitch.normalize_pipeline(pipeline)
        normalized_planner = PlannerSwitch.normalize_planner(normalized_pipeline, planner)
        if normalized_pipeline == "fairino":
            return normalized_planner in PlannerSwitch.FAIRINO_PLANNERS
        if normalized_pipeline == "ompl":
            return bool(normalized_planner)
        return False


class MoveItMotion:
    def __init__(
        self,
        node,
        arm=None,
        pose_tools=None,
        arm_clients=None,
        default_client=None,
        abort=None,
        gripper=None,
        select_best_path: Optional[Callable[..., Any]] = None,
        score_cfg: PlanScoreConfig = PlanScoreConfig(),
        action_delay: float = 0.5,
        joint_constraint: Optional[dict] = None,
        open_positions: Sequence[float] = (0.0305, -0.0305),
        close_positions: Sequence[float] = (0.0, 0.0),
    ):
        # ── resolve arm_clients / default_client from legacy arm= kwarg ──
        if arm is not None and arm_clients is None:
            arm_clients = {"fairino": arm}
        if arm_clients is None:
            raise ValueError("MoveItMotion requires either arm= or arm_clients=")
        if default_client is None:
            default_client = "fairino"

        self.node = node
        self.arm_clients = dict(arm_clients)
        self.current_client = PlannerSwitch.normalize_ik(default_client)
        self.gripper = gripper
        self.pose_tools = pose_tools
        self.abort = abort
        self.select_best_path = select_best_path
        self.score_cfg = score_cfg
        self.action_delay = float(action_delay)
        self.joint_constraint = joint_constraint
        self.open_positions = tuple(open_positions)
        self.close_positions = tuple(close_positions)
        self._fairino_cartesian_client = node.create_client(GetCartesianPath, "/fairino_cartesian_path")
        self._fairino_cartesian_start_guard_tol = self._node_param_float(
            "fairino.cartesian_path_planner.start_state_guard_tolerance_rad", 0.005
        )
        self.node.get_logger().info(
            "Motion planning policy: cartesian=True uses direct Cartesian planning; "
            "cartesian=False uses candidate scoring."
        )

    def wait_client_ready(self, planning_client=None, timeout_sec=3.0) -> bool:
        arm = self._select_arm(planning_client)
        cli = getattr(arm, "_plan_kinematic_path_service", None)
        if cli is None:
            self.node.get_logger().warn(
                f"No _plan_kinematic_path_service on '{self._client_name(arm)}'; "
                f"skipping ready check."
            )
            return True
        if not cli.wait_for_service(timeout_sec=float(timeout_sec)):
            if timeout_sec > 0.0:
                self.node.get_logger().error(
                    f"MoveIt planning service '{self._client_name(arm)}' not ready "
                    f"after {timeout_sec:.1f}s."
                )
            return False
        return True

    def _node_param_float(self, name: str, fallback: float) -> float:
        try:
            if not self.node.has_parameter(name):
                self.node.declare_parameter(name, float(fallback))
            return float(self.node.get_parameter(name).value)
        except Exception:
            return float(fallback)

    @property
    def arm(self):
        return self._select_arm(None)

    def _select_arm(self, planning_client: Optional[str]):
        key = PlannerSwitch.normalize_ik(planning_client or self.current_client)
        selected = self.arm_clients.get(key)
        if selected is not None:
            return selected
        self.node.get_logger().warn(f"Unknown planning client '{planning_client}', fallback to fairino.")
        return self.arm_clients.get("fairino", next(iter(self.arm_clients.values())))

    def _planning_client_key(self, planning_client: Optional[str], arm) -> str:
        if planning_client:
            return PlannerSwitch.normalize_ik(planning_client)
        for key, value in self.arm_clients.items():
            if value is arm:
                return PlannerSwitch.normalize_ik(key)
        return PlannerSwitch.normalize_ik(self.current_client)

    def set_ik(self, plugin: str) -> bool:
        key = PlannerSwitch.normalize_ik(plugin)
        if key not in self.arm_clients:
            self.node.get_logger().error(f"Unsupported IK plugin '{plugin}'. Use fairino or kdl.")
            return False
        self.current_client = key
        if self.abort is not None:
            self.abort.arm = self._select_arm(key)
        self.node.get_logger().info(f"IK client switched: {key}")
        return True

    def set_planner(self, pipeline: str, planner: str, raw_algorithm: Optional[str] = None) -> bool:
        normalized_pipeline = PlannerSwitch.normalize_pipeline(pipeline)
        normalized_planner = PlannerSwitch.normalize_planner(normalized_pipeline, planner)
        raw = raw_algorithm if raw_algorithm is not None else planner
        if not PlannerSwitch.is_valid(normalized_pipeline, normalized_planner):
            self.node.get_logger().error(
                f"Unsupported planner command: pipeline={pipeline}, algorithm={raw}. "
                "Fairino supports aapf_birrt*, tube_birrt*, birrt*, rrt*."
            )
            return False

        for arm in self.arm_clients.values():
            if arm is None:
                continue
            arm.pipeline_id = normalized_pipeline
            arm.planner_id = normalized_planner

        self.node.get_logger().info(
            f"Planner switched: pipeline={normalized_pipeline}, "
            f"raw_algorithm={raw}, algorithm={normalized_planner}"
        )
        return True

    def handle_command(self, msg):
        text = str(msg.data).strip()
        if not text:
            return
        parts = text.split()
        if len(parts) >= 2 and parts[0].lower() == "ik":
            self.set_ik(parts[1])
            return
        if len(parts) >= 3 and parts[0].lower() == "planner":
            self.set_planner(parts[1], parts[2], raw_algorithm=parts[2])
            return
        self.node.get_logger().warn(
            "Unsupported planner command. Use 'ik fairino', 'ik kdl', "
            "'planner fairino tube_birrt*', or 'planner ompl RRTConnect'."
        )

    def plan_to_pose(
        self,
        target_pose,
        planning_client: Optional[str] = None,
        cartesian: bool = False,
        action_name: str = "move",
        max_velocity: float = 0.2,
        max_acceleration: float = 0.2,
        max_step_size: Optional[float] = None,
        allowed_planning_time: Optional[float] = None,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
        allowed_start_tolerance: Optional[float] = None,
        joint_constraint: Optional[dict] = None,
        start_joint_state=None,
    ):
        """Plan a pose target and return its JointTrajectory without executing it."""
        arm = self._select_arm(planning_client)
        if isinstance(target_pose, Pose):
            target_pose = self.pose_tools.to_pose_stamped(target_pose)
        if self._aborted():
            self.node.get_logger().warn(f"{action_name}: motion control is blocked")
            return False

        self.node.get_logger().info(
            f"{action_name}: ({target_pose.pose.position.x:.3f}, "
            f"{target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f}), "
            f"cartesian={bool(cartesian)}, client={self._client_name(arm)}"
        )

        try:
            arm.max_velocity = float(max_velocity)
            arm.max_acceleration = float(max_acceleration)
            if allowed_planning_time is not None:
                arm.allowed_planning_time = float(allowed_planning_time)
            tolerance_position = 0.001 if position_tolerance is None else float(position_tolerance)
            tolerance_orientation = 0.001 if orientation_tolerance is None else float(orientation_tolerance)
            max_step = 0.0025 if max_step_size is None else float(max_step_size)
            pipeline_id = PlannerSwitch.normalize_pipeline(getattr(arm, "pipeline_id", "ompl"))
            if cartesian and pipeline_id == "fairino":
                planner_mode = "fairino_cartesian"
            elif cartesian:
                planner_mode = "moveit_cartesian"
            elif pipeline_id == "fairino":
                planner_mode = "fairino_global_single"
            else:
                planner_mode = "ompl_global_candidate_scored"
            self.node.get_logger().info(
                f"{action_name}: planner_mode={planner_mode}, "
                f"pipeline={pipeline_id}, planner={getattr(arm, 'planner_id', '')}, "
                f"velocity={arm.max_velocity:.3f}, acceleration={arm.max_acceleration:.3f}"
            )

            paths = []
            num_trials = 1 if cartesian or pipeline_id == "fairino" else int(self.score_cfg.num_candidates)
            for _ in range(max(1, num_trials)):
                if self._aborted():
                    return False
                try:
                    arm.clear_path_constraints()
                    if joint_constraint is False:
                        constraint = None
                    else:
                        constraint = joint_constraint if joint_constraint is not None else self.joint_constraint
                    if constraint is not None:
                        arm.set_path_joint_constraint(
                            joint_positions=constraint["joint_positions"],
                            joint_names=constraint["joint_names"],
                            tolerance=constraint.get("tolerance", 0.0),
                            weight=constraint.get("weight", 1.0),
                        )
                    if cartesian and pipeline_id == "fairino":
                        plan = self._plan_fairino_cartesian(
                            arm=arm,
                            target_pose=target_pose,
                            action_name=action_name,
                            fraction_threshold=0.98,
                        )
                    else:
                        plan = arm.plan(
                            target_pose,
                            cartesian=cartesian,
                            cartesian_fraction_threshold=0.98 if cartesian else 0.0,
                            tolerance_position=tolerance_position,
                            tolerance_orientation=tolerance_orientation,
                            max_step=max_step,
                            start_joint_state=start_joint_state,
                        )
                    if plan:
                        paths.append(plan)
                except Exception as exc:
                    self.node.get_logger().warn(f"{action_name}: plan failed: {exc}")

            if not paths:
                self.node.get_logger().error(f"{action_name}: No valid plan generated.")
                return None
            if self._aborted():
                self.node.get_logger().warn(f"{action_name}: aborted before execute")
                return None

            best_path = self._pick_path(paths, cartesian, action_name)
            return arm._retime_trajectory_if_needed(best_path, cartesian=cartesian)
        except Exception as exc:
            self.node.get_logger().error(f"✗ {action_name} exception: {exc}")
            return None

    def move_to_pose(
        self,
        target_pose,
        planning_client: Optional[str] = None,
        cartesian: bool = False,
        action_name: str = "move",
        max_velocity: float = 0.2,
        max_acceleration: float = 0.2,
        max_step_size: Optional[float] = None,
        allowed_planning_time: Optional[float] = None,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
        allowed_start_tolerance: Optional[float] = None,
        timeout_sec: float = 30.0,
        joint_constraint: Optional[dict] = None,
    ) -> bool:
        """Plan then execute a pose target using the shared planning path."""
        arm = self._select_arm(planning_client)
        best_path = self.plan_to_pose(
            target_pose,
            planning_client=planning_client,
            cartesian=cartesian,
            action_name=action_name,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_step_size=max_step_size,
            allowed_planning_time=allowed_planning_time,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            allowed_start_tolerance=allowed_start_tolerance,
            joint_constraint=joint_constraint,
        )
        if best_path is None:
            return False
        if cartesian:
            best_path.header.stamp.sec = 0
            best_path.header.stamp.nanosec = 0
        arm.execute(best_path)
        if not self._wait(arm, action_name, timeout_sec):
            self.node.get_logger().error(f"✗ {action_name} aborted/failed.")
            return False
        self.node.get_logger().info(f"✓ {action_name} done.")
        time.sleep(self.action_delay)
        return True

    def move_to_joints(
        self,
        joint_positions: Sequence[float],
        action_name: str = "move_joints",
        planning_client: Optional[str] = None,
        timeout_sec: float = 30.0,
        *,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        allowed_planning_time: Optional[float] = None,
        allowed_start_tolerance: Optional[float] = None,
    ) -> bool:
        arm = self._select_arm(planning_client)
        if self._aborted():
            self.node.get_logger().warn(f"{action_name}: motion control is blocked")
            return False
        try:
            self.node.get_logger().info(action_name)
            ok = self._plan_and_execute_configuration(
                arm, joint_positions, action_name, timeout_sec,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
                allowed_planning_time=allowed_planning_time,
                allowed_start_tolerance=allowed_start_tolerance,
            )
            if not ok:
                self.node.get_logger().error(f"✗ {action_name} aborted/failed.")
                return False
            time.sleep(self.action_delay)
            return True
        except Exception as exc:
            self.node.get_logger().error(f"✗ {action_name} exception: {exc}")
            return False

    def control_gripper(
        self,
        open_gripper: bool = True,
        action_name: Optional[str] = None,
        timeout_sec: float = 10.0,
        positions: Optional[Sequence[float]] = None,
    ) -> bool:
        if action_name is None:
            action_name = "Open gripper" if open_gripper else "Close gripper"
        positions = tuple(positions) if positions is not None else (
            self.open_positions if open_gripper else self.close_positions
        )
        if self._aborted():
            self.node.get_logger().warn(f"{action_name}: motion control is blocked")
            return False
        if not open_gripper and self._gripper_at_positions(positions):
            self.node.get_logger().info(f"{action_name}: already closed; skipping trajectory")
            return True
        self.node.get_logger().info(action_name)
        try:
            ok = self._plan_and_execute_configuration(
                self.gripper, positions, action_name, timeout_sec
            )
            if not ok:
                self.node.get_logger().error(f"✗ {action_name} aborted/failed.")
                return False
            time.sleep(self.action_delay)
            return True
        except Exception as exc:
            self.node.get_logger().warn(f"{action_name} exception: {exc}")
            time.sleep(self.action_delay)
            return False

    def _gripper_at_positions(self, positions: Sequence[float], tolerance: float = 0.001) -> bool:
        """Return true only when both physical gripper joints are at target."""
        current = self._gripper_joint_positions()
        if current is None or len(positions) != 2:
            return False
        return all(
            name in current
            and abs(float(current[name]) - float(target)) <= float(tolerance) + 1e-9
            for name, target in zip(("finger1_joint", "finger2_joint"), positions)
        )

    def _gripper_joint_positions(self):
        state = getattr(self.gripper, "joint_state", None)
        if state is None:
            return None
        names = getattr(state, "name", ())
        values = getattr(state, "position", ())
        return {name: values[i] for i, name in enumerate(names) if i < len(values)}

    @staticmethod
    def _has_strictly_increasing_times(trajectory) -> bool:
        joint_trajectory = getattr(trajectory, "joint_trajectory", trajectory)
        points = getattr(joint_trajectory, "points", ())
        if len(points) < 2:
            return False
        previous = -1.0
        for point in points:
            stamp = getattr(point, "time_from_start", None)
            seconds = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
            if seconds <= previous:
                return False
            previous = seconds
        return True

    def _plan_and_execute_configuration(
        self,
        moveit_obj,
        positions,
        action_name: str,
        timeout_sec: float,
        *,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        allowed_planning_time: Optional[float] = None,
        allowed_start_tolerance: Optional[float] = None,
    ) -> bool:
        """Plan and execute a joint target within one interruptible deadline.

        可选参数临时覆盖 moveit_obj 的运动属性，执行后恢复（向后兼容）。
        """
        saved = {}
        try:
            if max_velocity is not None:
                saved["max_velocity"] = getattr(moveit_obj, "max_velocity", None)
                moveit_obj.max_velocity = max_velocity
            if max_acceleration is not None:
                saved["max_acceleration"] = getattr(moveit_obj, "max_acceleration", None)
                moveit_obj.max_acceleration = max_acceleration
            if allowed_planning_time is not None:
                saved["allowed_planning_time"] = getattr(moveit_obj, "allowed_planning_time", None)
                moveit_obj.allowed_planning_time = allowed_planning_time
            if allowed_start_tolerance is not None:
                saved["allowed_start_tolerance"] = getattr(moveit_obj, "allowed_start_tolerance", None)
                moveit_obj.allowed_start_tolerance = allowed_start_tolerance
            return self._plan_and_execute_configuration_inner(
                moveit_obj, positions, action_name, timeout_sec
            )
        finally:
            for name, value in saved.items():
                setattr(moveit_obj, name, value)

    def _plan_and_execute_configuration_inner(
        self, moveit_obj, positions, action_name: str, timeout_sec: float
    ) -> bool:
        """Plan and execute a joint target within one interruptible deadline."""
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        future = moveit_obj.plan_async(joint_positions=list(positions))
        if future is None:
            self.node.get_logger().error(f"{action_name}: planning did not start")
            return False

        while not future.done():
            if self._aborted():
                self.node.get_logger().warn(f"{action_name}: aborted while planning")
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.node.get_logger().error(f"{action_name}: planning timeout")
                return False
            time.sleep(min(0.02, remaining))

        trajectory = moveit_obj.get_trajectory(future)
        if trajectory is None:
            self.node.get_logger().error(f"{action_name}: no valid plan generated")
            return False
        if moveit_obj is self.gripper and not self._has_strictly_increasing_times(trajectory):
            if self._gripper_at_positions(positions):
                self.node.get_logger().info(f"{action_name}: already at target; ignoring empty trajectory")
                return True
            self.node.get_logger().error(
                f"{action_name}: gripper trajectory has non-increasing timestamps; "
                f"target={tuple(float(value) for value in positions)}, "
                f"current={self._gripper_joint_positions()}"
            )
            return False
        if self._aborted():
            self.node.get_logger().warn(f"{action_name}: aborted before execute")
            return False

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            self.node.get_logger().error(f"{action_name}: timeout before execute")
            return False
        moveit_obj.execute(trajectory)
        return self._wait(moveit_obj, action_name, remaining)

    def _plan_fairino_cartesian(self, arm, target_pose, action_name: str, fraction_threshold: float):
        if not self._fairino_cartesian_client.wait_for_service(timeout_sec=0.5):
            self.node.get_logger().error(
                f"{action_name}: /fairino_cartesian_path service not available."
            )
            return None
        if arm.joint_state is None:
            self.node.get_logger().error(f"{action_name}: no joint state for Fairino Cartesian planner.")
            return None

        req = GetCartesianPath.Request()
        req.header = target_pose.header
        req.start_state.joint_state = arm.joint_state
        req.group_name = getattr(arm, "group_name", "robot_arm")
        req.link_name = arm.end_effector_name
        req.waypoints = [target_pose.pose]
        req.max_step = 0.0025
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        future = self._fairino_cartesian_client.call_async(req)
        rate = self.node.create_rate(200)
        while not future.done():
            if self._aborted():
                return None
            rate.sleep()
        res = future.result()
        if res is None:
            self.node.get_logger().error(f"{action_name}: Fairino Cartesian service returned None.")
            return None
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().warn(
                f"{action_name}: Fairino Cartesian planner failed: fraction={res.fraction:.3f}, "
                f"error={res.error_code.val}."
            )
            return None
        if res.fraction < float(fraction_threshold):
            self.node.get_logger().warn(
                f"{action_name}: Fairino Cartesian planner completed {res.fraction:.3f}, "
                f"less than threshold {fraction_threshold:.3f}."
            )
            return None
        if not self._fairino_cartesian_start_is_valid(res.solution.joint_trajectory, arm.joint_state, action_name):
            return None
        return res.solution.joint_trajectory

    def _fairino_cartesian_start_is_valid(self, trajectory, joint_state, action_name: str) -> bool:
        if not trajectory.points:
            self.node.get_logger().error(f"{action_name}: Fairino Cartesian trajectory has no points.")
            return False
        if not trajectory.points[0].positions:
            self.node.get_logger().error(f"{action_name}: Fairino Cartesian trajectory first point has no positions.")
            return False

        current_by_name = {
            name: joint_state.position[i]
            for i, name in enumerate(joint_state.name)
            if i < len(joint_state.position)
        }
        max_delta = 0.0
        max_joint = ""
        for i, name in enumerate(trajectory.joint_names):
            if i >= len(trajectory.points[0].positions) or name not in current_by_name:
                self.node.get_logger().error(
                    f"{action_name}: Fairino Cartesian start check missing joint '{name}'."
                )
                return False
            delta = abs(float(trajectory.points[0].positions[i]) - float(current_by_name[name]))
            if delta > max_delta:
                max_delta = delta
                max_joint = name
        if max_delta > self._fairino_cartesian_start_guard_tol:
            self.node.get_logger().error(
                f"{action_name}: Fairino Cartesian trajectory start mismatch: "
                f"joint={max_joint}, delta={max_delta:.6f} rad, "
                f"tol={self._fairino_cartesian_start_guard_tol:.6f} rad."
            )
            return False
        return True

    def _pick_path(self, paths, cartesian: bool, action_name: str):
        if cartesian:
            if len(paths) > 1:
                self.node.get_logger().warn(
                    f"{action_name}: cartesian=True uses direct path mode; "
                    f"received {len(paths)} candidates, using first and skipping scoring."
                )
            return paths[0]
        if self.select_best_path is None or len(paths) == 1:
            return paths[0]
        try:
            best_path, best_score = self.select_best_path(
                paths,
                wrist_weight=self.score_cfg.wrist_weight,
                wrist_joint_indices=self.score_cfg.wrist_joint_indices,
                return_score=True,
            )
            if best_score is not None:
                self.node.get_logger().info(
                    f"{action_name}: selected best trajectory from {len(paths)} candidates: "
                    f"cost={best_score.total_cost:.4f}, path={best_score.path_length:.4f}, "
                    f"wrist={best_score.wrist_length:.4f}, max_step={best_score.max_joint_step:.4f}, "
                    f"smooth={best_score.smoothness:.4f}, duration={best_score.duration:.3f}, "
                    f"valid={best_score.valid}"
                )
                if not best_score.valid and best_score.reason:
                    self.node.get_logger().warn(f"{action_name}: best trajectory score warning: {best_score.reason}")
            return best_path if best_path is not None else paths[0]
        except TypeError:
            return self.select_best_path(
                paths,
                wrist_weight=self.score_cfg.wrist_weight,
                wrist_joint_indices=self.score_cfg.wrist_joint_indices,
            )

    def _wait(self, moveit_obj, action_name: str, timeout_sec: float) -> bool:
        if self.abort is not None:
            return self.abort.wait_idle_or_abort(moveit_obj, action_name, timeout_sec=float(timeout_sec))
        time.sleep(min(float(timeout_sec), 0.5))
        return True

    def _aborted(self) -> bool:
        if self.abort is None:
            return False
        if hasattr(self.abort, "is_blocked"):
            return bool(self.abort.is_blocked())
        return bool(self.abort.is_set())

    def _client_name(self, arm) -> str:
        for key, value in self.arm_clients.items():
            if value is arm:
                return key
        return "unknown"
