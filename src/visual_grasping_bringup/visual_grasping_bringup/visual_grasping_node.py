#!/usr/bin/env python3
import time

from typing import Sequence

import rclpy
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import PointStamped, Vector3Stamped
from pymoveit2 import MoveIt2
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from myrobot_common.perception.detection_cache import DetectionCache
from myrobot_common.perception.target_selector import TargetSelector
from myrobot_common.planning.motion_executor import MoveItMotion, PlanScoreConfig, PlannerSwitch
from myrobot_common.planning.trajectory_scoring import select_best_path
from myrobot_common.task.abort_manager import AbortManager
from myrobot_common.utils.params import param
from myrobot_common.utils.pose_tools import PoseTools
from myrobot_common.utils.tf_tools import TfTools
from visual_grasping_bringup.task.grasp_profile import load_grasp_profiles
from visual_grasping_bringup.task.visual_grasping_state_machine import VisualGraspingStateMachine
from visual_grasping_bringup.task.task_types import TaskState


class VisualGraspingNode(Node):
    def __init__(self):
        super().__init__(
            "visual_grasping",
            automatically_declare_parameters_from_overrides=True,
        )

        self.callback_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.abort_cb_group = MutuallyExclusiveCallbackGroup()
        self._startup_ready_logged = False
        self._controller_manager_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
            callback_group=self.callback_group,
        )

        self._load_params()

        self.tf_tools = TfTools(self, base_frame=self.base_frame, camera_frame=self.camera_frame)
        self.pose_tools = PoseTools(self, base_frame=self.base_frame)
        self.pregrasp_pose = self._build_pregrasp_pose()
        self.det_cache = DetectionCache(self)
        self.target_selector = TargetSelector(self, self.target_priority)
        self.target_selector.set_preference(self.preferred_target)
        self.target_selector.set_timeout(self.detection_timeout)

        self._setup_detection_subscribers()
        self.yolo_inference_client = self.create_client(
            SetBool, "/yolo_detector_obb/set_inference_enabled", callback_group=self.callback_group
        )
        self._setup_moveit()

        self.abort = AbortManager(self, arm=self.moveit2_arm, gripper=self.moveit2_gripper)
        self.abort.set_recovery_hooks(
            open_gripper_fn=lambda: self.control_gripper(True),
            close_gripper_fn=lambda: self.control_gripper(False),
            go_home_fn=self.move_to_pregrasp_pose,
            reset_fn=self._reset_task_cache,
            restore_arm_limits_fn=self._restore_arm_limits,
            recovery_complete_fn=self._recovery_complete,
        )
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

        self.current_state = TaskState.IDLE
        self.active_target = None
        self.active_target_yaw = None
        self.poses = {}
        self._g_requested = False
        self.state_machine = VisualGraspingStateMachine(self)

        self.create_subscription(
            Bool,
            "/manual_abort",
            self.abort.on_manual_abort,
            10,
            callback_group=self.abort_cb_group,
        )
        self.create_subscription(
            String,
            "/visual_grasping/planner_command",
            self._on_planner_command,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            "/motion_control/command",
            self._on_motion_command,
            10,
            callback_group=self.callback_group,
        )
        self.state_publisher = self.create_publisher(String, "/task_state", 10)
        self.create_timer(self.control_period_sec, self.control_loop, callback_group=self.control_cb_group)

        self.get_logger().info("VisualGraspingNode initialized")

    def _load_params(self):
        self.arm_group_name = str(param(self, "arm_group_name", "robot_arm"))
        self.hand_group_name = str(param(self, "hand_group_name", "hand"))
        self.base_frame = str(param(self, "base_frame", "base_link"))
        self.camera_frame = str(param(self, "camera_frame", "camera_color_optical_frame"))
        self.ee_frame = str(param(self, "ee_frame", "tool0"))
        self.move_group_ns_fairino = str(param(self, "move_group_ns_fairino", "/move_group_fairino"))
        self.move_group_ns_kdl = str(param(self, "move_group_ns_kdl", "/move_group_kdl"))
        self.move_group_ready_timeout_sec = float(param(self, "move_group_ready_timeout_sec", 10.0))
        self.allow_cross_client_fallback = bool(param(self, "allow_cross_client_fallback", True))

        self.ik_plugin = PlannerSwitch.normalize_ik(str(param(self, "ik_plugin", "fairino")))
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(str(param(self, "planning_pipeline_id", "fairino")))
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            str(param(self, "planner_id", "tube_birrt*")),
        )
        if not PlannerSwitch.is_valid(self.planning_pipeline_id, self.planner_id):
            raise ValueError(
                f"Unsupported planner config: pipeline={self.planning_pipeline_id}, "
                f"planner={self.planner_id}"
            )
        self.max_step_size = float(param(self, "max_step_size", 0.05))
        self.arm_max_velocity = float(param(self, "arm_max_velocity", 0.3))
        self.arm_max_acceleration = float(param(self, "arm_max_acceleration", 0.3))
        self.allowed_planning_time = float(param(self, "allowed_planning_time", 15.0))
        self.position_tolerance = float(param(self, "position_tolerance", 0.005))
        self.orientation_tolerance = float(param(self, "orientation_tolerance", 0.005))
        self.allowed_start_tolerance = float(param(self, "allowed_start_tolerance", 0.1))

        self.preferred_target = str(
            param(self, "preferred_target", "elongated_object")
        ).lower().strip()
        self.target_priority = self._string_list_param(
            "target_priority", ["elongated_object", "cube", "stone"]
        )
        self.grasp_above = float(param(self, "grasp_above", 0.04))
        self.grasp_offset = float(param(self, "grasp_offset", 0.008))
        self.place_offset = float(param(self, "place_offset", 0.20))
        self.descend_to_box = float(param(self, "descend_to_box", 0.07))
        self.action_delay = float(param(self, "action_delay", 0.5))
        self.detection_timeout = float(param(self, "detection_timeout", 3.0))
        self.control_period_sec = float(param(self, "control_period_sec", 0.2))
        self.use_continuous_yolo = bool(param(self, "use_continuous_yolo", True))
        self._yolo_inference_enabled = self.use_continuous_yolo

        self.pregrasp_pose_cfg = {
            "x": self._compat_float_param("pregrasp_pose.x", "home_pose.x", 0.149),
            "y": self._compat_float_param("pregrasp_pose.y", "home_pose.y", 0.327),
            "z": self._compat_float_param("pregrasp_pose.z", "home_pose.z", 0.364),
            "roll": self._compat_float_param("pregrasp_pose.roll", "home_pose.roll", -174.091),
            "pitch": self._compat_float_param("pregrasp_pose.pitch", "home_pose.pitch", 1.040),
            "yaw": self._compat_float_param("pregrasp_pose.yaw", "home_pose.yaw", -50.0),
        }

        self.num_candidate_plans = int(param(self, "num_candidate_plans", 5))
        self.wrist_weight = float(param(self, "wrist_weight", 50.0))
        self.wrist_joint_indices = tuple(int(v) for v in param(self, "wrist_joint_indices", [2, 3, 4]))
        self.gripper_open_positions = tuple(float(v) for v in param(self, "gripper_open_positions", [0.0305, -0.0305]))
        self.gripper_close_positions = tuple(float(v) for v in param(self, "gripper_close_positions", [0.0, 0.0]))

        self.grasp_profiles = load_grasp_profiles(self)

        self.j2_constraint = {
            "joint_positions": [float(param(self, "j2_constraint_position", -1.5708))],
            "joint_names": ["j2"],
            "tolerance": float(param(self, "j2_constraint_tolerance", 1.5708)),
            "weight": 1.0,
        }
        self.get_logger().info("Params loaded")

    def _setup_detection_subscribers(self):
        qos_reliable_latest = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PointStamped,
            "/elongated_object_position_3d",
            self.det_cache.on_elongated_object_pos,
            qos_reliable_latest,
        )
        self.create_subscription(PointStamped, "/cube_position_3d", self.det_cache.on_cube_pos, qos_reliable_latest)
        self.create_subscription(PointStamped, "/box_position_3d", self.det_cache.on_box_pos, qos_reliable_latest)
        self.create_subscription(PointStamped, "/stone_position_3d", self.det_cache.on_stone_pos, qos_reliable_latest)
        self.create_subscription(Vector3Stamped, "/elongated_object_axis_3d", self.det_cache.on_elongated_object_axis, qos_reliable_latest)
        self.create_subscription(Vector3Stamped, "/cube_axis_3d", self.det_cache.on_cube_axis, qos_reliable_latest)
        self.create_subscription(Vector3Stamped, "/stone_axis_3d", self.det_cache.on_stone_axis, qos_reliable_latest)
        self.get_logger().info("Detection subscribers set")

    def _setup_moveit(self):
        self.moveit2_arm_fairino = self._make_arm_client(self.move_group_ns_fairino)
        self.moveit2_arm_kdl = self._make_arm_client(self.move_group_ns_kdl)
        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            self._configure_arm_planner(arm)

        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            arm.max_step_size = self.max_step_size
            arm.max_velocity = self.arm_max_velocity
            arm.max_acceleration = self.arm_max_acceleration
            arm.allowed_planning_time = self.allowed_planning_time
            arm.position_tolerance = self.position_tolerance
            arm.orientation_tolerance = self.orientation_tolerance
            arm.allowed_start_tolerance = self.allowed_start_tolerance

        self.moveit2_arm = self.moveit2_arm_fairino if self.ik_plugin == "fairino" else self.moveit2_arm_kdl
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
        self.get_logger().info("MoveIt2 initialized")

    def _configure_arm_planner(self, arm):
        arm.pipeline_id = self.planning_pipeline_id
        arm.planner_id = self.planner_id

    def motion_limits_kwargs(self) -> dict:
        return {
            "max_step_size": self.max_step_size,
            "allowed_planning_time": self.allowed_planning_time,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "allowed_start_tolerance": self.allowed_start_tolerance,
        }

    def _compat_float_param(self, name: str, legacy_name: str, default: float) -> float:
        if self.has_parameter(name):
            return float(self.get_parameter(name).value)
        if self.has_parameter(legacy_name):
            return float(self.get_parameter(legacy_name).value)
        self.declare_parameter(name, float(default))
        return float(self.get_parameter(name).value)

    def _make_arm_client(self, namespace: str):
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

    def _on_planner_command(self, msg: String):
        self.motion.handle_command(msg)
        self.ik_plugin = self.motion.current_client
        self.moveit2_arm = self.motion.arm
        self.get_logger().info(f"Active IK/planning client: {self.ik_plugin}")

    def _on_motion_command(self, msg: String):
        if str(msg.data).strip().lower() == "g" and self.current_state == TaskState.WAIT_G:
            self._g_requested = True

    def startup_motion_ready(self, planning_client: str | None = None) -> bool:
        arm = self.motion._select_arm(planning_client)
        arm_ready = self._moveit_service_ready(arm, self.move_group_ready_timeout_sec)
        gripper_ready = self._moveit_service_ready(self.moveit2_gripper, self.move_group_ready_timeout_sec)
        arm_exec_ready = self._action_ready(self.arm_execute_action, self.move_group_ready_timeout_sec)
        gripper_exec_ready = self._action_ready(self.gripper_execute_action, self.move_group_ready_timeout_sec)
        controllers_ready = self._controllers_active(
            ("robot_arm_controller", "hand_controller"),
            self.move_group_ready_timeout_sec,
        )
        ready = arm_ready and gripper_ready and arm_exec_ready and gripper_exec_ready and controllers_ready
        if ready and not self._startup_ready_logged:
            self.get_logger().info("MoveIt services ready for startup motions")
            self._startup_ready_logged = True
        return ready

    def startup_client(self) -> str:
        requested = PlannerSwitch.normalize_ik(self.ik_plugin)
        if self.startup_motion_ready(requested):
            return requested
        if self.allow_cross_client_fallback:
            for candidate in ("fairino", "kdl"):
                if candidate == requested:
                    continue
                if self.startup_motion_ready(candidate):
                    self.get_logger().warn(
                        f"Startup motions fallback from client={requested} to ready client={candidate}"
                    )
                    return candidate
        return requested

    def _reset_task_cache(self):
        self.active_target = None
        self.active_target_yaw = None
        self.poses = {}
        self._g_requested = False
        self.det_cache.reset()

    def _recovery_complete(self, recovered: bool):
        self.set_yolo_inference(False)
        if recovered:
            self.current_state = TaskState.WAIT_G
            self.get_logger().info(
                "YOLO reset complete: 已到达 pregrasp，夹爪已闭合。输入 g 开始一次抓取。"
            )
        else:
            self.current_state = TaskState.ERROR

    def _restore_arm_limits(self):
        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            try:
                arm.max_velocity = self.arm_max_velocity
                arm.max_acceleration = self.arm_max_acceleration
            except Exception:
                pass

    def move_to_pose(self, *args, **kwargs):
        return self.motion.move_to_pose(*args, **kwargs)

    def control_gripper(self, open_gripper=True):
        return self.motion.control_gripper(open_gripper=open_gripper, timeout_sec=90.0)

    def set_yolo_inference(self, enabled: bool) -> bool:
        if self.use_continuous_yolo:
            return True
        enabled = bool(enabled)
        if enabled == self._yolo_inference_enabled:
            return True
        if not self.yolo_inference_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warning("YOLO inference control service is unavailable")
            return False
        future = self.yolo_inference_client.call_async(SetBool.Request(data=enabled))
        deadline = time.monotonic() + 1.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            return False
        self._yolo_inference_enabled = enabled
        if enabled:
            self.det_cache.reset()
        return True

    def move_to_pregrasp_pose(self):
        planning_client = self.startup_client()
        return self.motion.move_to_pose(
            self.pregrasp_pose,
            action_name=f"Move to pre-grasp pose [client={planning_client}]",
            planning_client=planning_client,
            cartesian=False,
            joint_constraint=False,
            max_velocity=self.arm_max_velocity,
            max_acceleration=self.arm_max_acceleration,
            timeout_sec=180.0,
            **self.motion_limits_kwargs(),
        )

    def control_loop(self):
        self.state_machine.tick()

    def _string_list_param(self, name: str, default: Sequence[str]):
        value = param(self, name, list(default))
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def _build_pregrasp_pose(self):
        return self.pose_tools.make_pose(
            self.pregrasp_pose_cfg["x"],
            self.pregrasp_pose_cfg["y"],
            self.pregrasp_pose_cfg["z"],
            self.pregrasp_pose_cfg["roll"],
            self.pregrasp_pose_cfg["pitch"],
            self.pregrasp_pose_cfg["yaw"],
        )


    def _moveit_service_ready(self, moveit_obj, timeout_sec: float) -> bool:
        cli = getattr(moveit_obj, "_plan_kinematic_path_service", None)
        if cli is None:
            return True
        try:
            return bool(cli.wait_for_service(timeout_sec=float(timeout_sec)))
        except Exception:
            return False

    def _action_ready(self, action_client, timeout_sec: float) -> bool:
        try:
            return bool(action_client.wait_for_server(timeout_sec=float(timeout_sec)))
        except Exception:
            return False

    def _controllers_active(self, names: tuple[str, ...], timeout_sec: float) -> bool:
        client = self._controller_manager_client
        try:
            if not client.wait_for_service(timeout_sec=float(timeout_sec)):
                return False
            future = client.call_async(ListControllers.Request())
            deadline = time.time() + float(timeout_sec)
            while rclpy.ok() and not future.done():
                if time.time() >= deadline:
                    return False
                time.sleep(0.05)
            if not future.done() or future.result() is None:
                return False
            states = {c.name: c.state for c in future.result().controller}
            return all(states.get(name) == "active" for name in names)
        except Exception:
            return False


def main(args=None):
    rclpy.init(args=args)
    node = VisualGraspingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
