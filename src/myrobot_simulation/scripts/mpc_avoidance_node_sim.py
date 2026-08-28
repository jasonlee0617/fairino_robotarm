#!/usr/bin/env python3
"""Gazebo demo: Tube-BiRRT reference trajectories tracked by MPC avoidance."""

import math
import threading
import time
import xml.etree.ElementTree as ET
from enum import Enum, auto

import rclpy
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from pymoveit2 import MoveIt2
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from myrobot_common.planning.motion_executor import MoveItMotion, PlannerSwitch
from myrobot_common.utils.params import param
from myrobot_common.utils.pose_tools import PoseTools


class DemoStage(Enum):
    WAIT_RUNTIME = auto()
    MOVE_BOX_ABOVE = auto()
    START_MPC = auto()
    WAIT_MPC = auto()
    COMPLETE = auto()
    FAILED = auto()


class MPCAvoidanceDemoNode(Node):
    """Owns the MPC-specific reference and status loop, not generic MoveIt plumbing."""

    def __init__(self):
        super().__init__(
            "mpc_avoidance_demo_node",
            automatically_declare_parameters_from_overrides=True,
        )
        self.callback_group = ReentrantCallbackGroup()
        self._load_parameters()
        self._assert_group_exists_in_srdf()

        self.pose_tools = PoseTools(self, base_frame=self.base_frame)
        self.targets = {
            "box_above": {
                "pose": self.pose_tools.make_pose(0.420, 0.330, 0.200, 0.0, 180.0, 0.0),
                "description": "box上方",
            },
            "case_place": {
                "pose": self.pose_tools.make_pose(0.350, -0.330, 0.120, 0.0, 180.0, 0.0),
                "description": "放置位置",
            },
        }

        self.arms = {
            "fairino": self._make_arm_client(self.move_group_ns_fairino),
            "kdl": self._make_arm_client(self.move_group_ns_kdl),
        }
        self.motion = MoveItMotion(
            node=self,
            arm_clients=self.arms,
            default_client=self.ik_plugin,
            pose_tools=self.pose_tools,
            action_delay=0.0,
        )
        if not self.motion.set_planner(self.planning_pipeline_id, self.planner_id):
            raise RuntimeError("Unsupported planner configuration")

        self.current_joints = None
        self.mpc_reached_goal = False
        self.replan_requested = False
        self.replan_in_flight = False
        self.mpc_blocked_reason = None
        self.replan_count = 0
        self.stage = DemoStage.WAIT_RUNTIME
        self._controller_topic_warned = False

        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
            callback_group=self.callback_group,
        )
        status_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            "/mpc_status",
            self._on_mpc_status,
            status_qos,
            callback_group=self.callback_group,
        )
        self.ref_traj_pub = self.create_publisher(JointTrajectory, "/planned_trajectory", 10)
        self.mpc_cmd_pub = self.create_publisher(String, "/mpc_command", 10)
        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            f"{self.move_group_namespace}/execute_trajectory",
            callback_group=self.callback_group,
        )
        self.create_timer(2.0, self._check_controller_topic_ready)

        self.get_logger().info(
            "MPC demo configured: "
            f"profile={self.robot_profile}, client={self.ik_plugin}, "
            f"namespace={self.move_group_namespace}, pipeline={self.planning_pipeline_id}, "
            f"planner={self.planner_id}"
        )

    def _load_parameters(self):
        self.robot_profile = str(
            param(self, "robot_profile", "fairino_arm_gripper_onbase")
        ).strip()
        self.group_name = str(param(self, "group_name", "robot_arm")).strip()
        self.ee_link = str(param(self, "ee_link", "tool0")).strip()
        self.base_frame = str(param(self, "base_frame", "base_link")).strip()
        self.joint_names = [
            str(name)
            for name in param(self, "joint_names", ["j1", "j2", "j3", "j4", "j5", "j6"])
        ]
        self.controller_topic = str(
            param(self, "controller_topic", "/robot_arm_controller/joint_trajectory")
        ).strip()
        self.move_group_ns_fairino = str(
            param(self, "move_group_ns_fairino", "/move_group_fairino")
        ).rstrip("/")
        self.move_group_ns_kdl = str(
            param(self, "move_group_ns_kdl", "/move_group_kdl")
        ).rstrip("/")
        self.startup_timeout_sec = float(param(self, "startup_timeout_sec", 60.0))
        self.planning_attempts = int(param(self, "planning_attempts", 5))
        self.allowed_planning_time = float(param(self, "allowed_planning_time", 15.0))
        self.position_tolerance = float(param(self, "position_tolerance", 0.005))
        self.orientation_tolerance = float(param(self, "orientation_tolerance", 0.05))
        self.max_velocity = float(param(self, "max_velocity", 1.0))
        self.max_acceleration = float(param(self, "max_acceleration", 1.0))
        self.robot_description_semantic = str(
            param(self, "robot_description_semantic", "") or ""
        ).strip()

        self.ik_plugin = PlannerSwitch.normalize_ik(str(param(self, "ik_plugin", "fairino")))
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline(
            str(param(self, "planning_pipeline_id", "fairino"))
        )
        self.planner_id = PlannerSwitch.normalize_planner(
            self.planning_pipeline_id,
            str(param(self, "planner_id", "tube_birrt*")),
        )
        if self.ik_plugin not in ("fairino", "kdl"):
            raise RuntimeError(f"Unsupported ik_plugin: {self.ik_plugin}")
        if not PlannerSwitch.is_valid(self.planning_pipeline_id, self.planner_id):
            raise RuntimeError(
                "Unsupported planner: "
                f"pipeline={self.planning_pipeline_id}, planner={self.planner_id}"
            )
        if not all((self.group_name, self.ee_link, self.base_frame)) or not self.joint_names:
            raise RuntimeError("group_name, ee_link, base_frame and joint_names must be set")

        namespaces = {"fairino": self.move_group_ns_fairino, "kdl": self.move_group_ns_kdl}
        self.move_group_namespace = namespaces[self.ik_plugin]

    def _assert_group_exists_in_srdf(self):
        if not self.robot_description_semantic:
            self.get_logger().warn(
                "robot_description_semantic is absent; skipping SRDF group validation"
            )
            return
        try:
            groups = {
                group.attrib["name"].strip()
                for group in ET.fromstring(self.robot_description_semantic).findall("group")
                if group.attrib.get("name")
            }
        except ET.ParseError as exc:
            raise RuntimeError(f"robot_description_semantic parse failed: {exc}") from exc
        if self.group_name not in groups:
            raise RuntimeError(f"group_name='{self.group_name}' is not in SRDF: {sorted(groups)}")

    def _make_arm_client(self, namespace):
        arm = MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_frame,
            end_effector_name=self.ee_link,
            group_name=self.group_name,
            ignore_new_calls_while_executing=False,
            callback_group=self.callback_group,
            move_group_namespace=namespace,
        )
        arm.num_planning_attempts = self.planning_attempts
        arm.allowed_planning_time = self.allowed_planning_time
        arm.max_velocity = self.max_velocity
        arm.max_acceleration = self.max_acceleration
        return arm

    def _check_controller_topic_ready(self):
        if self._controller_topic_warned:
            return
        topics = {name for name, _ in self.get_topic_names_and_types()}
        if self.controller_topic not in topics:
            self.get_logger().warn(f"Controller topic is not visible yet: {self.controller_topic}")
            self._controller_topic_warned = True

    def _on_joint_state(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in self.joint_names):
            self.current_joints = [float(positions[name]) for name in self.joint_names]

    def _on_mpc_status(self, msg):
        status = str(msg.data)
        if status == "REACHED":
            self.mpc_reached_goal = True
        elif status == "REPLAN_REQUIRED":
            if self.replan_count >= 1 or self.replan_in_flight:
                self.get_logger().warn("Ignoring duplicate MPC replan request")
            else:
                self.replan_requested = True
        elif status == "WAITING_DYNAMIC_CLEARANCE":
            self.get_logger().warn("MPC is safely waiting for dynamic clearance")
        elif status.startswith("BLOCKED_"):
            self.mpc_blocked_reason = status
            self.get_logger().error(f"MPC safely stopped: {status}")

    def _set_stage(self, stage):
        self.stage = stage
        self.get_logger().info(f"MPC demo stage: {stage.name}")

    def _wait_for_runtime_ready(self):
        deadline = time.monotonic() + self.startup_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            plan_ready = self.motion.wait_client_ready(self.ik_plugin, timeout_sec=0.0)
            execute_ready = self.execute_client.wait_for_server(timeout_sec=0.0)
            if self.current_joints is not None and plan_ready and execute_ready:
                return True
            time.sleep(0.1)
        self.get_logger().error(
            "MPC demo startup timed out waiting for joint state, planner or executor"
        )
        return False

    def _current_start_state(self):
        if self.current_joints is None:
            return None
        return JointState(name=list(self.joint_names), position=list(self.current_joints))

    def _plan_reference(self, target_name, *, require_two_points=False):
        target = self.targets[target_name]
        trajectory = self.motion.plan_to_pose(
            target["pose"],
            planning_client=self.ik_plugin,
            action_name=f"Tube-BiRRT reference to {target['description']}",
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration,
            allowed_planning_time=self.allowed_planning_time,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            start_joint_state=self._current_start_state(),
            joint_constraint=False,
        )
        if trajectory is None or (require_two_points and len(trajectory.points) < 2):
            self.get_logger().error(f"No valid reference trajectory for {target['description']}")
            return None
        self.get_logger().info(
            f"Reference planned for {target['description']}: {len(trajectory.points)} points"
        )
        return trajectory

    @staticmethod
    def _wait_future(future, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        return future.result() if future.done() else None

    def _execute_direct(self, trajectory, target_name, timeout_sec):
        if not self.execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveIt ExecuteTrajectory action is unavailable")
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = trajectory
        handle = self._wait_future(self.execute_client.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            self.get_logger().error("MoveIt rejected the direct trajectory")
            return False
        result = self._wait_future(handle.get_result_async(), timeout_sec)
        if result is None:
            self.get_logger().error("Direct trajectory execution timed out")
            return False
        code = result.result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            return True
        if code in (MoveItErrorCodes.PREEMPTED, MoveItErrorCodes.TIMED_OUT) and trajectory.points:
            target_by_joint = dict(zip(trajectory.joint_names, trajectory.points[-1].positions))
            final_joints = [target_by_joint.get(name) for name in self.joint_names]
            deadline = time.monotonic() + timeout_sec
            while (
                rclpy.ok()
                and time.monotonic() < deadline
                and all(value is not None for value in final_joints)
            ):
                if self.current_joints is not None:
                    max_error = max(abs(a - b) for a, b in zip(self.current_joints, final_joints))
                    if max_error <= math.radians(3.0):
                        self.get_logger().warn(
                            f"{target_name}: MoveIt returned {code}, but final joint error is "
                            f"{math.degrees(max_error):.2f} deg; treating as complete"
                        )
                        return True
                time.sleep(0.05)
        self.get_logger().error(f"Direct trajectory failed: MoveIt error {code}")
        return False

    def _publish_mpc_reference(self, trajectory, *, start):
        self.ref_traj_pub.publish(trajectory)
        if start:
            self.mpc_cmd_pub.publish(String(data="START"))

    def _start_mpc(self, target_name):
        trajectory = self._plan_reference(target_name, require_two_points=True)
        if trajectory is None:
            return False
        self.mpc_reached_goal = False
        self.replan_requested = False
        self.replan_in_flight = False
        self.mpc_blocked_reason = None
        self.replan_count = 0
        self._publish_mpc_reference(trajectory, start=True)
        self.get_logger().info(f"Published MPC reference ({len(trajectory.points)} points)")
        return True

    def _replan_mpc_reference(self):
        if self.replan_in_flight or self.replan_count >= 1:
            self.get_logger().error("MPC replan budget is exhausted")
            return False
        self.replan_in_flight = True
        self.replan_count += 1
        try:
            trajectory = self._plan_reference("case_place", require_two_points=True)
            if trajectory is None:
                return False
            self._publish_mpc_reference(trajectory, start=False)
            self.get_logger().info(
                f"Published MPC replan reference ({len(trajectory.points)} points)"
            )
            return True
        finally:
            self.replan_in_flight = False

    def _wait_for_mpc(self, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        last_log = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            if self.mpc_reached_goal:
                return True
            if self.mpc_blocked_reason is not None:
                self.get_logger().error(f"MPC stopped: {self.mpc_blocked_reason}")
                return False
            if self.replan_requested:
                self.replan_requested = False
                if not self._replan_mpc_reference():
                    return False
            now = time.monotonic()
            if now - last_log >= 5.0:
                self.get_logger().info("MPC tracking dynamic obstacles")
                last_log = now
            time.sleep(0.05)
        self.get_logger().error(f"MPC timed out after {timeout_sec:.1f}s")
        return False

    def _stop_mpc(self):
        if rclpy.ok():
            self.mpc_cmd_pub.publish(String(data="STOP"))

    def run_demo(self):
        self._set_stage(DemoStage.WAIT_RUNTIME)
        if not self._wait_for_runtime_ready():
            self._set_stage(DemoStage.FAILED)
            return

        self._set_stage(DemoStage.MOVE_BOX_ABOVE)
        trajectory = self._plan_reference("box_above")
        if trajectory is None or not self._execute_direct(trajectory, "box_above", 30.0):
            self._set_stage(DemoStage.FAILED)
            return

        self._set_stage(DemoStage.START_MPC)
        if not self._start_mpc("case_place"):
            self._set_stage(DemoStage.FAILED)
            return

        self._set_stage(DemoStage.WAIT_MPC)
        if self._wait_for_mpc(300.0):
            self._set_stage(DemoStage.COMPLETE)
            self.get_logger().info("MPC dynamic-avoidance demo completed")
            return

        self._stop_mpc()
        self._set_stage(DemoStage.FAILED)


def main(args=None):
    rclpy.init(args=args)
    node = MPCAvoidanceDemoNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    worker = threading.Thread(target=node.run_demo, daemon=True)
    worker.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_mpc()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
