#!/usr/bin/env python3
"""Interactive terminal client for the LLM robot control server."""

from __future__ import annotations

import getpass
import json
import readline  # noqa: F401  # Enable Unicode-aware input editing.
import select
import sys
import termios
import threading
import time
import tty
import uuid

from action_msgs.msg import GoalStatus
from llm_arm_control.action import ExecutePreview
from llm_arm_control.srv import PreviewCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from myrobot_common.nodes.motion_control_node import (
    motion_command_for_key,
    trajectory_event_for_command,
)

from llm_arm_control_nodes import deepseek_credentials


HELP = """Commands:
  <natural language>  create a task preview
  y / n               execute or discard the pending preview
  mode yolo|graspnet  select LLM-YOLO or GraspNet (mode status shows both states)
  g                   start GraspNet only while it is in WAIT_G
  h                   stop, open gripper, and return to pregrasp immediately
  r                   clear stop state; never resumes a cancelled task
  clear               clear the 10-turn language session
  status              show server state
  key set|status|delete  run this at the llm-control> prompt, not the Linux shell
  help / quit
During execution: SPACE stops immediately; h stops and returns to pregrasp with the gripper closed.
"""


def compact_preview(preview_json: str) -> str:
    """Render only the operator-facing object poses; keep service JSON untouched."""
    try:
        preview = json.loads(preview_json)
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(preview, dict):
        return ""
    detections = {
        item.get("index"): item for item in preview.get("detections", [])
        if isinstance(item, dict)
    }

    def describe(role, detection):
        if not isinstance(detection, dict):
            return ""
        xyz = detection.get("base_xyz")
        yaw = detection.get("yaw")
        if not isinstance(xyz, list) or len(xyz) != 3 or not isinstance(yaw, (int, float)):
            return ""
        return (
            f"{role} {detection.get('class_name', 'object')}: base_link "
            f"x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}, yaw={yaw:.3f} rad"
        )

    lines = []
    for action in preview.get("actions", []):
        if not isinstance(action, dict):
            continue
        action_type = action.get("type")
        source = detections.get(action.get("source_index"))
        destination = detections.get(action.get("destination_index"))
        if action_type == "pick":
            lines.append(describe("pick", source))
        elif action_type == "pick_place":
            lines.extend((describe("pick", source), describe("box", destination)))
        elif action_type == "place":
            held = next(
                (item for index, item in detections.items() if index != action.get("destination_index")),
                None,
            )
            if held is not None:
                lines.append(describe("holding", held))
            lines.append(describe("box", destination))
    return "\n".join(line for line in lines if line)


def execution_key_effect(key: str) -> tuple[str, bool]:
    """Return the safety command and whether the ROS action should be cancelled."""
    if key == "\x03":
        return "stop", True
    command = motion_command_for_key(key)
    if command == "stop":
        return command, True
    if command == "reset":
        # Reset owns the complete stop -> open -> pregrasp recovery. Sending a
        # later action cancel would race that recovery and turn it into stop.
        return command, False
    if command == "g":
        return "", False
    return command, False


class LlmControlCli(Node):
    def __init__(self):
        super().__init__("llm_control_cli")
        self.preview_client = self.create_client(PreviewCommand, "/llm_control/preview_command")
        self.status_client = self.create_client(Trigger, "/llm_control/status")
        self.execute_client = ActionClient(self, ExecutePreview, "/llm_control/execute_preview")
        self.abort_pub = self.create_publisher(Bool, "/manual_abort", 10)
        self.command_pub = self.create_publisher(String, "/motion_control/command", 10)
        self.event_pub = self.create_publisher(String, "/trajectory_execution_event", 1)
        mode_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.mode_pub = self.create_publisher(String, "/llm_control/active_mode", mode_qos)
        self.active_mode = "yolo"
        self.command_burst_count = int(self.declare_parameter("command_burst_count", 1).value)
        self.command_burst_period_sec = float(
            self.declare_parameter("command_burst_period_sec", 0.01).value
        )
        self.command_sub = self.create_subscription(
            String, "/motion_control/command", self._relay_command, 10
        )
        self.clear_session_pub = self.create_publisher(String, "/llm_control/clear_session", 10)
        self.session_id = uuid.uuid4().hex
        self.preview_id = ""
        self._last_feedback = ""
        self.mode_pub.publish(String(data=self.active_mode))

    @staticmethod
    def _wait_future(future, timeout_sec=None):
        deadline = None if timeout_sec is None else time.monotonic() + float(timeout_sec)
        while not future.done():
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.02)
        return future.result()

    def preview(self, instruction: str):
        try:
            if not self.preview_client.wait_for_service(timeout_sec=2.0):
                print("Preview service is unavailable. Is llm_robot_control_sim.launch.py running?")
                return
            req = PreviewCommand.Request(session_id=self.session_id, instruction=instruction)
            response = self._wait_future(
                self.preview_client.call_async(req), timeout_sec=45.0
            )
        except KeyboardInterrupt:
            print("\nPreview cancelled. The CLI is still active.")
            return
        except Exception as exc:
            print(f"Preview request failed: {exc}")
            print("The CLI is still active; enter `key set` at the next llm-control> prompt if needed.")
            return
        if response is None:
            print("Preview request timed out.")
            return
        self.preview_id = response.preview_id if response.accepted else ""
        details = compact_preview(response.preview_json)
        if details:
            print(details)
        print(response.message)
        if (
            not response.accepted
            and deepseek_credentials.MISSING_MESSAGE in str(response.message)
        ):
            print("Enter `key set` at this llm-control> prompt, then paste the key when asked.")
            print("Do not run `key set` at the Linux robot@...$ shell.")
        if response.accepted:
            print("Execute this complete plan? [y/N]")

    def _feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        text = f"[{feedback.step_index}/{feedback.step_count}] {feedback.phase}: {feedback.message}"
        if text != self._last_feedback:
            print(text, flush=True)
            self._last_feedback = text

    def _publish_command(self, command: str):
        count = max(1, self.command_burst_count)
        for index in range(count):
            self.command_pub.publish(String(data=command))
            if index + 1 < count and self.command_burst_period_sec > 0.0:
                time.sleep(self.command_burst_period_sec)

    def _relay_command(self, msg):
        command = str(msg.data).strip().lower()
        if command == "g":
            return
        event = trajectory_event_for_command(command)
        if event is not None:
            self.event_pub.publish(String(data=event))
        if command == "stop":
            self.abort_pub.publish(Bool(data=True))

    def _read_execution_key(self, stream):
        try:
            readable, _, _ = select.select([stream], [], [], 0.0)
        except (OSError, ValueError):
            return ""
        if not readable:
            return False
        key = stream.read(1)
        command, cancel_action = execution_key_effect(key)
        if command == "stop":
            self._publish_command(command)
            print("\nSTOP requested.", flush=True)
        elif command == "reset":
            self._publish_command(command)
            print("\nHOME reset requested.", flush=True)
        return cancel_action

    def execute_pending(self):
        if not self.preview_id:
            print("No executable preview. Enter an instruction first.")
            return
        if not self.execute_client.wait_for_server(timeout_sec=2.0):
            print("Execute action is unavailable.")
            return
        goal = ExecutePreview.Goal(session_id=self.session_id, preview_id=self.preview_id)
        goal_handle = self._wait_future(
            self.execute_client.send_goal_async(goal, feedback_callback=self._feedback),
            timeout_sec=5.0,
        )
        if goal_handle is None or not goal_handle.accepted:
            print("Task execution was rejected.")
            self.preview_id = ""
            return

        result_future = goal_handle.get_result_async()
        stream = sys.stdin
        old_settings = None
        try:
            if stream.isatty():
                old_settings = termios.tcgetattr(stream.fileno())
                tty.setcbreak(stream.fileno())
            while not result_future.done():
                cancel_action = self._read_execution_key(stream) if old_settings is not None else False
                if cancel_action:
                    goal_handle.cancel_goal_async()
                time.sleep(0.02)
        finally:
            if old_settings is not None:
                termios.tcsetattr(stream.fileno(), termios.TCSADRAIN, old_settings)

        wrapped = result_future.result()
        self.preview_id = ""
        if wrapped is None:
            print("Task ended without a result.")
            return
        result = wrapped.result
        ok = result.success and wrapped.status == GoalStatus.STATUS_SUCCEEDED
        print(f"{'SUCCESS' if ok else 'FAILED'} [{result.terminal_state}]: {result.message}")

    def show_status(self):
        if not self.status_client.wait_for_service(timeout_sec=1.0):
            print("Status service is unavailable.")
            return
        response = self._wait_future(self.status_client.call_async(Trigger.Request()), timeout_sec=2.0)
        print(response.message if response is not None else "Status request timed out.")

    def set_mode(self, words):
        requested = words[1].lower() if len(words) > 1 else "status"
        if requested == "status":
            status = self._server_status()
            print(f"mode={self.active_mode}, yolo_state={status.get('state', 'unknown')}, graspnet_state={status.get('graspnet_state', 'unknown')}")
            return
        if requested not in ("yolo", "graspnet"):
            print("Use: mode yolo | mode graspnet | mode status")
            return
        if requested == self.active_mode:
            print(f"Already in {requested} mode.")
            return
        status = self._server_status()
        if requested == "graspnet" and status.get("state") not in ("IDLE", "HOLDING"):
            print("YOLO mode can switch only when /llm_control/status reports IDLE or HOLDING.")
            return
        if requested == "yolo" and status.get("graspnet_state") != "WAIT_G":
            print("GraspNet mode can switch only when /llm_control/status reports WAIT_G.")
            return
        self.preview_id = ""
        self.mode_pub.publish(String(data=requested))
        deadline = time.monotonic() + 75.0
        while time.monotonic() < deadline:
            confirmed = self._server_status()
            if confirmed.get("active_mode") == requested:
                self.active_mode = requested
                print(f"Switched to {requested} mode.")
                return
            error = confirmed.get("mode_switch_error")
            if error:
                print(f"Mode switch failed: {error}")
                return
            time.sleep(0.05)
        print("Mode switch timed out; run `mode status` for the server state.")

    def _server_state(self):
        return str(self._server_status().get("state", "unknown"))

    def _server_status(self):
        if not self.status_client.wait_for_service(timeout_sec=0.5):
            return {}
        response = self._wait_future(self.status_client.call_async(Trigger.Request()), timeout_sec=1.0)
        try:
            return json.loads(response.message)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return {}

    def key_command(self, words):
        subcommand = words[1].lower() if len(words) > 1 else "status"
        try:
            if subcommand == "status":
                print(f"DeepSeek credential source: {deepseek_credentials.credential_status()}")
            elif subcommand == "set":
                try:
                    api_key = getpass.getpass("DeepSeek API key: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nKey entry cancelled; the CLI is still active.")
                    return
                deepseek_credentials.set_deepseek_api_key(api_key)
                print("DeepSeek API key saved to GNOME Keyring.")
            elif subcommand == "delete":
                deleted = deepseek_credentials.delete_deepseek_api_key()
                if deleted:
                    print("Key deleted. Enter `key set` here before the next LLM request.")
                else:
                    print("No Keyring key was configured.")
            else:
                print("Use: key set | key status | key delete")
        except (ValueError, deepseek_credentials.DeepSeekCredentialError) as exc:
            print(f"Credential error: {exc}")

    def run(self):
        print(HELP)
        while rclpy.ok():
            try:
                line = input("llm-control> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\nCLI remains active; enter `quit` to exit.")
                continue
            if not line:
                continue
            words = line.split()
            command = words[0].lower()
            if command in ("quit", "exit"):
                break
            if command == "help":
                print(HELP)
            elif command == "key":
                self.key_command(words)
            elif command == "clear":
                self.clear_session_pub.publish(String(data=self.session_id))
                self.session_id = uuid.uuid4().hex
                self.preview_id = ""
                print("Language context cleared.")
            elif command == "status":
                self.show_status()
            elif command == "mode":
                self.set_mode(words)
            elif command == "h":
                self._publish_command("reset")
                print("Pregrasp reset requested.")
            elif command == "r":
                self._publish_command("resume")
                print("Stop state clear requested; cancelled trajectories will not resume.")
            elif command == "g":
                if self.active_mode != "graspnet":
                    print("g is available only in GraspNet mode.")
                elif self._server_status().get("holding"):
                    print("GraspNet cannot start while an object is held; place it or press h first.")
                elif self._server_status().get("graspnet_request_pending"):
                    print("GraspNet request already pending.")
                elif self._server_status().get("graspnet_state") != "WAIT_G":
                    print("GraspNet is not ready for g.")
                else:
                    self._publish_command("g")
                    print("GraspNet computation requested.")
            elif command == "y":
                self.execute_pending()
            elif command == "n":
                self.preview_id = ""
                print("Preview discarded.")
            else:
                self.preview(line)


def main(args=None):
    rclpy.init(args=args)
    node = LlmControlCli()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        node.run()
    finally:
        executor.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
