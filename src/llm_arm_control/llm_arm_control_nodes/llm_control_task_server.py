#!/usr/bin/env python3
"""Local Qwen tool boundary, YOLO RGB-D planner, and robot task server."""

from __future__ import annotations

import json
import signal
import threading
import time
import uuid

from llm_arm_control.action import ExecutePreview
from llm_arm_control.srv import AgentCommand
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_ros

from llm_arm_control_nodes.robot_motion_base import RobotMotionBase
from llm_arm_control_nodes.agent_protocol import AGENT_TOOLS, parse_agent_tool_call
from visual_perception_utils.llm_rgbd import (
    PerceptionUnavailable,
    RgbdPerception,
)
from llm_arm_control_nodes.task_logic import (
    PreviewFailure,
    PreviewRecord,
    SafetyState,
    TaskContext,
    invocation_spoken_text,
    apply_safety_command,
    complete_safety_reset,
    execution_step_count,
    preview_status,
    safety_execution_valid,
    spoken_error_text,
)
from llm_arm_control_nodes.task.llm_control_state_machine import (
    LlmControlTaskState,
    LlmControlTaskStateMachine,
)
from llm_arm_control_nodes.task.visual_task_planner import VisualTaskPlannerMixin


class LlmControlTaskServer(VisualTaskPlannerMixin, RobotMotionBase):
    def __init__(self):
        super().__init__("llm_control_task_server")
        self._declare_task_parameters()
        self._read_task_parameters()
        self._lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._previews: dict[str, PreviewRecord] = {}
        self._task_contexts: dict[str, TaskContext] = {}
        self._tool_results = {}
        self._state = LlmControlTaskState.PREGRASP_POSE.value
        self._safety = SafetyState()
        self._visual_available = True
        self._visual_last_error = ""
        self._execution_active = False
        self._reset_failed = False
        self._held_source = None
        self._shutting_down = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.perception = RgbdPerception(
            self,
            self.tf_buffer,
            base_frame=self.base_frame,
            yolo_topic=self.yolo_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            rgb_depth_tolerance_sec=self.rgb_depth_tolerance_sec,
            detection_max_age_sec=self.detection_max_age_sec,
            vision_wait_timeout_sec=self.vision_wait_timeout_sec,
            tf_wait_timeout_sec=self.tf_wait_timeout_sec,
            callback_group=self.callback_group,
        )
        self.abort.set_command_hook(self._advance_safety)
        self.abort.set_command_enabled(lambda: True)
        self.llm_yolo_status_client = self.create_client(
            Trigger, "/llm_visual_perception/status", callback_group=self.callback_group
        )
        self.clear_session_subscription = self.create_subscription(
            String,
            "/llm_control/clear_session",
            self._clear_session,
            10,
            callback_group=self.callback_group,
        )
        self.agent_service = self.create_service(
            AgentCommand,
            "/llm_control/agent_command",
            self._agent_command,
            callback_group=self.callback_group,
        )
        self.status_service = self.create_service(
            Trigger, "/llm_control/status", self._status, callback_group=self.callback_group
        )
        self.execute_action = ActionServer(
            self,
            ExecutePreview,
            "/llm_control/execute_preview",
            execute_callback=self._execute_preview,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group,
        )
        self.abort.set_recovery_hooks(
            open_gripper_fn=self._open_gripper,
            close_gripper_fn=self._close_gripper,
            go_home_fn=self._move_to_pregrasp_pose,
            recovery_complete_fn=self._recovery_complete,
            wait_task_stopped_fn=self._wait_execution_stopped,
            stop_timeout_sec=self.reset_stop_timeout_sec,
        )
        self._state_machine = LlmControlTaskStateMachine(self)
        self._pregrasp_timer = self.create_timer(
            0.2, self._tick_state_machines, callback_group=self.callback_group
        )
        self.get_logger().info(
            "Robot agent boundary ready: /llm_control/agent_command, /llm_control/execute_preview"
        )

    def _declare_task_parameters(self):
        defaults = {
            "yolo_topic": "/yolo/detected_result",
            "depth_topic": "/yolo/detected_result/depth",
            "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            "preview_max_age_sec": 15.0,
            "detection_max_age_sec": 1.0,
            "rgb_depth_tolerance_sec": 0.05,
            "vision_wait_timeout_sec": 3.5,
            "tf_wait_timeout_sec": 0.6,
            "pick_classes": ["elongated_object", "cube", "stone"],
            "place_classes": ["box"],
            "workspace_min_xy": [-0.9, -0.9],
            "workspace_max_xy": [0.9, 0.9],
            "pregrasp_pose.x": 0.1,
            "pregrasp_pose.y": 0.35,
            "pregrasp_pose.z": 0.30,
            "pregrasp_pose.roll": 0.0,
            "pregrasp_pose.pitch": -180.0,
            "pregrasp_pose.yaw": 100.0,
            "grasp_above": 0.04,
            "grasp_offset": 0.010,
            "place_offset": 0.08,
            "descend_to_box": 0.04,
            "grasp.elongated_object.roll": 0.0,
            "grasp.elongated_object.pitch": -180.0,
            "grasp.elongated_object.yaw_offset": 90.0,
            "grasp.cube.roll": 0.0,
            "grasp.cube.pitch": -180.0,
            "grasp.cube.yaw_offset": 0.0,
            "grasp.stone.roll": 0.0,
            "grasp.stone.pitch": -180.0,
            "grasp.stone.yaw_offset": -45.0,
            "reset_stop_timeout_sec": 5.0,
            "batch_box_zone_radius_m": 0.08,
            "batch_rebind_max_distance_m": 0.10,
            "batch_rebind_timeout_sec": 5.0,
            "batch_rebind_stable_frames": 2,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_task_parameters(self):
        def value(name):
            return self.get_parameter(name).value

        self.yolo_topic = str(value("yolo_topic"))
        self.depth_topic = str(value("depth_topic"))
        self.camera_info_topic = str(value("camera_info_topic"))
        self.preview_max_age_sec = float(value("preview_max_age_sec"))
        self.detection_max_age_sec = float(value("detection_max_age_sec"))
        self.rgb_depth_tolerance_sec = float(value("rgb_depth_tolerance_sec"))
        self.vision_wait_timeout_sec = float(value("vision_wait_timeout_sec"))
        self.tf_wait_timeout_sec = float(value("tf_wait_timeout_sec"))
        self.pick_classes = frozenset(str(item) for item in value("pick_classes"))
        self.place_classes = frozenset(str(item) for item in value("place_classes"))
        self.workspace_min_xy = tuple(float(item) for item in value("workspace_min_xy"))
        self.workspace_max_xy = tuple(float(item) for item in value("workspace_max_xy"))
        if len(self.workspace_min_xy) != 2 or len(self.workspace_max_xy) != 2:
            raise ValueError("workspace XY bounds must each contain exactly two values")
        if any(
            lower > upper
            for lower, upper in zip(self.workspace_min_xy, self.workspace_max_xy)
        ):
            raise ValueError("workspace XY lower bounds must not exceed upper bounds")
        self.pregrasp_pose_cfg = {
            axis: float(value(f"pregrasp_pose.{axis}"))
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.pregrasp_pose = self._build_pregrasp_pose()
        for name in (
            "grasp_above", "grasp_offset", "place_offset", "descend_to_box",
            "reset_stop_timeout_sec",
        ):
            setattr(self, name, float(value(name)))
        self.grasp_profiles = {
            name: tuple(float(value(f"grasp.{name}.{axis}"))
                        for axis in ("roll", "pitch", "yaw_offset"))
            for name in ("elongated_object", "cube", "stone")
        }
        self.batch_box_zone_radius_m = float(value("batch_box_zone_radius_m"))
        self.batch_rebind_max_distance_m = float(value("batch_rebind_max_distance_m"))
        self.batch_rebind_timeout_sec = float(value("batch_rebind_timeout_sec"))
        self.batch_rebind_stable_frames = max(
            1, int(value("batch_rebind_stable_frames"))
        )

    def _tick_state_machines(self):
        self._state_machine.tick()

    def _advance_safety(self, command):
        command = str(command).strip().lower()
        with self._lock:
            if command in ("stop", "reset"):
                for context_session in tuple(self._task_contexts):
                    self._clear_task_context_locked(context_session)
            if command in ("reset", "resume"):
                self._reset_failed = False
            updated = apply_safety_command(self._safety, command)
            if updated is self._safety:
                return
            self._safety = updated
            if command == "stop":
                self._state = "RESET_FAILED" if self._reset_failed else "STOPPED"
            elif command == "reset":
                self._state = "RESETTING"
            elif command == "resume":
                if self._state in ("STOPPED", "RESETTING", "RESET_FAILED"):
                    if not self._execution_active:
                        self._state = self._resting_state_locked()

    def _refresh_yolo_health(self) -> bool:
        health = getattr(self, "llm_yolo_status_client", None)
        if health is None or not health.service_is_ready():
            self._visual_available = False
            self._visual_last_error = "YOLO status service is unavailable"
            return False
        future = health.call_async(Trigger.Request())
        deadline = time.monotonic() + 1.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        response = future.result() if future.done() else None
        self._visual_available = bool(response is not None and response.success)
        self._visual_last_error = (
            "" if self._visual_available
            else str(getattr(response, "message", "YOLO status request timed out"))
        )
        return self._visual_available

    def _resting_state_locked(self):
        return "HOLDING" if self._held_source is not None else "IDLE"

    def _clear_holding_locked(self):
        self._held_source = None

    def _motion_block_reason_locked(self):
        reasons = []
        if self._safety.blocked:
            reasons.append("safety state is blocked")
        if self.abort.is_set():
            reasons.append("abort manager is set")
        return "; ".join(reasons)

    def _prune_previews_locked(self, now=None):
        for preview_id in [
            key for key, record in self._previews.items()
            if preview_status(record.preview, now) != "ready"
        ]:
            self._previews.pop(preview_id, None)
        if self._state == "PREVIEW_READY" and not self._previews:
            self._state = self._resting_state_locked()

    def _take_preview_locked(self, preview_id):
        self._prune_previews_locked()
        return self._previews.pop(preview_id, None)

    def _execution_interrupted(self, execution_epoch, goal_handle=None):
        with self._lock:
            valid = safety_execution_valid(self._safety, execution_epoch)
            shutting_down = self._shutting_down
        cancel_requested = goal_handle is not None and goal_handle.is_cancel_requested
        return shutting_down or not valid or self.abort.is_set() or cancel_requested

    def _mark_stop_state(self):
        with self._lock:
            if self.abort.is_stop_requested() or self._safety.command == "stop":
                self._state = "STOPPED"

    def _stop_for_motion_failure(self, reason: str):
        if self.abort.request_abort(str(reason), command="stop"):
            self.abort.cancel_all_motion_now()
        with self._lock:
            self._state = "STOPPED"

    def _mark_holding_recovery(self, source, destination):
        with self._lock:
            if self._held_source is not None:
                self._held_source = source
                self._state = "HOLDING"

    def _record_holding_valid_locked(self, record):
        if record is None:
            return False
        visual = next(
            (
                action
                for action in record.enriched_actions
                if action.get("type") in ("pick", "place", "pick_place")
            ),
            None,
        )
        if visual is None:
            return True
        if visual["type"] == "place":
            return self._held_source == visual["source"]
        return self._held_source is None

    def _clear_task_context_locked(self, session_id):
        self._task_contexts.pop(str(session_id), None)

    def _clear_session(self, msg):
        session_id = str(msg.data).strip()
        if not session_id:
            return
        with self._lock:
            for preview_id in [
                key for key, record in self._previews.items()
                if record.session_id == session_id
            ]:
                self._previews.pop(preview_id, None)
            self._clear_task_context_locked(session_id)
            for key in [key for key in self._tool_results if key[0] == session_id]:
                self._tool_results.pop(key, None)
            if self._state == "PREVIEW_READY" and not self._previews:
                self._state = self._resting_state_locked()

    @staticmethod
    def _fill_agent_response(response, value):
        response.accepted = bool(value.get("accepted", False))
        response.status = str(value.get("status", "rejected"))
        response.preview_id = str(value.get("preview_id", ""))
        response.error_code = str(value.get("error_code", ""))
        response.result_json = json.dumps(value.get("result", {}), ensure_ascii=False)
        return response

    def _agent_reject(self, response, detail, error_code=None):
        code = str(error_code or "PLAN_INVALID")
        self.get_logger().error(f"Agent command rejected [{code}]: {detail}")
        return self._fill_agent_response(response, {
            "accepted": False, "status": "rejected", "error_code": code,
            "result": {
                "spoken_text": spoken_error_text(code), "detail": str(detail)
            },
        })

    def _planning_metadata(self, required):
        if not required:
            return []
        if not self._refresh_yolo_health():
            raise PreviewFailure("VISION_UNAVAILABLE", self._visual_last_error)
        try:
            metadata = self._wait_for_planning_metadata()
            self._visual_available = True
            self._visual_last_error = ""
            return self._exclude_box_contents(metadata)
        except PerceptionUnavailable as exc:
            self._visual_available = False
            self._visual_last_error = exc.error_code
            raise PreviewFailure(exc.error_code, str(exc)) from exc

    def _store_tool_result(self, key, response):
        value = {
            "accepted": bool(response.accepted), "status": response.status,
            "preview_id": response.preview_id, "error_code": response.error_code,
            "result": json.loads(response.result_json or "{}"),
        }
        with self._lock:
            self._tool_results[key] = value
        return response

    def _agent_command(self, request, response):
        # Serializing this narrow boundary makes call_id idempotency atomic.
        with self._command_lock:
            return self._agent_command_locked(request, response)

    def _handle_cancel_tool(self, key, response, session_id, spoken_text):
        with self._lock:
            self._clear_task_context_locked(session_id)
        self._fill_agent_response(response, {
            "accepted": True,
            "status": "cancelled",
            "result": {"spoken_text": spoken_text},
        })
        return self._store_tool_result(key, response)

    def _handle_inspect_scene_tool(
        self, key, response, session_id, *, include_rgb
    ):
        frame = self.perception.current_frame()
        metadata = (
            [] if frame is None
            else self.perception.planning_metadata(frame, include_unavailable=True)
        )
        scene_id = "scene-" + uuid.uuid4().hex[:12]
        scene_entities = self._scene_entities(metadata)
        diagnostics = self.perception.diagnostics(frame)
        with self._lock:
            self._task_contexts[session_id] = TaskContext(
                scene_id=scene_id,
                scene_created_at=time.monotonic(),
                entities=scene_entities,
            )
        self.get_logger().debug("SCENE_TRACE " + json.dumps({
            "session_id": session_id,
            "scene_id": scene_id,
            "result_seq": max(
                (item.result_seq for item in scene_entities), default=0
            ),
            "entities": [
                {
                    "entity_id": item.entity_id,
                    "class_name": item.class_name,
                    "role": item.role,
                    "selectable": item.selectable,
                    "unavailable_reason": item.unavailable_reason,
                }
                for item in scene_entities
            ],
        }, ensure_ascii=False, separators=(",", ":")))
        self._fill_agent_response(response, {
            "accepted": True,
            "status": "scene",
            "result": {
                "scene_id": scene_id,
                "entities": [item.public() for item in scene_entities],
                "diagnostics": diagnostics,
                "scene_error": "" if frame is not None else "VISION_SYNC_STALE",
                "include_rgb": include_rgb,
            },
        })
        return self._store_tool_result(key, response)

    def _handle_task_tool(
        self, key, response, session_id, name, value, preview_epoch
    ):
        with self._lock:
            context = self._task_contexts.get(session_id)
        if name == "ask_user":
            self._fill_agent_response(response, {
                "accepted": True,
                "status": "clarification_required",
                "result": {"spoken_text": value["question"]},
            })
            return self._store_tool_result(key, response)

        invocation = value
        if invocation.skill.startswith("yolo."):
            if context is None or invocation.scene_id != context.scene_id:
                raise PreviewFailure(
                    "SCENE_EXPIRED",
                    "scene_id does not match the current snapshot",
                )
            if time.monotonic() - context.scene_created_at > self.preview_max_age_sec:
                raise PreviewFailure(
                    "SCENE_EXPIRED", "scene snapshot expired before submission"
                )
        metadata = self._planning_metadata(invocation.skill.startswith("yolo."))
        plan = self._compile_invocation(
            invocation, context.entities if context else (), metadata
        )
        preview_id = self._build_preview(plan, session_id, preview_epoch)
        self._fill_agent_response(response, {
            "accepted": True,
            "status": "ready",
            "preview_id": preview_id,
            "result": {
                "spoken_text": invocation_spoken_text(invocation, len(plan.actions)),
            },
        })
        return self._store_tool_result(key, response)

    def _agent_command_locked(self, request, response):
        session_id = str(request.session_id).strip()
        call_id = str(request.call_id).strip()
        tool_name = str(request.tool_name).strip()
        if not session_id or not call_id:
            return self._agent_reject(
                response,
                "session_id and call_id are required",
                "MODEL_RESPONSE_INVALID",
            )
        key = (session_id, call_id)
        with self._lock:
            cached = self._tool_results.get(key)
            state = self._state
            blocked = self._motion_block_reason_locked()
            preview_epoch = self._safety.epoch
            shutting_down = self._shutting_down
        if cached is not None:
            return self._fill_agent_response(response, cached)
        if shutting_down:
            return self._store_tool_result(
                key, self._agent_reject(response, "server is shutting down", "ROBOT_BUSY")
            )
        if tool_name not in ("inspect_scene", "cancel_task") and (
            blocked
            or state
            in ("PREGRASP_POSE", "STOPPED", "RESETTING", "RESET_FAILED", "EXECUTING")
        ):
            code = (
                "ROBOT_BUSY" if state == "EXECUTING" else
                "ROBOT_INITIALIZING" if state == "PREGRASP_POSE" else
                "SAFETY_BLOCKED"
            )
            return self._store_tool_result(
                key, self._agent_reject(response, blocked or f"state={state}", code)
            )
        try:
            arguments = json.loads(str(request.arguments_json) or "{}")
            try:
                name, value = parse_agent_tool_call(
                    tool_name, arguments, allowed_tools=AGENT_TOOLS
                )
            except PreviewFailure:
                raise
            except ValueError as exc:
                return self._store_tool_result(
                    key, self._agent_reject(response, exc, "MODEL_RESPONSE_INVALID")
                )
            if name == "cancel_task":
                return self._handle_cancel_tool(key, response, session_id, value)
            if name == "inspect_scene":
                return self._handle_inspect_scene_tool(
                    key,
                    response,
                    session_id,
                    include_rgb=value["include_rgb"],
                )
            return self._handle_task_tool(
                key, response, session_id, name, value, preview_epoch
            )
        except PreviewFailure as exc:
            return self._store_tool_result(key, self._agent_reject(
                response, exc, exc.error_code
            ))
        except ValueError as exc:
            return self._store_tool_result(
                key, self._agent_reject(response, exc, "PLAN_INVALID")
            )
        except Exception as exc:
            return self._store_tool_result(
                key, self._agent_reject(response, exc, "PLAN_INVALID")
            )

    def _goal_callback(self, goal_request):
        rejection_reason = ""
        with self._lock:
            self._prune_previews_locked()
            record = self._previews.get(goal_request.preview_id)
            motion_block_reason = self._motion_block_reason_locked()
            if self._shutting_down:
                rejection_reason = "server is shutting down"
            elif motion_block_reason:
                rejection_reason = motion_block_reason
            elif self._state in (
                LlmControlTaskState.PREGRASP_POSE.value,
                "STOPPED",
                "RESETTING",
                "RESET_FAILED",
                "EXECUTING",
            ):
                rejection_reason = f"state={self._state}"
            elif record is None:
                rejection_reason = "preview id is unknown"
            elif record.session_id != goal_request.session_id:
                rejection_reason = "preview belongs to a different CLI session"
            elif record.safety_epoch != self._safety.epoch:
                rejection_reason = "safety epoch changed"
            elif not self._record_holding_valid_locked(record):
                rejection_reason = "holding state changed"
            else:
                status = preview_status(record.preview)
                if status != "ready":
                    rejection_reason = f"preview status is {status}"
        if rejection_reason:
            self.get_logger().warning(
                f"Execute preview rejected ({goal_request.preview_id}): {rejection_reason}."
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle):
        if self.abort.request_abort("execute action cancelled", command="stop"):
            self.abort.cancel_all_motion_now()
        return CancelResponse.ACCEPT

    def _revalidate(self, record):
        with self._lock:
            if not self._record_holding_valid_locked(record):
                raise ValueError("held-object state changed; regenerate preview")
        pick_actions = [
            action for action in record.enriched_actions
            if action["type"] in ("pick", "pick_place")
        ]
        if not pick_actions:
            return
        if not self._refresh_yolo_health():
            raise ValueError("YOLO inference is unavailable for pick revalidation")
        planned_result_seq = max(
            int(getattr(action.get("source"), "result_seq", 0))
            for action in pick_actions
        )
        self._wait_for_planning_metadata(after_result_seq=planned_result_seq)
        for action in pick_actions:
            source = self.perception.fresh_match(action["source"])
            if source is None:
                raise ValueError("pick target is no longer detectable")
            action["source"] = source

    def _feedback(self, goal_handle, index, count, phase, message, pose=None):
        feedback = ExecutePreview.Feedback()
        feedback.step_index = int(index)
        feedback.step_count = int(count)
        feedback.phase = str(phase)
        feedback.message = str(message)
        if pose is not None:
            feedback.active_target = pose
        goal_handle.publish_feedback(feedback)

    def _move_pose(self, pose, name, cartesian=False, velocity=None):
        return self.motion.move_to_pose(
            pose,
            planning_client="fairino",
            cartesian=cartesian,
            action_name=name,
            max_velocity=self.arm_max_velocity if velocity is None else velocity,
            max_acceleration=self.arm_max_acceleration if velocity is None else velocity,
            max_step_size=self.max_step_size,
            allowed_planning_time=self.allowed_planning_time,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            allowed_start_tolerance=self.allowed_start_tolerance,
            timeout_sec=self.execute_timeout_sec,
        )

    def _move_to_pregrasp_pose(self):
        return self._move_pose(self.pregrasp_pose, "Move to pregrasp pose")

    def _wait_execution_stopped(self, timeout_sec):
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while time.monotonic() <= deadline:
            with self._lock:
                if not self._execution_active:
                    return True
            if self.abort.is_stop_requested():
                return False
            time.sleep(0.02)
        return False

    def _recovery_complete(self, ok_home):
        released = self.abort.recovery_released()
        stopped = self.abort.is_stop_requested()
        with self._lock:
            self._previews.clear()
            if released:
                self._clear_holding_locked()
            if ok_home:
                self._reset_failed = False
                self._safety = complete_safety_reset(self._safety)
                self._state = "STOPPED" if self._execution_active else "IDLE"
            elif stopped:
                self._reset_failed = False
                self._state = "STOPPED"
            else:
                self._reset_failed = True
                self._state = "RESET_FAILED"

    @staticmethod
    def _finish_execution(goal_handle, result, terminal_state, message, *, cancelled=False):
        if cancelled:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.terminal_state = terminal_state
        result.message = message
        return result

    def _execute_preview(self, goal_handle):
        request = goal_handle.request
        result = ExecutePreview.Result()
        with self._lock:
            self._prune_previews_locked()
            record = self._previews.get(request.preview_id)
            execution_epoch = self._safety.epoch
            if self._execution_active:
                return self._finish_execution(
                    goal_handle,
                    result,
                    "REJECTED",
                    "another task began execution before this goal",
                )
            if (
                record is None
                or record.safety_epoch != execution_epoch
                or not safety_execution_valid(self._safety, execution_epoch)
                or self.abort.is_set()
                or not self._record_holding_valid_locked(record)
            ):
                return self._finish_execution(
                    goal_handle,
                    result,
                    "STOPPED",
                    "motion is stopped or preview is unavailable",
                )
            if self._take_preview_locked(request.preview_id) is not record:
                return self._finish_execution(
                    goal_handle,
                    result,
                    "REJECTED",
                    "preview expired before execution began",
                )
            self._execution_active = True
            self._state = "EXECUTING"
        try:
            visual_actions = [
                action for action in record.enriched_actions
                if action["type"] in ("pick", "place", "pick_place")
            ]
            is_batch = len(visual_actions) > 1
            if not is_batch:
                self._revalidate(record)
            if self._execution_interrupted(execution_epoch, goal_handle):
                self._mark_stop_state()
                return self._finish_execution(
                    goal_handle, result, "STOPPED", "task invalidated before execution"
                )
            step_count = execution_step_count(record.enriched_actions)
            step_index = 0
            batch_ordinal = 0
            for action in record.enriched_actions:
                if self._execution_interrupted(execution_epoch, goal_handle):
                    self._mark_stop_state()
                    return self._finish_execution(
                        goal_handle,
                        result,
                        "STOPPED",
                        "task cancelled",
                        cancelled=goal_handle.is_cancel_requested,
                    )
                if is_batch and action["type"] in ("pick", "place", "pick_place"):
                    batch_ordinal += 1
                    action = self._rebind_batch_action(
                        record,
                        action,
                        visual_actions[batch_ordinal - 1:],
                        batch_ordinal,
                        len(visual_actions),
                    )
                    if self._execution_interrupted(execution_epoch, goal_handle):
                        self._mark_stop_state()
                        return self._finish_execution(
                            goal_handle,
                            result,
                            "STOPPED",
                            "task stopped during batch rebind",
                        )
                ok, message, action_steps = self._state_machine.execute_action(
                    action, goal_handle, step_index, step_count, execution_epoch
                )
                step_index += action_steps
                if self._execution_interrupted(execution_epoch, goal_handle):
                    self._mark_stop_state()
                    return self._finish_execution(
                        goal_handle,
                        result,
                        "STOPPED",
                        "task invalidated by stop/reset",
                        cancelled=goal_handle.is_cancel_requested,
                    )
                if not ok:
                    self._stop_for_motion_failure(message)
                    return self._finish_execution(
                        goal_handle, result, "STOPPED", message
                    )
                if is_batch and action["type"] == "pick_place":
                    self._batch_trace(
                        session_id=record.session_id,
                        preview_id=record.preview.preview_id,
                        ordinal=int(batch_ordinal),
                        total=len(visual_actions),
                        action_result="completed",
                    )
            goal_handle.succeed()
            result.success = True
            with self._lock:
                terminal_state = self._resting_state_locked()
                self._state = terminal_state
            result.terminal_state = terminal_state if terminal_state == "HOLDING" else "COMPLETED"
            picked = any(action["type"] == "pick" for action in record.enriched_actions)
            if terminal_state == "HOLDING":
                result.message = (
                    "pick complete; holding object"
                    if picked
                    else "complete plan executed; still holding object"
                )
            else:
                if is_batch:
                    count = len(visual_actions)
                    result.message = f"已执行 {count} 次抓放动作"
                else:
                    result.message = "complete plan executed"
            return result
        except Exception as exc:
            goal_handle.abort()
            if self._execution_interrupted(execution_epoch, goal_handle):
                self._mark_stop_state()
                result.terminal_state, result.message = "STOPPED", str(exc)
            else:
                result.terminal_state, result.message = "FAILED", str(exc)
            return result
        finally:
            self._mark_stop_state()
            with self._lock:
                if not result.success:
                    self._clear_task_context_locked(record.session_id)
                self._execution_active = False
                if self._state == "EXECUTING":
                    self._state = self._resting_state_locked()
                elif (
                    self._state == "STOPPED"
                    and not self._safety.blocked
                    and not self.abort.is_set()
                ):
                    self._state = self._resting_state_locked()

    def _status(self, _request, response):
        diagnostics = self.perception.diagnostics()
        with self._lock:
            state = self._state
            holding = self._held_source is not None
        response.success = True
        response.message = json.dumps({
            "state": state,
            "visual_available": self._visual_available,
            "visual_last_error": self._visual_last_error,
            **diagnostics,
            "holding": holding,
            "recovery_active": self.abort.recovery_active(),
            "reset_message": self.abort.recovery_message(),
            "agent_protocol": "qwen_realtime_tools",
        }, ensure_ascii=False)
        return response


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = LlmControlTaskServer()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with node._lock:
                node._shutting_down = True
            if rclpy.ok():
                node.abort.request_abort("shutdown", command="stop")
                node.abort.cancel_all_motion_now()
            executor.shutdown(timeout_sec=2.0)
            node.execute_action.destroy()
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
