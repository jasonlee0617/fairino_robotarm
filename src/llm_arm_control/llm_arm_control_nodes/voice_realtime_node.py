#!/usr/bin/env python3
"""Wake-gated realtime voice coordinator and robot tool bridge."""

from __future__ import annotations

import base64
import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid

from audio_common_msgs.msg import AudioDataStamped, AudioInfo
from ament_index_python.packages import get_package_share_directory
from llm_arm_control.action import ExecutePreview
from llm_arm_control.srv import AgentCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from llm_arm_control_nodes.agent_protocol import (
    AGENT_TOOLS,
    SYSTEM_PROMPT,
    compact_tool_result,
    normalize_tool_arguments,
    protocol_failure_response,
    scene_response_instructions,
    spoken_only_instructions,
    tool_argument_shape,
)
from llm_arm_control_nodes.realtime_provider import (
    ProviderConfigurationError,
    RealtimeEvent,
    RealtimeProviderConfig,
    create_realtime_provider,
    validate_realtime_provider,
)
from llm_arm_control_nodes.realtime_session import RealtimeSessionState
from llm_arm_control_nodes.voice_logic import classify_wake


class PcmPlayer:
    """One raw-PCM aplay stream per uninterrupted response."""

    def __init__(self, sink: str, sample_rate: int = 24000):
        self.sink = sink
        self.sample_rate = int(sample_rate)
        self._lock = threading.Lock()
        self._process = None

    def _start_locked(self):
        if self._process is not None and self._process.poll() is None:
            return self._process
        environment = dict(os.environ)
        environment["PULSE_SINK"] = self.sink
        self._process = subprocess.Popen(
            ["aplay", "-q", "-D", "pulse", "-t", "raw", "-f", "S16_LE",
             "-r", str(self.sample_rate), "-c", "1", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=environment,
        )
        return self._process

    def write_b64(self, value: str):
        data = base64.b64decode(value, validate=True)
        with self._lock:
            process = self._start_locked()
            try:
                process.stdin.write(data)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                self._terminate_locked()

    def play_file(self, path: str, timeout_sec: float = 3.0) -> bool:
        environment = dict(os.environ)
        environment["PULSE_SINK"] = self.sink
        with self._lock:
            self._terminate_locked()
            process = subprocess.Popen(
                ["aplay", "-q", "-D", "pulse", "-t", "raw", "-f", "S16_LE",
                 "-r", str(self.sample_rate), "-c", "1", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=environment,
            )
            self._process = process
        try:
            return process.wait(timeout=timeout_sec) == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            with self._lock:
                if self._process is process:
                    self._terminate_locked()

    def _terminate_locked(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=0.3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def interrupt(self):
        with self._lock:
            self._terminate_locked()


class VoiceRealtimeNode(Node):
    def __init__(self):
        super().__init__("voice_realtime_node")
        for name, value in {
            "realtime_provider": "qwen",
            "model": "qwen3.8-omni-flash-realtime",
            "voice": "Tina",
            "idle_timeout_sec": 30.0,
            "vad_silence_ms": 700,
            "response_transition_timeout_sec": 5.0,
            "audio_chunk_ms": 100,
            "temperature": 0.1,
            "aec_sink": "llm_aec_sink",
            "wake_ack_tail_ms": 150,
            "color_topic": "/camera/camera/color/image_raw",
        }.items():
            self.declare_parameter(name, value)
        self._lock = threading.RLock()
        self._events = queue.SimpleQueue()
        self._state = RealtimeSessionState()
        self._provider = None
        self._executing = False
        self._shutting_down = False
        self._execution_turn = -1
        self._pending_status_text = ""
        self._audio_buffer = bytearray()
        self._latest_image = None
        self._latest_image_stamp = 0.0
        self._player = PcmPlayer(str(self.get_parameter("aec_sink").value))
        self._wake_ack_pcm = os.path.join(
            get_package_share_directory("llm_arm_control"),
            "audio", "wake_ack_zh.pcm",
        )

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.listen_pub = self.create_publisher(String, "/voice_control/listen_mode", latched)
        self.clear_pub = self.create_publisher(String, "/llm_control/clear_session", 10)
        self.create_subscription(String, "/voice_control/wake_event", self._on_wake, 10)
        self.create_subscription(AudioDataStamped, "/voice_control/audio", self._on_audio, 30)
        self.create_subscription(
            AudioInfo, "/voice_control/audio_info", self._on_audio_info, latched
        )
        self.create_subscription(
            Image, str(self.get_parameter("color_topic").value), self._on_image,
            qos_profile_sensor_data,
        )
        self.agent_client = self.create_client(AgentCommand, "/llm_control/agent_command")
        self.execute_client = ActionClient(self, ExecutePreview, "/llm_control/execute_preview")
        self._publish_mode("MUTED")
        self.create_timer(0.02, self._drain_events)
        self.create_timer(0.5, self._tick)
        validate_realtime_provider(self.get_parameter("realtime_provider").value)
        self._start_connect()

    def _publish_mode(self, mode):
        self.listen_pub.publish(String(data=str(mode)))

    def _start_connect(self, delay=0.0):
        with self._lock:
            if self._shutting_down:
                return
            generation = self._state.start_connect()
            if generation is None:
                return
        threading.Thread(target=self._connect, args=(generation, delay), daemon=True).start()

    def _connect(self, generation, delay=0.0):
        if delay:
            time.sleep(delay)
        try:
            config = RealtimeProviderConfig(
                model=str(self.get_parameter("model").value),
                voice=str(self.get_parameter("voice").value),
                vad_silence_ms=int(self.get_parameter("vad_silence_ms").value),
                temperature=float(self.get_parameter("temperature").value),
                instructions=SYSTEM_PROMPT,
                tool_names=AGENT_TOOLS,
            )
            provider = create_realtime_provider(
                str(self.get_parameter("realtime_provider").value),
                config,
                generation,
                self._events.put,
            )
            with self._lock:
                if not self._state.accepts(generation) or self._shutting_down:
                    return
                self._provider = provider
            provider.connect()
        except ProviderConfigurationError as exc:
            self._events.put(RealtimeEvent(
                "provider_error", generation,
                {"message": str(exc), "retryable": False},
            ))
        except Exception as exc:
            self._events.put(RealtimeEvent(
                "provider_error", generation,
                {"message": str(exc), "retryable": True},
            ))

    def _on_audio_info(self, msg):
        if (
            int(msg.channels) != 1
            or int(msg.sample_rate) != 16000
            or str(msg.sample_format).upper() != "S16LE"
        ):
            self.get_logger().error("Qwen input requires 16 kHz mono S16LE audio.")

    def _on_audio(self, msg):
        with self._lock:
            if (
                not self._state.active or not self._state.audio_upload_enabled
                or not self._state.connected or self._provider is None
            ):
                return
            self._audio_buffer.extend(bytes(msg.audio.data))
            chunk_size = int(16000 * 2 * int(self.get_parameter("audio_chunk_ms").value) / 1000)
            chunks = []
            while len(self._audio_buffer) >= chunk_size:
                chunks.append(bytes(self._audio_buffer[:chunk_size]))
                del self._audio_buffer[:chunk_size]
            provider = self._provider
        for chunk in chunks:
            try:
                provider.send_audio(chunk)
            except Exception as exc:
                self._events.put(RealtimeEvent(
                    "provider_error", self._state.generation,
                    {"message": str(exc), "retryable": True},
                ))
                break

    def _on_image(self, msg):
        with self._lock:
            self._latest_image = msg
            self._latest_image_stamp = time.monotonic()

    def _on_wake(self, _msg):
        with self._lock:
            session_id = uuid.uuid4().hex
            if not self._state.begin_session(session_id, time.monotonic()):
                return
            self._audio_buffer.clear()
        self._publish_mode("MUTED")
        threading.Thread(
            target=self._play_wake_ack,
            args=(session_id,),
            daemon=True,
        ).start()
        self.get_logger().info("Qwen Realtime voice session activated.")

    def _play_wake_ack(self, session_id):
        played = False
        try:
            played = self._player.play_file(self._wake_ack_pcm)
        except Exception as exc:
            self.get_logger().error(f"Wake acknowledgement playback failed: {exc}")
        if not played:
            self.get_logger().error(
                "Wake acknowledgement was not played; opening input after bounded fallback."
            )
        time.sleep(max(0.0, float(self.get_parameter("wake_ack_tail_ms").value)) / 1000.0)
        with self._lock:
            if self._state.active and self._state.session_id == session_id:
                self._audio_buffer.clear()
                self._state.audio_upload_enabled = True
                self._state.last_activity = time.monotonic()

    def _audio_trace(self, **values):
        self.get_logger().debug(
            "QWEN_AUDIO_TRACE " + json.dumps(values, ensure_ascii=False, sort_keys=True)
        )

    def _handle_event(self, event: RealtimeEvent):
        if not self._state.accepts(event.generation):
            return
        kind, data = event.kind, event.data
        if kind in ("connection_opened", "session_ready", "connection_closed", "provider_error"):
            VoiceRealtimeNode._handle_connection_event(self, event)
        elif kind in (
            "speech_started", "speech_stopped", "transcription_completed",
            "transcription_failed",
        ):
            VoiceRealtimeNode._handle_input_event(self, kind, data)
        else:
            VoiceRealtimeNode._handle_response_event(self, kind, data)

    def _handle_connection_event(self, event: RealtimeEvent):
        kind, data = event.kind, event.data
        if kind == "connection_opened":
            with self._lock:
                self._state.connected = True
            self.get_logger().debug("Qwen Realtime connected; configuring session.")
        elif kind == "session_ready":
            with self._lock:
                recovering = self._state.connection_ready(time.monotonic())
                recovery_turn = self._state.turn_serial if recovering else -1
                pending_status = self._pending_status_text if recovering else ""
                if recovering:
                    self._pending_status_text = ""
            self._publish_mode("MUTED" if recovering else "WAKE_ONLY")
            self.get_logger().info("Qwen Realtime ready; voice input is wake-gated.")
            if recovering:
                text = pending_status or "连接已恢复，请重新说刚才的指令。"
                self._submit_response(recovery_turn, spoken_only_instructions(text))
        elif kind in ("connection_closed", "provider_error"):
            details = self._realtime_error_details(event)
            self.get_logger().error(
                "QWEN_CONNECTION_ERROR "
                + json.dumps(details, ensure_ascii=False, separators=(",", ":"))
            )
            self._handle_connection_loss(reconnect=bool(data.get("retryable", True)))

    def _handle_input_event(self, kind, data):
        if kind == "speech_started":
            self._player.interrupt()
            self._cancel_active_response("new_speech")
            item_id = str(data.get("item_id", "")).strip()
            with self._lock:
                turn_serial = self._state.record_speech(item_id, time.monotonic())
            self._audio_trace(
                event="speech_started", event_id=str(data.get("event_id", "")),
                item_id=item_id, turn_serial=turn_serial,
                audio_start_ms=data.get("audio_start_ms"),
            )
        elif kind == "speech_stopped":
            item_id = str(data.get("item_id", "")).strip()
            with self._lock:
                turn_serial = self._state.item_turns.get(
                    item_id, self._state.turn_serial
                )
            self._audio_trace(
                event="speech_stopped", event_id=str(data.get("event_id", "")),
                item_id=item_id, turn_serial=turn_serial,
                audio_end_ms=data.get("audio_end_ms"),
            )
        elif kind == "transcription_completed":
            item_id = str(data.get("item_id", "")).strip()
            transcript = str(data.get("transcript", ""))
            with self._lock:
                turn_serial, duplicate = self._state.record_transcript(
                    item_id, time.monotonic()
                )
            if duplicate:
                self._audio_trace(
                    event="transcription_ignored",
                    event_id=str(data.get("event_id", "")),
                    item_id=item_id, turn_serial=turn_serial,
                    reason="duplicate_item",
                )
                return
            wake = bool(classify_wake(transcript)[0])
            self.get_logger().info("QWEN_INPUT " + json.dumps({
                "text": transcript, "item_id": item_id, "turn_serial": turn_serial,
            }, ensure_ascii=False))
            self._audio_trace(
                event="transcription_completed",
                event_id=str(data.get("event_id", "")),
                item_id=item_id, turn_serial=turn_serial,
                classification="wake" if wake else "command",
            )
            if wake:
                with self._lock:
                    self._state.ignored_turns.add(turn_serial)
                self._player.interrupt()
                return
            self._request_turn_response(turn_serial)
        elif kind == "transcription_failed":
            item_id = str(data.get("item_id", "")).strip()
            with self._lock:
                turn_serial = self._state.item_turns.get(
                    item_id, self._state.turn_serial
                )
            self._audio_trace(
                event="transcription_failed",
                event_id=str(data.get("event_id", "")),
                item_id=item_id, turn_serial=turn_serial,
            )
            self.get_logger().warning(
                "Qwen input transcription failed: "
                + str(data.get("message", "unknown error"))
            )

    def _handle_response_event(self, kind, data):
        if kind == "audio_delta":
            with self._lock:
                turn_serial = self._state.response_event_turn(
                    str(data.get("response_id", ""))
                )
                ignored = (
                    turn_serial is None
                    or turn_serial in self._state.ignored_turns
                    or self._state.response_cancelled
                )
            if ignored:
                return
            try:
                self._player.write_b64(str(data.get("audio_b64", "")))
            except Exception as exc:
                self.get_logger().error(f"Qwen audio playback failed: {exc}")
        elif kind == "response_transcript_done":
            self.get_logger().debug("QWEN_OUTPUT " + json.dumps(
                {"text": str(data.get("transcript", ""))}, ensure_ascii=False
            ))
        elif kind == "tool_call":
            with self._lock:
                turn_serial = self._state.response_event_turn(
                    str(data.get("response_id", ""))
                )
                valid = (
                    turn_serial is not None
                    and turn_serial == self._state.response_turn
                    and not self._state.response_cancelled
                )
            if valid:
                self._call_tool(data, turn_serial=turn_serial)
            else:
                self._audio_trace(
                    event="tool_call_ignored", turn_serial=turn_serial,
                    reason="stale_or_cancelled_response",
                )
        elif kind == "response_started":
            response_id = str(data.get("response_id", "")).strip()
            with self._lock:
                turn_serial, cancel_after_create, pending_turn = (
                    self._state.response_started(response_id)
                )
            self._audio_trace(
                event="response_created", turn_serial=turn_serial,
                response_id=response_id, cancelled=cancel_after_create,
                pending_turn=pending_turn,
            )
            if cancel_after_create:
                self._send_response_cancel(
                    turn_serial, "cancelled_before_response_created"
                )
        elif kind == "response_done":
            with self._lock:
                turn_serial = self._state.response_turn
                cancelled = self._state.response_cancelled
                pending_turn = (
                    self._state.pending_response.turn_serial
                    if self._state.pending_response is not None else -1
                )
            self.get_logger().debug("QWEN_RESPONSE_TRACE " + json.dumps({
                "response_id": str(data.get("response_id", "")),
                "turn_serial": turn_serial,
                "status": str(data.get("status", "")),
                "detail_type": str(data.get("detail_type", "")),
                "reason": str(data.get("reason", "")),
                "cancelled": cancelled,
                "pending_turn": pending_turn,
            }, ensure_ascii=False, separators=(",", ":")))
            self._finish_response(str(data.get("response_id", "")).strip())

    @staticmethod
    def _realtime_error_details(event: RealtimeEvent):
        data = event.data
        return {
            "event_type": event.kind,
            "code": str(data.get("code", "")),
            "error_type": str(data.get("error_type", "")),
            "message": str(data.get("message", "unknown error")),
            "close_code": str(data.get("code", ""))
            if event.kind == "connection_closed" else "",
            "generation": event.generation,
        }

    def _cancel_active_response(self, reason):
        with self._lock:
            if self._provider is None:
                return False
            request = self._state.request_cancel(time.monotonic())
            provider = self._provider
        if request is None:
            return False
        turn_serial, response_id = request
        if not response_id:
            self._audio_trace(
                event="response_cancel_deferred", turn_serial=turn_serial, reason=reason,
            )
            return True
        return VoiceRealtimeNode._cancel_response_now(
            self, provider, turn_serial, reason
        )

    def _send_response_cancel(self, turn_serial, reason):
        with self._lock:
            if self._provider is None or self._state.response_turn != int(turn_serial):
                return False
            request = self._state.request_cancel(time.monotonic())
            if request is None or not request[1]:
                return False
            provider = self._provider
        return VoiceRealtimeNode._cancel_response_now(
            self, provider, turn_serial, reason
        )

    def _cancel_response_now(self, provider, turn_serial, reason):
        try:
            provider.cancel_response()
            self._audio_trace(
                event="response_cancelled", turn_serial=turn_serial, reason=reason,
            )
            return True
        except Exception as exc:
            self.get_logger().warning(f"Qwen response cancellation failed: {exc}")
            return False

    def _submit_response(self, turn_serial, instructions=None):
        with self._lock:
            if (
                not self._state.active or not self._state.connected
                or self._provider is None
            ):
                return False
            start, request = self._state.queue_response(
                int(turn_serial), instructions, time.monotonic()
            )
            if not start:
                self._audio_trace(
                    event="response_queued", turn_serial=int(turn_serial),
                    active_turn=self._state.response_turn,
                )
                return True
            provider = self._provider
        try:
            provider.create_response(instructions=request.instructions)
            return True
        except Exception as exc:
            with self._lock:
                if self._state.response_turn == int(turn_serial):
                    self._state.reset_response()
            self.get_logger().error(f"Qwen response start failed: {exc}")
            return False

    def _finish_response(self, response_id=""):
        with self._lock:
            pending = self._state.finish_response(response_id)
        if pending is not None:
            self._submit_response(pending.turn_serial, pending.instructions)

    def _request_turn_response(self, turn_serial=None):
        with self._lock:
            session_id = self._state.session_id
            turn_serial = (
                self._state.turn_serial if turn_serial is None else int(turn_serial)
            )
            generation = self._state.generation
        if not self.agent_client.wait_for_service(timeout_sec=0.0):
            self._create_turn_response(session_id, turn_serial, {
                "entities": [], "diagnostics": {},
                "scene_error": "TASK_SERVICE_UNAVAILABLE",
            })
            return
        request = AgentCommand.Request(
            session_id=session_id,
            call_id=f"scene-{turn_serial}-{uuid.uuid4().hex}",
            tool_name="inspect_scene",
            arguments_json='{"include_rgb":false}',
        )
        future = self.agent_client.call_async(request)
        future.add_done_callback(
            lambda done: self._on_scene_context(
                session_id, turn_serial, generation, done
            )
        )

    def _on_scene_context(self, session_id, turn_serial, generation, future):
        with self._lock:
            if not self._state.accepts(generation):
                return
        try:
            response = future.result()
            scene = json.loads(response.result_json or "{}")
            if not response.accepted:
                scene = {
                    "entities": [], "diagnostics": {},
                    "scene_error": response.error_code or "VISION_UNAVAILABLE",
                }
        except Exception as exc:
            scene = {
                "entities": [], "diagnostics": {},
                "scene_error": "TASK_SERVICE_ERROR",
            }
            self.get_logger().warning(f"Scene context unavailable: {exc}")
        self._create_turn_response(session_id, turn_serial, scene)

    def _create_turn_response(self, session_id, turn_serial, scene):
        with self._lock:
            if (
                not self._state.active or turn_serial in self._state.ignored_turns
                or session_id != self._state.session_id
                or turn_serial != self._state.turn_serial
                or self._provider is None
            ):
                return
            self._state.last_activity = time.monotonic()
            item_id = next(
                (
                    item for item, serial in self._state.item_turns.items()
                    if serial == turn_serial
                ),
                "",
            )
        self._audio_trace(
            event="response_requested", item_id=item_id, turn_serial=turn_serial,
            scene_error=str(scene.get("scene_error", "")),
        )
        self._submit_response(turn_serial, scene_response_instructions(scene))

    def _call_tool(self, event, turn_serial=None):
        call_id = str(event.get("call_id", "")).strip()
        name = str(event.get("name", "")).strip()
        raw_arguments = event.get("arguments", "{}")
        with self._lock:
            self._state.tool_count += 1
            busy = self._executing and name not in ("inspect_scene", "cancel_task")
            over_budget = self._state.tool_count > 2
            session_id = self._state.session_id
            turn_serial = (
                self._state.turn_serial if turn_serial is None else int(turn_serial)
            )
            generation = self._state.generation
        if busy or over_budget:
            code = "ROBOT_BUSY" if busy else "MODEL_TOOL_BUDGET_EXCEEDED"
            self._return_tool_and_respond(call_id, {
                "accepted": False,
                "status": "rejected",
                "error_code": code,
            })
            return
        try:
            arguments = normalize_tool_arguments(raw_arguments)
            parsed_arguments = json.loads(arguments)
            top_fields = sorted(parsed_arguments)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._protocol_failure(
                call_id, name, "MODEL_TOOL_ARGUMENTS_INVALID",
                f"JSON parse error at position {getattr(exc, 'pos', -1)}",
                len(str(raw_arguments)), (), turn_serial,
            )
            return
        self._tool_trace(
            turn_serial=turn_serial, tool=name, call_id=call_id,
            argument_length=len(arguments), parse_status="valid_json",
            top_fields=top_fields, repair_count=self._state.tool_argument_errors,
            **tool_argument_shape(name, parsed_arguments),
        )
        if not self.agent_client.wait_for_service(timeout_sec=0.0):
            self._return_tool_and_respond(call_id, {
                "accepted": False, "status": "rejected",
                "error_code": "TASK_SERVICE_UNAVAILABLE",
            })
            return
        request = AgentCommand.Request(
            session_id=session_id, call_id=call_id, tool_name=name,
            arguments_json=str(arguments),
        )
        future = self.agent_client.call_async(request)
        future.add_done_callback(lambda done: self._on_tool_response(
            call_id, name, session_id, turn_serial,
            {"argument_length": len(arguments), "top_fields": top_fields}, done,
            generation=generation,
        ))

    def _on_tool_response(
        self, call_id, name, session_id, turn_serial, argument_meta, future,
        generation=None,
    ):
        with self._lock:
            if (
                generation is not None
                and not self._state.accepts(generation)
                or session_id != self._state.session_id
                or turn_serial != self._state.turn_serial
            ):
                self._tool_trace(
                    turn_serial=turn_serial, tool=name, call_id=call_id,
                    argument_length=argument_meta["argument_length"],
                    parse_status="stale_response",
                    top_fields=argument_meta["top_fields"],
                    repair_count=self._state.tool_argument_errors,
                )
                return
        try:
            response = future.result()
            payload = json.loads(response.result_json or "{}")
            result = {
                "accepted": bool(response.accepted), "status": response.status,
                "preview_id": response.preview_id, "error_code": response.error_code,
                **payload,
            }
        except Exception as exc:
            result = {
                "accepted": False, "status": "rejected",
                "error_code": "TASK_SERVICE_ERROR", "detail": str(exc),
            }
            response = None
        if (
            not result.get("accepted")
            and result.get("error_code") in (
                "MODEL_TOOL_ARGUMENTS_INVALID", "MODEL_RESPONSE_INVALID",
            )
        ):
            self._protocol_failure(
                call_id, name, result["error_code"],
                str(result.get("detail", "tool schema validation failed")),
                argument_meta["argument_length"], argument_meta["top_fields"],
                turn_serial,
            )
            return
        if name == "inspect_scene" and result.get("include_rgb"):
            attached, error_code = self._append_latest_rgb()
            result["rgb_attached"] = attached
            if error_code:
                result["rgb_error"] = error_code
        if (
            response is not None and response.accepted and response.status == "ready"
            and response.preview_id and not self._send_execute(response.preview_id)
        ):
            result.update({
                "accepted": False, "status": "rejected",
                "error_code": "EXECUTION_SERVICE_UNAVAILABLE",
                "spoken_text": "机械臂执行服务不可用，本次任务未执行。",
            })
        model_result = compact_tool_result(name, result)
        self._tool_trace(
            turn_serial=turn_serial, tool=name, call_id=call_id,
            argument_length=argument_meta["argument_length"],
            parse_status="accepted" if model_result["accepted"] else "rejected",
            top_fields=argument_meta["top_fields"],
            repair_count=self._state.tool_argument_errors,
            preview_status=result.get("status", ""),
            error_code=result.get("error_code", ""),
        )
        self._return_tool(call_id, model_result)
        instruction = result.get("spoken_text")
        self._submit_response(
            turn_serial,
            spoken_only_instructions(instruction) if instruction else None,
        )

    def _protocol_failure(
        self, call_id, name, error_code, detail, argument_length,
        top_fields, turn_serial,
    ):
        with self._lock:
            self._state.tool_argument_errors += 1
            repair_count = self._state.tool_argument_errors
            final = repair_count >= 2
        self._tool_trace(
            turn_serial=turn_serial, tool=name, call_id=call_id,
            argument_length=argument_length, parse_status="invalid",
            top_fields=top_fields, error_path=detail,
            repair_count=repair_count, error_code=error_code,
        )
        value, instructions = protocol_failure_response(
            name, error_code, detail, final=final
        )
        self._return_tool_and_respond(call_id, value, instructions=instructions)

    def _tool_trace(self, **values):
        values.setdefault("session", self._state.session_id)
        self.get_logger().debug(
            "QWEN_TOOL_TRACE "
            + json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        )

    def _append_latest_rgb(self):
        with self._lock:
            image, age = self._latest_image, time.monotonic() - self._latest_image_stamp
            provider = self._provider
        if provider is None or not provider.supports_images:
            return False, "MODEL_RGB_UNSUPPORTED"
        if image is None or age > 1.0:
            return False, "VISION_RGB_STALE"
        try:
            import cv2
            from cv_bridge import CvBridge
            frame = CvBridge().imgmsg_to_cv2(image, desired_encoding="bgr8")
            scale = min(1.0, 1280 / frame.shape[1], 720 / frame.shape[0])
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            encoded = None
            for quality in (75, 65, 55, 45):
                ok, candidate = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok and candidate.nbytes <= 190000:
                    encoded = candidate.tobytes()
                    break
            if encoded is None:
                return False, "VISION_RGB_ENCODING_FAILED"
            provider.send_image(encoded)
            return True, ""
        except Exception as exc:
            self.get_logger().warning(f"RGB attachment skipped: {exc}")
            return False, "VISION_RGB_ENCODING_FAILED"

    def _return_tool(self, call_id, value):
        try:
            self._provider.send_tool_result(call_id, value)
        except Exception as exc:
            self.get_logger().error(f"Qwen tool result failed: {exc}")

    def _return_tool_and_respond(self, call_id, value, instructions=None):
        self._return_tool(call_id, value)
        with self._lock:
            turn_serial = self._state.turn_serial
        self._submit_response(turn_serial, instructions)

    def _send_execute(self, preview_id):
        if not self.execute_client.wait_for_server(timeout_sec=0.0):
            return False
        with self._lock:
            self._executing = True
            session_id = self._state.session_id
            self._execution_turn = self._state.turn_serial
        future = self.execute_client.send_goal_async(
            ExecutePreview.Goal(session_id=session_id, preview_id=preview_id)
        )
        future.add_done_callback(self._on_goal)
        return True

    def _on_goal(self, future):
        try:
            goal = future.result()
            if not goal.accepted:
                raise RuntimeError("execution goal rejected")
            goal.get_result_async().add_done_callback(self._on_result)
        except Exception as exc:
            self._finish_execution("FAILED", str(exc))

    def _on_result(self, future):
        try:
            result = future.result().result
            self._finish_execution(result.terminal_state, result.message)
        except Exception as exc:
            self._finish_execution("FAILED", str(exc))

    def _finish_execution(self, state, message):
        with self._lock:
            self._executing = False
            self._state.last_activity = time.monotonic()
            turn_serial = self._execution_turn
            self._execution_turn = -1
            connected = self._state.connected and self._provider is not None
        text = "任务已完成。" if state in ("COMPLETED", "HOLDING") else f"任务失败：{message}"
        if connected:
            self._submit_response(
                max(turn_serial, 0), spoken_only_instructions(text)
            )
        else:
            with self._lock:
                self._pending_status_text = text

    def _drain_events(self):
        for _ in range(100):
            try:
                self._handle_event(self._events.get_nowait())
            except queue.Empty:
                break

    def _tick(self):
        now = time.monotonic()
        with self._lock:
            timeout_phase = self._state.timeout_phase(
                now,
                float(self.get_parameter("response_transition_timeout_sec").value),
                float(self.get_parameter("idle_timeout_sec").value),
            )
            timeout_turn = self._state.response_turn
            timeout_response_id = self._state.response_id
            expired = (
                self._state.active and not self._executing
                and not self._state.response_active
                and now - self._state.last_activity
                > float(self.get_parameter("idle_timeout_sec").value)
            )
        if timeout_phase:
            self.get_logger().error("QWEN_RESPONSE_TIMEOUT " + json.dumps({
                "turn_serial": timeout_turn,
                "response_id": timeout_response_id,
                "phase": timeout_phase,
            }, ensure_ascii=False, separators=(",", ":")))
            self._handle_connection_loss(reconnect=True)
        elif expired:
            self._end_session(reconnect=True)

    def _handle_connection_loss(self, reconnect):
        with self._lock:
            provider = self._provider
            self._provider = None
            preserve = self._state.connection_lost(reconnect)
            self._audio_buffer.clear()
        self._player.interrupt()
        self._publish_mode("MUTED")
        if not preserve:
            self._end_session(reconnect=False)
        if provider is not None:
            threading.Thread(target=provider.close, daemon=True).start()
        if reconnect:
            self._start_connect(delay=1.0)

    def _end_session(self, reconnect=False):
        with self._lock:
            old_session = self._state.end_session()
            provider = self._provider
            if reconnect:
                self._state.connected = False
                self._state.configured = False
                self._state.connecting = False
                self._provider = None
            self._execution_turn = -1
            self._pending_status_text = ""
            self._audio_buffer.clear()
        self._player.interrupt()
        if old_session:
            self.clear_pub.publish(String(data=old_session))
        self._publish_mode("WAKE_ONLY" if self._state.connected else "MUTED")
        if reconnect:
            if provider is not None:
                threading.Thread(target=provider.close, daemon=True).start()
            self._start_connect(delay=1.0)

    def destroy_node(self):
        with self._lock:
            self._shutting_down = True
            provider = self._provider
        self._player.interrupt()
        if provider is not None:
            threading.Thread(target=provider.close, daemon=True).start()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceRealtimeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
