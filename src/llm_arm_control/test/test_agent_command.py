import threading
from pathlib import Path
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
import pytest
from scipy.spatial.transform import Rotation

from llm_arm_control_nodes.task.visual_task_planner import VisualTaskPlannerMixin
from llm_arm_control_nodes.realtime_provider import RealtimeEvent
from llm_arm_control_nodes.realtime_session import (
    RealtimeSessionState,
    ResponseRequest,
)
from llm_arm_control_nodes.agent_protocol import (
    compact_tool_result,
    normalize_tool_arguments,
    parse_agent_tool_call,
    scene_response_instructions,
    tool_argument_shape,
)
from llm_arm_control_nodes.task_logic import (
    PreviewFailure,
    TaskPlan,
)
from llm_arm_control_nodes.voice_realtime_node import VoiceRealtimeNode


PACKAGE = Path(__file__).resolve().parents[1]


def _logger(**handlers):
    def noop(_message):
        pass

    return SimpleNamespace(
        debug=handlers.get("debug", noop),
        info=handlers.get("info", noop),
        warning=handlers.get("warning", noop),
        error=handlers.get("error", noop),
    )


def _state(**values):
    state = RealtimeSessionState()
    for name, value in values.items():
        setattr(state, name, value)
    return state


class _Provider:
    supports_images = True

    def __init__(self):
        self.audio = []
        self.responses = []
        self.cancelled = 0
        self.closed = 0

    def send_audio(self, value):
        self.audio.append(value)

    def create_response(self, instructions=None):
        self.responses.append(instructions)

    def cancel_response(self):
        self.cancelled += 1

    def close(self):
        self.closed += 1


def test_inspect_scene_arguments_are_bounded():
    name, value = parse_agent_tool_call(
        "inspect_scene", {"include_rgb": True}, allowed_tools=("inspect_scene",)
    )
    assert name == "inspect_scene" and value == {"include_rgb": True}


def test_agent_service_replaces_old_text_instruction_service():
    text = (PACKAGE / "llm_arm_control_nodes" / "llm_control_task_server.py").read_text()
    assert "/llm_control/agent_command" in text
    assert "AgentCommand" in text
    assert "with self._command_lock" in text
    assert '"ROBOT_INITIALIZING"' in text


def test_scene_inspection_is_nonblocking_and_old_voice_timing_is_gone():
    server = (PACKAGE / "llm_arm_control_nodes" / "llm_control_task_server.py").read_text()
    planner = (
        PACKAGE / "llm_arm_control_nodes" / "task" / "visual_task_planner.py"
    ).read_text()
    inspect_branch = server.split("def _handle_inspect_scene_tool", 1)[1].split(
        "def _handle_task_tool", 1
    )[0]
    assert "current_frame()" in inspect_branch
    assert "wait_for_planning_metadata" not in inspect_branch
    assert "_voice_timing" not in planner


def test_realtime_uploads_zero_audio_bytes_before_wake():
    provider = _Provider()
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _state=_state(active=False, connected=True, audio_upload_enabled=False),
        _provider=provider,
        _audio_buffer=bytearray(),
    )
    VoiceRealtimeNode._on_audio(
        node, SimpleNamespace(audio=SimpleNamespace(data=b"\x00" * 3200))
    )
    assert provider.audio == []


def test_wake_is_rejected_until_qwen_session_is_configured():
    node = SimpleNamespace(
        _lock=threading.RLock(), _state=_state(configured=False),
    )
    VoiceRealtimeNode._on_wake(node, SimpleNamespace())
    assert node._state.active is False


def test_stale_realtime_events_are_ignored_after_reconnect():
    node = SimpleNamespace(_state=_state(generation=2))
    VoiceRealtimeNode._handle_event(node, RealtimeEvent("session_ready", 1))
    assert not node._state.connected


def test_executing_robot_rejects_new_robot_tool_without_queueing():
    returned = []
    node = SimpleNamespace(
        _lock=threading.RLock(), _executing=True,
        _state=_state(turn_serial=1, session_id="session"),
        _return_tool_and_respond=lambda call_id, value: returned.append((call_id, value)),
    )
    VoiceRealtimeNode._call_tool(node, {
        "call_id": "call", "name": "set_gripper", "arguments": "{}",
    })
    assert returned == [("call", {
        "accepted": False, "status": "rejected", "error_code": "ROBOT_BUSY",
    })]


def test_voice_safety_words_are_handled_as_plain_conversation():
    requested = []
    node = SimpleNamespace(
        _state=_state(generation=1), _lock=threading.RLock(),
        _audio_trace=lambda **_values: None,
        _request_turn_response=lambda turn: requested.append(turn),
        get_logger=_logger,
    )
    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "transcription_completed", 1,
        {"item_id": "item", "transcript": "停止"},
    ))

    assert requested == [1]
    assert not node._state.ignored_turns


def test_execution_is_not_reported_sent_when_action_server_is_unavailable():
    node = SimpleNamespace(
        execute_client=SimpleNamespace(wait_for_server=lambda timeout_sec: False),
    )
    assert VoiceRealtimeNode._send_execute(node, "preview") is False


def test_tool_arguments_are_normalized_before_ros_service():
    assert normalize_tool_arguments(' {"include_rgb": false} ') == '{"include_rgb":false}'


def test_visual_tool_argument_shape_reports_types_without_values():
    assert tool_argument_shape("submit_visual_task", {
        "scene_id": "scene-1", "source_entity_ids": ["entity-1"],
        "destination_entity_id": "entity-2",
    }) == {
        "scene_id_present": True, "source_ids_type": "array",
        "source_ids_count": 1, "destination_id_present": True,
    }


def test_visual_entity_id_errors_are_specific():
    common = {
        "operation": "pick", "scene_id": "scene-1",
        "destination_entity_id": "",
    }
    with pytest.raises(ValueError, match="source_entity_ids must contain"):
        parse_agent_tool_call(
            "submit_visual_task", {**common, "source_entity_ids": {}},
            allowed_tools=("submit_visual_task",),
        )
    with pytest.raises(ValueError, match="0 to 10"):
        parse_agent_tool_call(
            "submit_visual_task", {**common, "source_entity_ids": [str(i) for i in range(11)]},
            allowed_tools=("submit_visual_task",),
        )


def test_malformed_tool_arguments_get_one_repair_then_stop():
    returned = []
    node = SimpleNamespace(
        _lock=threading.RLock(), _executing=False,
        _state=_state(turn_serial=1, session_id="session"),
        get_logger=_logger,
        _return_tool_and_respond=lambda call_id, value, instructions=None: returned.append(
            (call_id, value, instructions)
        ),
    )
    node._tool_trace = lambda **values: VoiceRealtimeNode._tool_trace(node, **values)
    node._protocol_failure = lambda *args: VoiceRealtimeNode._protocol_failure(node, *args)
    event = {"call_id": "call", "name": "set_gripper", "arguments": '{"state":'}

    VoiceRealtimeNode._call_tool(node, event)
    VoiceRealtimeNode._call_tool(node, event)

    assert returned[0][1]["error_code"] == "MODEL_TOOL_ARGUMENTS_INVALID"
    assert "spoken_text" not in returned[0][1]
    assert "按schema重调同一工具" in returned[0][2]
    assert returned[1][1]["spoken_text"].startswith("模型生成的任务参数不完整")


def test_local_schema_failure_uses_the_same_single_repair_budget():
    returned = []
    response = SimpleNamespace(
        accepted=False, status="rejected", preview_id="",
        error_code="MODEL_RESPONSE_INVALID",
        result_json='{"detail":"set_gripper contains unknown fields: extra"}',
    )
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _state=_state(turn_serial=3, session_id="session"),
        get_logger=_logger,
        _return_tool_and_respond=lambda call_id, value, instructions=None: returned.append(
            (call_id, value, instructions)
        ),
    )
    node._tool_trace = lambda **values: VoiceRealtimeNode._tool_trace(node, **values)
    node._protocol_failure = lambda *args: VoiceRealtimeNode._protocol_failure(node, *args)
    future = SimpleNamespace(result=lambda: response)

    VoiceRealtimeNode._on_tool_response(
        node, "call", "set_gripper", "session", 3,
        {"argument_length": 30, "top_fields": ["extra", "state"]}, future,
    )

    assert returned[0][1]["error_code"] == "MODEL_RESPONSE_INVALID"
    assert returned[0][1]["expected_fields"] == "state"
    assert "unknown fields: extra" in returned[0][2]


def test_successful_tool_output_is_compact_and_omits_preview_details():
    result = compact_tool_result("set_gripper", {
        "accepted": True, "status": "ready", "error_code": "",
        "spoken_text": "准备闭合夹爪。", "preview_id": "local-only",
        "actions": [{"type": "set_gripper"}], "detections": [1],
        "base_xyz": [0.1, 0.2, 0.3],
    })
    assert result == {
        "accepted": True, "status": "ready", "error_code": "",
        "spoken_text": "准备闭合夹爪。",
    }


def test_twenty_consecutive_ready_results_execute_once_and_stay_compact():
    outputs, previews, responses = [], [], []
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _state=_state(turn_serial=1, session_id="session"),
        _return_tool=lambda call_id, value: outputs.append((call_id, value)),
        _send_execute=lambda preview_id: previews.append(preview_id) or True,
        _tool_trace=lambda **_values: None,
        _submit_response=lambda turn, instructions=None: responses.append(
            {"turn": turn, "instructions": instructions}
        ),
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )
    for index in range(20):
        response = SimpleNamespace(
            accepted=True, status="ready", preview_id=f"preview-{index}",
            error_code="", result_json=(
                '{"spoken_text":"准备闭合夹爪。","actions":[{"type":"set_gripper"}]}'
            ),
        )
        VoiceRealtimeNode._on_tool_response(
            node, f"call-{index}", "set_gripper", "session", 1,
            {"argument_length": 17, "top_fields": ["state"]},
            SimpleNamespace(result=lambda response=response: response),
        )

    assert previews == [f"preview-{index}" for index in range(20)]
    assert len(outputs) == 20
    assert all("actions" not in value and "preview_id" not in value for _, value in outputs)


def test_scene_context_is_injected_before_turn_response():
    calls = []
    scene = {"entities": [{"class_name": "box"}], "scene_error": ""}
    instructions = scene_response_instructions(scene)
    assert "CURRENT_YOLO_SCENE=" in instructions
    assert '"class_name":"box"' in instructions

    node = SimpleNamespace(
        _lock=threading.RLock(),
        _state=_state(
            active=True, session_id="session", turn_serial=2,
            item_turns={"item-2": 2},
        ),
        _provider=object(),
        _audio_trace=lambda **_values: None,
        _submit_response=lambda turn, instructions=None: calls.append({
            "turn": turn, "instructions": instructions,
        }),
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )
    VoiceRealtimeNode._create_turn_response(node, "session", 2, scene)
    assert calls and "CURRENT_YOLO_SCENE=" in calls[0]["instructions"]


def test_residual_wake_item_does_not_poison_next_command_without_speech_started():
    requested = []
    node = SimpleNamespace(
        _state=_state(generation=1), _lock=threading.RLock(),
        _player=SimpleNamespace(interrupt=lambda: None),
        _cancel_active_response=lambda _reason: False,
        _request_turn_response=lambda turn: requested.append(turn),
        _audio_trace=lambda **_values: None,
        get_logger=_logger,
    )

    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "speech_started", 1, {"item_id": "wake-item"},
    ))
    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "transcription_completed", 1,
        {"item_id": "wake-item", "transcript": "小鹏小鹏"},
    ))
    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "transcription_completed", 1,
        {"item_id": "command-item", "transcript": "关闭夹爪"},
    ))

    assert requested == [2]
    assert node._state.ignored_turns == {1}


def test_duplicate_qwen_input_item_is_handled_once():
    requested = []
    node = SimpleNamespace(
        _state=_state(generation=1), _lock=threading.RLock(),
        _player=SimpleNamespace(interrupt=lambda: None),
        _request_turn_response=lambda turn: requested.append(turn),
        _audio_trace=lambda **_values: None,
        get_logger=_logger,
    )
    event = RealtimeEvent(
        "transcription_completed", 1,
        {"item_id": "command-item", "transcript": "关闭夹爪"},
    )

    VoiceRealtimeNode._handle_event(node, event)
    VoiceRealtimeNode._handle_event(node, event)

    assert requested == [1]


def test_realtime_uses_manual_response_and_sensor_qos():
    node_source = (PACKAGE / "llm_arm_control_nodes" / "voice_realtime_node.py").read_text()
    provider_source = (PACKAGE / "llm_arm_control_nodes" / "realtime_provider.py").read_text()
    assert '"create_response": False' in provider_source
    assert '"interrupt_response": False' in provider_source
    assert "qos_profile_sensor_data" in node_source
    assert "max_tokens=2048" in provider_source


def test_response_done_logs_status_without_response_content():
    logs = []
    node = SimpleNamespace(
        _state=_state(generation=1), _lock=threading.RLock(),
        get_logger=lambda: SimpleNamespace(debug=lambda message: logs.append(message)),
        _finish_response=lambda _response_id="": None,
    )
    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "response_done", 1, {
            "response_id": "response-1", "status": "incomplete",
            "detail_type": "incomplete", "reason": "max_tokens",
        },
    ))

    assert logs == [
        'QWEN_RESPONSE_TRACE {"response_id":"response-1","turn_serial":-1,'
        '"status":"incomplete","detail_type":"incomplete","reason":"max_tokens",'
        '"cancelled":false,"pending_turn":-1}'
    ]


def test_qwen_responses_are_single_flight_and_latest_turn_waits():
    provider = _Provider()
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _state=_state(
            active=True, connected=True, session_id="session", turn_serial=2,
        ),
        _provider=provider,
        _audio_trace=lambda **_values: None,
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )
    node._submit_response = lambda turn, instructions=None: (
        VoiceRealtimeNode._submit_response(node, turn, instructions)
    )

    assert node._submit_response(1, "first")
    assert node._submit_response(2, "second")
    assert provider.responses == ["first"]

    VoiceRealtimeNode._finish_response(node)

    assert provider.responses == ["first", "second"]
    assert node._state.response_active and node._state.response_turn == 2


def test_new_speech_cancels_active_response_and_stale_tool_is_ignored():
    provider, tool_calls = _Provider(), []
    node = SimpleNamespace(
        _state=_state(
            generation=1, response_active=True, response_turn=4,
            response_id="response-4", turn_serial=4,
        ),
        _lock=threading.RLock(), _provider=provider,
        _player=SimpleNamespace(interrupt=lambda: None),
        _audio_trace=lambda **_values: None,
        _cancel_active_response=lambda reason: VoiceRealtimeNode._cancel_active_response(
            node, reason
        ),
        _send_response_cancel=lambda turn, reason: VoiceRealtimeNode._send_response_cancel(
            node, turn, reason
        ),
        _call_tool=lambda event, turn_serial=None: tool_calls.append((event, turn_serial)),
        get_logger=lambda: SimpleNamespace(
            warning=lambda _message: None, info=lambda _message: None,
        ),
    )

    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "speech_started", 1, {"item_id": "item-5"},
    ))
    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "tool_call", 1, {
            "response_id": "response-4", "call_id": "stale",
            "name": "set_gripper",
        },
    ))

    assert provider.cancelled == 1
    assert node._state.response_cancelled and node._state.response_cancel_sent
    assert tool_calls == []


def test_precreated_interruption_cancels_on_created_and_dispatches_latest():
    provider = _Provider()
    node = SimpleNamespace(
        _state=_state(
            generation=1, response_active=True, response_turn=7,
            response_started_at=1.0, response_transition_started_at=1.0,
            pending_response=ResponseRequest("session", 8, "latest"),
            active=True, connected=True, session_id="session", turn_serial=8,
        ),
        _lock=threading.RLock(), _provider=provider,
        _audio_trace=lambda **_values: None,
        get_logger=_logger,
    )
    node._send_response_cancel = lambda turn, reason: (
        VoiceRealtimeNode._send_response_cancel(node, turn, reason)
    )
    node._submit_response = lambda turn, instructions=None: (
        VoiceRealtimeNode._submit_response(node, turn, instructions)
    )
    node._finish_response = lambda response_id="": (
        VoiceRealtimeNode._finish_response(node, response_id)
    )

    assert VoiceRealtimeNode._cancel_active_response(node, "new_speech")
    assert provider.cancelled == 0
    assert node._state.response_cancelled and not node._state.response_cancel_sent

    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "response_started", 1, {"response_id": "response-7"},
    ))
    assert provider.cancelled == 1
    assert node._state.response_cancel_sent

    VoiceRealtimeNode._handle_event(node, RealtimeEvent(
        "response_done", 1,
        {"response_id": "response-7", "status": "cancelled"},
    ))
    assert provider.responses == ["latest"]
    assert node._state.response_active and node._state.response_turn == 8


@pytest.mark.parametrize((
    "phase", "response_id", "cancel_sent", "response_started", "transition_started",
), (
    ("create_wait", "", False, 90.0, 90.0),
    ("cancel_wait", "response-7", True, 98.0, 90.0),
    ("response_max", "response-7", False, 60.0, 0.0),
))
def test_response_watchdog_reconnects_instead_of_deadlocking(
    monkeypatch, phase, response_id, cancel_sent, response_started,
    transition_started,
):
    reconnects, logs = [], []
    node = SimpleNamespace(
        _lock=threading.RLock(), _executing=False,
        _state=_state(
            active=True, response_active=True, response_turn=7,
            response_id=response_id, response_cancel_sent=cancel_sent,
            response_started_at=response_started,
            response_transition_started_at=transition_started,
            last_activity=100.0,
        ),
        get_parameter=lambda name: SimpleNamespace(value={
            "response_transition_timeout_sec": 5.0,
            "idle_timeout_sec": 30.0,
        }[name]),
        get_logger=lambda: SimpleNamespace(error=lambda message: logs.append(message)),
        _handle_connection_loss=lambda reconnect: reconnects.append(reconnect),
        _end_session=lambda reconnect=False: None,
    )
    monkeypatch.setattr(
        "llm_arm_control_nodes.voice_realtime_node.time.monotonic", lambda: 100.0
    )

    VoiceRealtimeNode._tick(node)

    assert reconnects == [True]
    assert logs and f'"phase":"{phase}"' in logs[0]


def test_wake_ack_pcm_is_packaged_raw_24khz_mono_asset():
    asset = PACKAGE / "audio" / "wake_ack_zh.pcm"
    assert asset.is_file()
    assert asset.stat().st_size > 2400
    assert asset.stat().st_size % 2 == 0


def test_wake_ack_failure_still_opens_input_without_buffering_echo():
    errors = []
    node = SimpleNamespace(
        _player=SimpleNamespace(play_file=lambda _path: False),
        _wake_ack_pcm="missing.pcm",
        _lock=threading.RLock(),
        _state=_state(active=True, session_id="session"),
        _audio_buffer=bytearray(b"echo"),
        get_parameter=lambda _name: SimpleNamespace(value=0),
        get_logger=lambda: SimpleNamespace(error=lambda message: errors.append(message)),
    )

    VoiceRealtimeNode._play_wake_ack(node, "session")

    assert node._state.audio_upload_enabled
    assert node._audio_buffer == bytearray()
    assert errors == [
        "Wake acknowledgement was not played; opening input after bounded fallback."
    ]


@pytest.mark.parametrize(("box_metadata", "expected_code"), (
    ([], "BATCH_DESTINATION_NOT_VISIBLE"),
    ([{
        "index": 4,
        "class_name": "box",
        "unavailable_reason": "VISION_TF_UNAVAILABLE",
    }], "VISION_TF_UNAVAILABLE"),
))
def test_stable_batch_source_preserves_destination_failure(
    monkeypatch, box_metadata, expected_code,
):
    moments = iter((0.0, 0.0, 0.1, 1.0))
    frames = iter((
        {"stamp_ns": 1, "result_seq": 1},
        {"stamp_ns": 2, "result_seq": 2},
    ))
    source_metadata = [{
        "index": 2,
        "class_name": "elongated_object",
        "base_xyz": [0.0045, 0.0, 0.0],
        "result_seq": 2,
    }]
    traces = []
    node = SimpleNamespace(
        batch_rebind_timeout_sec=0.5,
        batch_rebind_stable_frames=2,
        batch_rebind_max_distance_m=0.10,
        pick_classes={"elongated_object"},
        place_classes={"box"},
        perception=SimpleNamespace(
            current_frame=lambda: next(frames),
            planning_metadata=lambda _frame, include_unavailable: (
                source_metadata + box_metadata
            ),
        ),
        _refresh_yolo_health=lambda: True,
        _exclude_box_contents=lambda metadata: metadata,
        _batch_trace=lambda **values: traces.append(values),
    )
    frozen_source = SimpleNamespace(
        class_name="elongated_object", xyz=(0.0, 0.0, 0.0),
        result_seq=0, index=1,
    )
    frozen_destination = SimpleNamespace(
        class_name="box", xyz=(0.5, 0.3, 0.0), container_xyz=None, index=4,
    )
    action = {"source": frozen_source, "destination": frozen_destination}
    record = SimpleNamespace(
        session_id="session", preview=SimpleNamespace(preview_id="preview")
    )
    monkeypatch.setattr(
        "llm_arm_control_nodes.task.visual_task_planner.time.monotonic",
        lambda: next(moments),
    )
    monkeypatch.setattr(
        "llm_arm_control_nodes.task.visual_task_planner.time.sleep", lambda _value: None
    )

    with pytest.raises(PreviewFailure) as failure:
        VisualTaskPlannerMixin._rebind_batch_action(
            node, record, action, [action], ordinal=3, total=3
        )

    assert failure.value.error_code == expected_code
    assert "已完成 2 次抓放" in str(failure.value)
    assert "剩余目标稳定" in str(failure.value)
    assert traces[-1]["failure_stage"] == "destination"
    assert traces[-1]["source_stable_frames"] == 2
    assert traces[-1]["destination_stable_frames"] == 0


def test_batch_box_must_be_currently_visible_for_two_frames(monkeypatch):
    moments = iter((0.0, 0.0, 0.1, 0.2))
    frames = iter((
        {"stamp_ns": 1, "result_seq": 1},
        {"stamp_ns": 2, "result_seq": 2},
        {"stamp_ns": 3, "result_seq": 3},
    ))
    source = SimpleNamespace(
        class_name="elongated_object", xyz=(0.0, 0.0, 0.0),
        result_seq=0, index=1,
    )
    frozen_destination = SimpleNamespace(
        class_name="box", xyz=(0.5, 0.3, 0.0), container_xyz=None, index=4,
    )
    current_destination = SimpleNamespace(index=8)
    metadata = iter((
        [{"index": 2, "class_name": "elongated_object", "base_xyz": [0.0, 0.0, 0.0]}],
        [
            {"index": 2, "class_name": "elongated_object", "base_xyz": [0.0, 0.0, 0.0]},
            {"index": 8, "class_name": "box", "base_xyz": [0.5, 0.3, 0.0]},
        ],
        [
            {"index": 2, "class_name": "elongated_object", "base_xyz": [0.0, 0.0, 0.0]},
            {"index": 8, "class_name": "box", "base_xyz": [0.5, 0.3, 0.0]},
        ],
    ))
    traces = []
    node = SimpleNamespace(
        batch_rebind_timeout_sec=1.0,
        batch_rebind_stable_frames=2,
        batch_rebind_max_distance_m=0.10,
        pick_classes={"elongated_object"},
        place_classes={"box"},
        perception=SimpleNamespace(
            current_frame=lambda: next(frames),
            planning_metadata=lambda _frame, include_unavailable: next(metadata),
            resolve_candidate_detailed=lambda _index, _frame: (current_destination, ""),
        ),
        _refresh_yolo_health=lambda: True,
        _exclude_box_contents=lambda value: value,
        _batch_trace=lambda **values: traces.append(values),
        _pick_place_preview_poses=lambda _source, _destination: {},
        _check_pose=lambda _pose: None,
    )
    action = {"source": source, "destination": frozen_destination}
    record = SimpleNamespace(
        session_id="session", preview=SimpleNamespace(preview_id="preview")
    )
    monkeypatch.setattr(
        "llm_arm_control_nodes.task.visual_task_planner.time.monotonic",
        lambda: next(moments),
    )
    monkeypatch.setattr(
        "llm_arm_control_nodes.task.visual_task_planner.time.sleep", lambda _value: None
    )

    rebound = VisualTaskPlannerMixin._rebind_batch_action(
        node, record, action, [action], ordinal=1, total=1
    )

    assert rebound["destination"] is current_destination
    assert [item["destination_stable_frames"] for item in traces[:3]] == [0, 1, 2]


def test_realtime_error_details_include_nested_server_error():
    assert VoiceRealtimeNode._realtime_error_details(RealtimeEvent(
        "provider_error", 7,
        {"code": "invalid_request", "error_type": "request_error", "message": "busy"},
    )) == {
        "event_type": "provider_error", "code": "invalid_request",
        "error_type": "request_error", "message": "busy",
        "close_code": "", "generation": 7,
    }


def test_transient_disconnect_preserves_active_session_and_recovers_without_wake():
    modes, reconnects, responses = [], [], []
    node = SimpleNamespace(
        _lock=threading.RLock(), _provider=None,
        _state=_state(
            generation=2, active=True, session_id="session", connected=True,
            configured=True, turn_serial=3, item_turns={"item": 1},
            handled_input_items={"item"}, response_active=True,
            response_turn=3, response_id="response-3", response_started_at=1.0,
        ),
        _audio_buffer=bytearray(b"partial"), _pending_status_text="",
        _player=SimpleNamespace(interrupt=lambda: None),
        _publish_mode=lambda mode: modes.append(mode),
        _start_connect=lambda delay=0.0: reconnects.append(delay),
        _submit_response=lambda turn, instructions=None: responses.append((turn, instructions)),
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
    )

    VoiceRealtimeNode._handle_connection_loss(node, reconnect=True)

    assert node._state.session_id == "session" and node._state.recover_active_session
    assert node._state.active and not node._state.connected
    assert node._audio_buffer == bytearray()
    assert reconnects == [1.0]

    VoiceRealtimeNode._handle_event(node, RealtimeEvent("session_ready", 2))

    assert node._state.active and node._state.configured
    assert modes[-1] == "MUTED"
    assert responses == [(3, "只朗读以下内容，不添加或改写：连接已恢复，请重新说刚才的指令。")]


def test_combined_relative_steps_accumulate_base_pose_and_rotation():
    current = PoseStamped()
    current.pose.position.x = 0.4
    current.pose.position.y = -0.1
    current.pose.position.z = 0.3
    current.pose.orientation.w = 1.0

    def pose_from_xyz_quat(xyz, quat):
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = xyz
        pose.pose.orientation.x, pose.pose.orientation.y = quat[:2]
        pose.pose.orientation.z, pose.pose.orientation.w = quat[2:]
        return pose

    planner = SimpleNamespace(
        perception=SimpleNamespace(current_frame=lambda: None),
        _current_pose=lambda: current,
        _pose_from_xyz_quat=pose_from_xyz_quat,
        _check_pose=lambda _pose: None,
    )
    first = {
        "type": "move_relative",
        "dx": 0.1, "dy": 0.2, "dz": -0.05,
        "droll_deg": 30.0, "dpitch_deg": -20.0, "dyaw_deg": 60.0,
        "frame_id": "base_link",
    }
    second = {
        "type": "move_relative",
        "dx": -0.05, "dy": 0.0, "dz": 0.05,
        "droll_deg": 0.0, "dpitch_deg": 0.0, "dyaw_deg": 0.0,
        "frame_id": "base_link",
    }

    enriched = VisualTaskPlannerMixin._enrich_plan(
        planner, TaskPlan((first, second))
    )

    assert (
        enriched[0]["target_pose"].pose.position.x,
        enriched[0]["target_pose"].pose.position.y,
        enriched[0]["target_pose"].pose.position.z,
    ) == pytest.approx((0.5, 0.1, 0.25))
    assert (
        enriched[1]["target_pose"].pose.position.x,
        enriched[1]["target_pose"].pose.position.y,
        enriched[1]["target_pose"].pose.position.z,
    ) == pytest.approx((0.45, 0.1, 0.30))
    expected = Rotation.from_euler("xyz", [30.0, -20.0, 60.0], degrees=True).as_quat()
    actual = enriched[0]["target_pose"].pose.orientation
    assert [actual.x, actual.y, actual.z, actual.w] == pytest.approx(expected)
