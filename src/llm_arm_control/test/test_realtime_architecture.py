import base64
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from llm_arm_control_nodes.realtime_provider import (
    ProviderConfigurationError,
    QwenRealtimeProvider,
    RealtimeEvent,
    RealtimeProviderConfig,
    create_realtime_provider,
)
from llm_arm_control_nodes.realtime_session import RealtimeSessionState


def _config():
    return RealtimeProviderConfig(
        model="qwen3.8-omni-flash-realtime",
        voice="Tina",
        vad_silence_ms=700,
        temperature=0.1,
        instructions="system",
        tool_names=("inspect_scene",),
    )


def test_provider_factory_rejects_unknown_provider():
    with pytest.raises(ProviderConfigurationError, match="supported: qwen"):
        create_realtime_provider("other", _config(), 1, lambda _event: None)


def test_qwen_provider_requires_cloud_credentials(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_WORKSPACE_ID", raising=False)
    provider = QwenRealtimeProvider(_config(), 1, lambda _event: None)
    with pytest.raises(ProviderConfigurationError, match="DASHSCOPE_API_KEY"):
        provider.connect()


def test_qwen_connect_applies_required_session_contract(monkeypatch):
    updates = []

    class Conversation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def connect(self):
            return None

        def update_session(self, **kwargs):
            updates.append(kwargs)

    dashscope = ModuleType("dashscope")
    dashscope.__path__ = []
    audio = ModuleType("dashscope.audio")
    audio.__path__ = []
    qwen_omni = ModuleType("dashscope.audio.qwen_omni")
    qwen_omni.AudioFormat = SimpleNamespace(
        PCM_16000HZ_MONO_16BIT="pcm16-in",
        PCM_24000HZ_MONO_16BIT="pcm24-out",
    )
    qwen_omni.MultiModality = SimpleNamespace(AUDIO="audio", TEXT="text")
    qwen_omni.OmniRealtimeCallback = object
    qwen_omni.OmniRealtimeConversation = Conversation
    monkeypatch.setitem(sys.modules, "dashscope", dashscope)
    monkeypatch.setitem(sys.modules, "dashscope.audio", audio)
    monkeypatch.setitem(sys.modules, "dashscope.audio.qwen_omni", qwen_omni)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "workspace-1")

    provider = QwenRealtimeProvider(_config(), 1, lambda _event: None)
    provider.connect()

    assert updates[0]["input_audio_format"] == "pcm16-in"
    assert updates[0]["output_audio_format"] == "pcm24-out"
    assert updates[0]["turn_detection_param"] == {
        "create_response": False,
        "interrupt_response": False,
    }
    assert updates[0]["max_tokens"] == 2048
    assert updates[0]["tools"][0]["function"]["name"] == "inspect_scene"


def test_qwen_events_are_normalized_without_leaking_raw_protocol():
    events = []
    provider = QwenRealtimeProvider(_config(), 7, events.append)
    provider._translate({
        "type": "response.function_call_arguments.done",
        "response_id": "response-1",
        "call_id": "call-1",
        "name": "inspect_scene",
        "arguments": '{"include_rgb":false}',
    })
    provider._translate({
        "type": "response.done",
        "response": {
            "id": "response-1",
            "status": "incomplete",
            "status_details": {"type": "incomplete", "reason": "max_tokens"},
            "output": [{"must_not_escape": True}],
        },
    })

    assert events == [
        RealtimeEvent("tool_call", 7, {
            "response_id": "response-1",
            "call_id": "call-1",
            "name": "inspect_scene",
            "arguments": '{"include_rgb":false}',
        }),
        RealtimeEvent("response_done", 7, {
            "response_id": "response-1",
            "status": "incomplete",
            "detail_type": "incomplete",
            "reason": "max_tokens",
        }),
    ]


@pytest.mark.parametrize(("raw", "expected"), (
    ({"type": "session.updated"}, "session_ready"),
    ({"type": "input_audio_buffer.speech_started"}, "speech_started"),
    ({"type": "input_audio_buffer.speech_stopped"}, "speech_stopped"),
    ({"type": "conversation.item.input_audio_transcription.completed"},
     "transcription_completed"),
    ({"type": "conversation.item.input_audio_transcription.failed"},
     "transcription_failed"),
    ({"type": "response.audio.delta"}, "audio_delta"),
    ({"type": "response.audio_transcript.done"}, "response_transcript_done"),
    ({"type": "response.created"}, "response_started"),
))
def test_qwen_common_events_have_provider_neutral_names(raw, expected):
    events = []
    QwenRealtimeProvider(_config(), 3, events.append)._translate(raw)
    assert events[0].kind == expected and events[0].generation == 3


def test_qwen_provider_owns_wire_encoding_for_audio_tools_and_images():
    calls = []

    class Conversation:
        def append_audio(self, value):
            calls.append(("audio", value))

        def create_item(self, value):
            calls.append(("tool", value))

        def append_video(self, value):
            calls.append(("image", value))

    provider = QwenRealtimeProvider(_config(), 1, lambda _event: None)
    provider._conversation = Conversation()
    provider.send_audio(b"\x00\x01")
    provider.send_tool_result("call-1", {"accepted": True})
    provider.send_image(b"jpeg")

    assert calls[0] == ("audio", base64.b64encode(b"\x00\x01").decode("ascii"))
    assert calls[1][0] == "tool"
    assert calls[1][1]["call_id"] == "call-1"
    assert json.loads(calls[1][1]["output"]) == {"accepted": True}
    assert calls[2] == ("image", base64.b64encode(b"jpeg").decode("ascii"))


def test_rgb_request_fails_explicitly_when_provider_has_no_image_capability():
    from llm_arm_control_nodes.voice_realtime_node import VoiceRealtimeNode

    node = SimpleNamespace(
        _lock=threading.RLock(),
        _latest_image=object(),
        _latest_image_stamp=0.0,
        _provider=SimpleNamespace(supports_images=False),
    )
    assert VoiceRealtimeNode._append_latest_rgb(node) == (
        False, "MODEL_RGB_UNSUPPORTED"
    )


def test_session_filters_duplicate_transcripts_and_resets_turn_budget():
    state = RealtimeSessionState(tool_count=2, tool_argument_errors=1)
    first = state.record_transcript("item-1", 10.0)
    duplicate = state.record_transcript("item-1", 11.0)

    assert first == (1, False)
    assert duplicate == (1, True)
    assert state.tool_count == 0 and state.tool_argument_errors == 0


def test_session_single_flight_keeps_only_latest_response():
    state = RealtimeSessionState(
        active=True, connected=True, session_id="session", turn_serial=1
    )
    start, first = state.queue_response(1, "first", 1.0)
    state.turn_serial = 2
    queued, _second = state.queue_response(2, "second", 2.0)
    state.turn_serial = 3
    queued_latest, latest = state.queue_response(3, "latest", 3.0)

    assert start and first.instructions == "first"
    assert not queued and not queued_latest
    assert state.finish_response() == latest


def test_session_defers_cancel_until_response_id_exists():
    state = RealtimeSessionState(
        active=True, connected=True, session_id="session", turn_serial=4
    )
    state.queue_response(4, None, 1.0)
    assert state.request_cancel(2.0) == (4, "")
    assert state.response_started("response-4") == (4, True, -1)
    assert state.request_cancel(3.0) == (4, "response-4")
    assert state.request_cancel(4.0) is None


def test_session_rejects_audio_and_tools_without_the_active_response_id():
    state = RealtimeSessionState(response_active=True, response_turn=4)
    assert state.response_event_turn("") is None
    state.response_started("response-4")
    assert state.response_event_turn("") is None
    assert state.response_event_turn("response-old") is None
    assert state.response_event_turn("response-4") == 4


def test_session_reconnect_discards_unfinished_response_but_keeps_wake_session():
    state = RealtimeSessionState(
        generation=2, active=True, connected=True, configured=True,
        session_id="session", turn_serial=3, response_active=True,
        response_turn=3, response_id="response-3",
    )
    assert state.connection_lost(reconnect=True)
    assert state.session_id == "session" and not state.response_active
    assert state.connection_ready(20.0)
    assert state.active and state.audio_upload_enabled


@pytest.mark.parametrize(("expected", "response_id", "cancel_sent", "started", "transition"), (
    ("create_wait", "", False, 90.0, 90.0),
    ("cancel_wait", "response-1", True, 98.0, 90.0),
    ("response_max", "response-1", False, 60.0, 0.0),
))
def test_session_reports_response_timeout_phase(
    expected, response_id, cancel_sent, started, transition,
):
    state = RealtimeSessionState(
        response_active=True,
        response_id=response_id,
        response_cancel_sent=cancel_sent,
        response_started_at=started,
        response_transition_started_at=transition,
    )
    assert state.timeout_phase(100.0, 5.0, 30.0) == expected
