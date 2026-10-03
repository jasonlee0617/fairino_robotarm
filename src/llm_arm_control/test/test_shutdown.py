from pathlib import Path
import threading
from types import SimpleNamespace

from llm_arm_control_nodes.realtime_session import RealtimeSessionState
from llm_arm_control_nodes.voice_realtime_node import VoiceRealtimeNode


def test_realtime_connection_loss_interrupts_player_and_closes_provider(monkeypatch):
    calls = []

    class ImmediateThread:
        def __init__(self, target, daemon):
            assert daemon
            self.target = target

        def start(self):
            self.target()

    provider = SimpleNamespace(close=lambda: calls.append("closed"))
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _provider=provider,
        _state=RealtimeSessionState(
            active=True, connected=True, configured=True, session_id="session"
        ),
        _audio_buffer=bytearray(b"partial"),
        _player=SimpleNamespace(interrupt=lambda: calls.append("interrupted")),
        _publish_mode=lambda mode: calls.append(mode),
        _start_connect=lambda delay=0.0: calls.append(("reconnect", delay)),
    )
    monkeypatch.setattr(
        "llm_arm_control_nodes.voice_realtime_node.threading.Thread", ImmediateThread
    )

    VoiceRealtimeNode._handle_connection_loss(node, reconnect=True)

    assert calls == ["interrupted", "MUTED", "closed", ("reconnect", 1.0)]


def test_task_server_executor_shutdown_has_timeout():
    source = (
        Path(__file__).resolve().parents[1]
        / "llm_arm_control_nodes" / "llm_control_task_server.py"
    ).read_text(encoding="utf-8")
    assert "executor.shutdown(timeout_sec=2.0)" in source
    assert "SignalHandlerOptions.NO" in source
    assert source.index('node.abort.request_abort("shutdown"') < source.index(
        "executor.shutdown(timeout_sec=2.0)"
    )
