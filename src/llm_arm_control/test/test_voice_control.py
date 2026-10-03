from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import json
import threading
import time

import pytest

from llm_arm_control_nodes.agent_protocol import (
    agent_tool_definitions,
    parse_agent_tool_call,
)
from llm_arm_control_nodes.task_logic import (
    CapabilityInvocation,
    chinese_number,
    invocation_spoken_text,
    sanitize_spoken_text,
    validate_spoken_text,
)
from llm_arm_control_nodes.voice_logic import classify_wake
from llm_arm_control_nodes import voice_launch
from llm_arm_control_nodes.voice_wake_node import VoiceWakeNode


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value,expected", [
    (5, "五"), (10, "十"), (20, "二十"), (0.05, "零点零五"), (-5, "负五"),
])
def test_chinese_number(value, expected):
    assert chinese_number(value) == expected


def test_relative_motion_spoken_text_is_local_and_chinese():
    invocation = CapabilityInvocation(
        skill="arm.move_relative",
        parameters={"steps": [{
            "type": "move_relative",
            "dx": 0.0, "dy": 0.05, "dz": 0.0,
            "droll_deg": 0.0, "dpitch_deg": 0.0, "dyaw_deg": 0.0,
            "frame_id": "base_link",
        }]},
    )
    text = invocation_spoken_text(invocation)
    assert text == "机械臂向左移动五厘米。"
    assert "five" not in text.lower() and "5" not in text


def test_combined_relative_motion_spoken_text_uses_simultaneous_and_then():
    _name, invocation = parse_agent_tool_call(
        "move_relative",
        {"steps": [
            {
                "x_m": 0.10, "y_m": -0.05, "z_m": 0.0,
                "rx_deg": 30.0, "ry_deg": 0.0, "rz_deg": -20.0,
            },
            {
                "x_m": 0.0, "y_m": 0.0, "z_m": 0.02,
                "rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": 0.0,
            },
        ]},
        allowed_tools=("move_relative",),
    )

    assert invocation_spoken_text(invocation) == (
        "机械臂向前移动十厘米，同时向右移动五厘米，"
        "同时绕基座X轴正向旋转三十度，同时绕基座Z轴负向旋转二十度，"
        "然后向上移动二厘米。"
    )


def test_visual_action_count_is_local_chinese_text():
    invocation = CapabilityInvocation(skill="yolo.pick_place")
    assert invocation_spoken_text(invocation, 5) == "准备执行五次抓放任务。"


def test_trusted_fault_speech_and_untrusted_model_speech_have_separate_policy():
    assert sanitize_spoken_text("任务失败，请按 h 复位。") == "任务失败，请按 h 复位。"
    with pytest.raises(ValueError, match="reserved"):
        validate_spoken_text("请复位")


@pytest.mark.parametrize(("value", "expected"), (
    ("小鹏同学", ("小鹏同学", "zh")),
    ("小鹏小鹏！", ("小鹏小鹏", "zh")),
    ("HI, ROBOT", ("Hi Robot", "en")),
))
def test_three_wake_phrases_use_normalized_matching(value, expected):
    assert classify_wake(value) == expected


@pytest.mark.parametrize("value", ("小彭同学", "小朋小朋", "嗨机器人"))
def test_old_wake_aliases_are_rejected(value):
    assert classify_wake(value) == ("", "")


def test_qwen_tool_catalog_has_scene_inspection_and_no_unsafe_tool():
    tools = agent_tool_definitions(("submit_visual_task", "set_gripper", "inspect_scene"))
    names = {item["function"]["name"] for item in tools}
    assert names == {"submit_visual_task", "set_gripper", "inspect_scene"}
    rendered = str(tools).lower()
    assert "trajectory" not in rendered and "home" not in rendered


def test_wake_node_has_no_command_text_publisher():
    source = (PACKAGE / "llm_arm_control_nodes" / "voice_wake_node.py").read_text()
    assert "/voice_control/wake_event" in source


def test_successful_wake_always_emits_source_trace():
    traces, publications = [], []
    node = SimpleNamespace(
        _mode="WAKE_ONLY", _last_wake_at=0.0,
        _kws_hits=Counter(),
        _param=lambda name: {"wake_dedup_sec": 2.0}[name],
        get_logger=lambda: SimpleNamespace(info=lambda message: traces.append(message)),
        wake_pub=SimpleNamespace(publish=lambda message: publications.append(message.data)),
        _spotter=SimpleNamespace(create_stream=lambda: object()),
    )

    assert VoiceWakeNode._publish_wake(node, "小鹏小鹏")
    assert publications
    assert traces[0].startswith("WAKE_TRACE ")
    assert '"source": "kws"' in traces[0]


def test_pulse_module_cleanup_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        voice_launch, "_pactl",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("No such entity")),
    )
    assert voice_launch._unload_pulse_module(None, 42) == []


def test_voice_runtime_requires_flat_package_model_files(monkeypatch, tmp_path):
    sherpa = SimpleNamespace(__version__="1.13.8")
    monkeypatch.setitem(__import__("sys").modules, "sherpa_onnx", sherpa)
    for name in ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"):
        (tmp_path / name).touch()
    with pytest.raises(RuntimeError, match="keywords.txt"):
        voice_launch.validate_voice_runtime(tmp_path)
    (tmp_path / "keywords.txt").touch()
    voice_launch.validate_voice_runtime(tmp_path)


def _source(name, card="0", device="0", *, monitor=False, volume=100):
    return {
        "name": name,
        "mute": False,
        "properties": {
            "device.class": "monitor" if monitor else "sound",
            "alsa.card": card,
            "alsa.device": device,
        },
        "volume": {"front-left": {"value_percent": f"{volume}%"}},
    }


def test_microphone_resolution_rejects_invalid_explicit_device():
    sources = [
        _source("alsa_input.real", "0", "0"),
        _source("alsa_output.hdmi.monitor", "1", "0", monitor=True),
    ]
    assert voice_launch._resolve_pulse_source(
        sources, "auto", "alsa_input.real"
    )["name"] == "alsa_input.real"
    with pytest.raises(RuntimeError, match="plughw:1,0.*not found"):
        voice_launch._resolve_pulse_source(
            sources, "plughw:1,0", "alsa_input.real"
        )


def test_auto_microphone_selection_fails_when_physical_sources_are_ambiguous():
    sources = [_source("mic.one"), _source("mic.two", "2", "0")]
    with pytest.raises(RuntimeError, match="ambiguous.*mic.one.*mic.two"):
        voice_launch._resolve_pulse_source(
            sources, "auto", "alsa_output.hdmi.monitor"
        )


def test_source_level_is_set_and_verified(monkeypatch):
    calls = []

    def fake_pactl(*args):
        calls.append(args)
        if args[:4] == ("-f", "json", "list", "sources"):
            return json.dumps([_source("mic", volume=100)])
        return ""

    monkeypatch.setattr(voice_launch, "_pactl", fake_pactl)
    selected = voice_launch._set_source_level("mic", 100)
    assert selected["name"] == "mic"
    assert ("set-source-mute", "mic", "0") in calls
    assert ("set-source-volume", "mic", "100%") in calls


def test_aec_module_parser_keeps_only_project_owned_modules():
    modules = voice_launch._owned_aec_modules(
        "7\tmodule-echo-cancel\tsource_master=mic sink_master=sink "
        "source_name=llm_aec_source sink_name=llm_aec_sink\t1\n"
        "8\tmodule-echo-cancel\tsource_name=other sink_name=other\t1"
    )
    assert modules == [(7, {
        "source_master": "mic", "sink_master": "sink",
        "source_name": "llm_aec_source", "sink_name": "llm_aec_sink",
    })]


@pytest.mark.parametrize("existing_master,expected_return,expects_load", [
    ("mic", None, False),
    ("old-mic", 9, True),
])
def test_aec_route_is_reused_or_rebuilt(
    monkeypatch, existing_master, expected_return, expects_load
):
    calls = []
    module_line = (
        "7\tmodule-echo-cancel\t"
        f"source_master={existing_master} sink_master=sink "
        "source_name=llm_aec_source sink_name=llm_aec_sink\t1"
    )

    def fake_pactl(*args):
        calls.append(args)
        if args == ("-f", "json", "list", "sources"):
            return json.dumps([_source("mic")])
        if args == ("-f", "json", "list", "sinks"):
            return json.dumps([{"name": "sink"}])
        if args == ("get-default-source",):
            return "mic"
        if args == ("get-default-sink",):
            return "sink"
        if args == ("list", "short", "modules"):
            return module_line
        if args[:2] == ("load-module", "module-echo-cancel"):
            return "9"
        return ""

    monkeypatch.setattr(voice_launch, "_pactl", fake_pactl)
    assert voice_launch.ensure_pulse_aec("auto", 100) == expected_return
    loaded = any(
        call[:2] == ("load-module", "module-echo-cancel") for call in calls
    )
    assert loaded is expects_load
    assert (("unload-module", "7") in calls) is expects_load


def test_first_audio_health_window_reports_low_input_without_debug_flag():
    info, warnings = [], []
    now = time.monotonic()
    node = SimpleNamespace(
        _lock=threading.RLock(), _audio_metrics_started_at=now - 5.0,
        _audio_energy=0.001 ** 2 * 16000, _audio_samples=16000,
        _last_audio_at=now, _audio_frames=50, _audio_total_frames=50,
        _audio_clipped=0, _audio_silent=15900, _audio_peak=0.01,
        _audio_info_seen=True, _audio_valid=True,
        _audio_format={"channels": 1, "sample_rate": 16000,
                       "sample_format": "S16LE"},
        _mode="WAKE_ONLY", _kws_hits=Counter(), _last_audio_issue="",
        _health_reported=False,
        count_publishers=lambda _topic: 1,
        _param=lambda name: {"voice_diagnostics_enabled": False}[name],
        get_logger=lambda: SimpleNamespace(
            info=lambda message: info.append(message),
            warning=lambda message: warnings.append(message),
        ),
    )
    VoiceWakeNode._report_audio_diagnostics(node)
    assert info[0].startswith("VOICE_AUDIO_HEALTH ")
    assert '"reason": "input_level_low"' in warnings[0]
