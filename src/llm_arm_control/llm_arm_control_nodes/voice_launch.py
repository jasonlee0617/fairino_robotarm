"""Shared launch construction for wake-gated realtime voice."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.logging import get_logger
from launch_ros.actions import Node

from myrobot_common.launch_utils.yaml_loader import load_node_parameters_yaml


_LOGGER = get_logger("llm_voice")


def _enabled(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def kws_model_directory() -> Path:
    return Path(get_package_share_directory("llm_arm_control")) / "model" / "kws"


def validate_voice_runtime(model_directory: str | Path) -> None:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError("sherpa_onnx is missing; run setup_voice_runtime.sh") from exc
    if sherpa_onnx.__version__ != "1.13.8":
        raise RuntimeError(
            f"sherpa_onnx 1.13.8 is required, found {sherpa_onnx.__version__}"
        )
    root = Path(model_directory).resolve()
    required = (
        "tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx", "keywords.txt",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"wake runtime assets are missing: {', '.join(missing)}")


def _pactl(*args):
    try:
        result = subprocess.run(
            ["pactl", *args], check=False, capture_output=True, text=True, timeout=5.0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("PulseAudio pactl is unavailable") from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "pactl failed")
    return result.stdout.strip()


def _physical_sources(sources):
    return [
        item for item in sources
        if str(item.get("name", ""))
        and str(item.get("name", "")) != "llm_aec_source"
        and str(item.get("properties", {}).get("device.class", "")) != "monitor"
        and not str(item.get("name", "")).endswith(".monitor")
    ]


def _resolve_pulse_source(sources, requested: str, default_source: str) -> dict:
    physical = _physical_sources(sources)
    requested = str(requested).strip()
    if requested.lower() == "auto":
        default = next(
            (item for item in physical if item.get("name") == default_source), None
        )
        if default is not None:
            return default
        if len(physical) == 1:
            return physical[0]
        names = sorted(str(item.get("name", "")) for item in physical)
        raise RuntimeError(
            "automatic microphone selection is ambiguous; available physical "
            f"sources: {names or ['none']}"
        )

    match = re.fullmatch(r"(?:plug)?hw:(\d+),(\d+)", requested)
    if match:
        card, device = match.groups()
        selected = next((
            item for item in physical
            if str(item.get("properties", {}).get("alsa.card", "")) == card
            and str(item.get("properties", {}).get("alsa.device", "")) == device
        ), None)
    else:
        selected = next(
            (item for item in physical if str(item.get("name", "")) == requested),
            None,
        )
    if selected is None:
        names = sorted(str(item.get("name", "")) for item in physical)
        raise RuntimeError(
            f"microphone {requested!r} was not found; available physical sources: "
            f"{names or ['none']}"
        )
    return selected


def _source_volume_percentages(source: dict) -> list[int]:
    percentages = []
    for channel in (source.get("volume") or {}).values():
        value = str(channel.get("value_percent", "")).strip()
        if value.endswith("%") and value[:-1].isdigit():
            percentages.append(int(value[:-1]))
    return percentages


def _set_source_level(source_name: str, volume_percent: int) -> dict:
    if not 1 <= int(volume_percent) <= 100:
        raise RuntimeError("audio_input_volume_percent must be between 1 and 100")
    _pactl("set-source-mute", source_name, "0")
    _pactl("set-source-volume", source_name, f"{int(volume_percent)}%")
    sources = json.loads(_pactl("-f", "json", "list", "sources") or "[]")
    selected = next(
        (item for item in sources if str(item.get("name", "")) == source_name), None
    )
    if selected is None:
        raise RuntimeError(f"microphone disappeared after level update: {source_name}")
    observed = _source_volume_percentages(selected)
    if bool(selected.get("mute")) or not observed or any(
        value != int(volume_percent) for value in observed
    ):
        raise RuntimeError(
            f"failed to set microphone {source_name!r} to {volume_percent}% and unmuted; "
            f"observed mute={selected.get('mute')!r}, volume={observed}"
        )
    return selected


def _owned_aec_modules(text: str) -> list[tuple[int, dict]]:
    modules = []
    for line in str(text).splitlines():
        columns = line.split("\t", 3)
        if len(columns) < 3 or columns[1] != "module-echo-cancel":
            continue
        try:
            values = dict(
                token.split("=", 1) for token in shlex.split(columns[2]) if "=" in token
            )
            module_id = int(columns[0])
        except (TypeError, ValueError):
            continue
        if (
            values.get("source_name") == "llm_aec_source"
            and values.get("sink_name") == "llm_aec_sink"
        ):
            modules.append((module_id, values))
    return modules


def ensure_pulse_aec(audio_input_device: str, volume_percent: int = 100):
    sources = json.loads(_pactl("-f", "json", "list", "sources") or "[]")
    default_source = _pactl("get-default-source")
    selected = _resolve_pulse_source(sources, audio_input_device, default_source)
    master = str(selected.get("name", ""))
    selected = _set_source_level(master, int(volume_percent))

    sinks = json.loads(_pactl("-f", "json", "list", "sinks") or "[]")
    sink = _pactl("get-default-sink")
    if sink not in {str(item.get("name", "")) for item in sinks}:
        raise RuntimeError(f"default PulseAudio sink is unavailable: {sink!r}")

    owned = _owned_aec_modules(_pactl("list", "short", "modules"))
    matching = next((
        (module_id, values) for module_id, values in owned
        if values.get("source_master") == master
        and values.get("sink_master") == sink
    ), None)
    for module_id, _values in owned:
        if matching is None or module_id != matching[0]:
            _pactl("unload-module", str(module_id))

    created_module_id = None
    if matching is None:
        module_id = _pactl(
            "load-module",
            "module-echo-cancel",
            f"source_master={master}",
            f"sink_master={sink}",
            "source_name=llm_aec_source",
            "sink_name=llm_aec_sink",
            "aec_method=webrtc",
        )
        if not module_id.isdigit():
            raise RuntimeError("PulseAudio WebRTC echo cancellation failed to load")
        created_module_id = int(module_id)
        active_module_id = created_module_id
    else:
        active_module_id = matching[0]

    props = selected.get("properties", {})
    _LOGGER.info("VOICE_AUDIO_ROUTE " + json.dumps({
        "requested": str(audio_input_device),
        "source": master,
        "alsa_card": str(props.get("alsa.card", "")),
        "alsa_device": str(props.get("alsa.device", "")),
        "volume_percent": _source_volume_percentages(selected),
        "muted": bool(selected.get("mute")),
        "aec_source": "llm_aec_source",
        "sink_master": sink,
        "aec_sink": "llm_aec_sink",
        "aec_module_id": active_module_id,
        "aec_reused": matching is not None,
    }, ensure_ascii=False, sort_keys=True))
    return created_module_id


def _unload_pulse_module(_context, module_id):
    try:
        _pactl("unload-module", str(module_id))
    except RuntimeError:
        # PulseAudio may have already removed the module during shutdown.
        pass
    return []


def build_voice_launch_actions(
    context,
    *,
    environment: str,
    enable_voice,
    audio_input_device,
    audio_input_volume_percent,
    use_sim_time,
    start_delay_sec: float = 0.0,
):
    if not _enabled(enable_voice.perform(context)):
        return []
    root = kws_model_directory()
    validate_voice_runtime(root)
    module_id = ensure_pulse_aec(
        audio_input_device.perform(context),
        int(audio_input_volume_percent.perform(context)),
    )
    wake_params = load_node_parameters_yaml(
        "llm_arm_control",
        "config/llm_robot_control_params.yaml",
        "voice_wake_node",
        environment,
    )
    wake_params.update({
        "kws_tokens": str(root / "tokens.txt"),
        "kws_encoder": str(root / "encoder.onnx"),
        "kws_decoder": str(root / "decoder.onnx"),
        "kws_joiner": str(root / "joiner.onnx"),
        "kws_keywords_file": str(root / "keywords.txt"),
    })
    realtime_params = load_node_parameters_yaml(
        "llm_arm_control",
        "config/llm_robot_control_params.yaml",
        "voice_realtime_node",
        environment,
    )
    nodes = [
        Node(
            package="audio_capture",
            executable="audio_capture_node",
            name="audio_capture",
            namespace="voice_control",
            output="screen",
            remappings=[("audio", "audio_raw"), ("audio_stamped", "audio")],
            parameters=[{
                "src": "pulsesrc",
                "dst": "appsink",
                "device": "llm_aec_source",
                "format": "wave",
                "channels": 1,
                "depth": 16,
                "sample_rate": 16000,
                "sample_format": "S16LE",
                "use_sim_time": use_sim_time,
            }],
        ),
        Node(
            package="llm_arm_control",
            executable="voice_wake_node",
            name="voice_wake_node",
            output="screen",
            parameters=[wake_params, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="llm_arm_control",
            executable="voice_realtime_node",
            name="voice_realtime_node",
            output="screen",
            parameters=[realtime_params, {"use_sim_time": use_sim_time}],
        ),
    ]
    actions = (
        TimerAction(period=float(start_delay_sec), actions=nodes)
        if start_delay_sec
        else nodes
    )
    handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=[
                    EmitEvent(event=Shutdown(reason="critical voice process exited"))
                ],
            )
        )
        for node in nodes
    ]
    if module_id is not None:
        handlers.append(
            RegisterEventHandler(
                OnShutdown(
                    on_shutdown=[
                        OpaqueFunction(
                            function=_unload_pulse_module,
                            args=[module_id],
                        )
                    ]
                )
            )
        )
    return [*(actions if isinstance(actions, list) else [actions]), *handlers]
