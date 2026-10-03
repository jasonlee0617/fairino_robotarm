"""Provider-neutral realtime voice transport with the Qwen implementation."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
import re
from typing import Callable
import uuid

from llm_arm_control_nodes.agent_protocol import agent_tool_definitions


@dataclass(frozen=True)
class RealtimeEvent:
    """A provider event consumed by the ROS-independent session policy."""

    kind: str
    generation: int
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RealtimeProviderConfig:
    model: str
    voice: str
    vad_silence_ms: int
    temperature: float
    instructions: str
    tool_names: tuple[str, ...]


class ProviderConfigurationError(RuntimeError):
    """A non-retryable provider configuration problem."""


class QwenRealtimeProvider:
    """Translate DashScope Qwen Realtime calls and events at one boundary."""

    supports_images = True

    def __init__(
        self,
        config: RealtimeProviderConfig,
        generation: int,
        emit: Callable[[RealtimeEvent], None],
    ):
        self._config = config
        self._generation = int(generation)
        self._emit = emit
        self._conversation = None

    def _event(self, kind: str, **data) -> None:
        self._emit(RealtimeEvent(kind, self._generation, data))

    def connect(self) -> None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        workspace = os.environ.get("DASHSCOPE_WORKSPACE_ID", "").strip()
        if not api_key or not workspace:
            raise ProviderConfigurationError(
                "DASHSCOPE_API_KEY and DASHSCOPE_WORKSPACE_ID are required"
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]+", workspace):
            raise ProviderConfigurationError("DASHSCOPE_WORKSPACE_ID is invalid")

        try:
            import dashscope
            from dashscope.audio.qwen_omni import (
                AudioFormat,
                MultiModality,
                OmniRealtimeCallback,
                OmniRealtimeConversation,
            )
        except ImportError as exc:
            raise ProviderConfigurationError(
                "dashscope>=1.26.5 is required; run setup_voice_runtime.sh"
            ) from exc

        dashscope.api_key = api_key
        owner = self

        class Callback(OmniRealtimeCallback):
            def on_open(self):
                owner._event("connection_opened")

            def on_event(self, message):
                raw = dict(message) if isinstance(message, dict) else json.loads(message)
                owner._translate(raw)

            def on_close(self, code, message):
                owner._event(
                    "connection_closed", code=code, message=message, retryable=True
                )

        conversation = OmniRealtimeConversation(
            api_key=api_key,
            url=(
                f"wss://{workspace}.cn-beijing.maas.aliyuncs.com/"
                "api-ws/v1/realtime"
            ),
            model=self._config.model,
            callback=Callback(),
        )
        self._conversation = conversation
        conversation.connect()
        conversation.update_session(
            output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
            voice=self._config.voice,
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            enable_input_audio_transcription=True,
            enable_turn_detection=True,
            turn_detection_type="semantic_vad",
            turn_detection_silence_duration_ms=self._config.vad_silence_ms,
            turn_detection_param={
                "create_response": False,
                "interrupt_response": False,
            },
            temperature=self._config.temperature,
            max_tokens=2048,
            instructions=self._config.instructions,
            tools=agent_tool_definitions(self._config.tool_names),
            enable_search=False,
        )

    def _translate(self, raw: dict) -> None:
        kind = str(raw.get("type", ""))
        if kind == "session.updated":
            self._event("session_ready")
        elif kind == "input_audio_buffer.speech_started":
            self._event(
                "speech_started",
                event_id=str(raw.get("event_id", "")),
                item_id=str(raw.get("item_id", "")),
                audio_start_ms=raw.get("audio_start_ms"),
            )
        elif kind == "input_audio_buffer.speech_stopped":
            self._event(
                "speech_stopped",
                event_id=str(raw.get("event_id", "")),
                item_id=str(raw.get("item_id", "")),
                audio_end_ms=raw.get("audio_end_ms"),
            )
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._event(
                "transcription_completed",
                event_id=str(raw.get("event_id", "")),
                item_id=str(raw.get("item_id", "")),
                transcript=str(raw.get("transcript", "")),
            )
        elif kind == "conversation.item.input_audio_transcription.failed":
            error = raw.get("error") or {}
            self._event(
                "transcription_failed",
                event_id=str(raw.get("event_id", "")),
                item_id=str(raw.get("item_id", "")),
                message=str(error.get("message", "unknown error")),
            )
        elif kind == "response.audio.delta":
            self._event(
                "audio_delta",
                response_id=str(raw.get("response_id", "")),
                audio_b64=str(raw.get("delta", "")),
            )
        elif kind == "response.audio_transcript.done":
            self._event("response_transcript_done", transcript=str(raw.get("transcript", "")))
        elif kind == "response.function_call_arguments.done":
            self._event(
                "tool_call",
                response_id=str(raw.get("response_id", "")),
                call_id=str(raw.get("call_id", "")),
                name=str(raw.get("name", "")),
                arguments=raw.get("arguments", "{}"),
            )
        elif kind == "response.created":
            response = raw.get("response") or {}
            self._event("response_started", response_id=str(response.get("id", "")))
        elif kind == "response.done":
            response = raw.get("response") or {}
            details = response.get("status_details") or {}
            self._event(
                "response_done",
                response_id=str(response.get("id", "")),
                status=str(response.get("status", "")),
                detail_type=str(details.get("type", "")),
                reason=str(details.get("reason", "")),
            )
        elif kind == "error":
            error = raw.get("error") or {}
            if not isinstance(error, dict):
                error = {"message": str(error)}
            self._event(
                "provider_error",
                code=str(error.get("code", raw.get("code", ""))),
                error_type=str(error.get("type", "")),
                message=str(error.get("message", raw.get("message", "unknown error"))),
                retryable=True,
            )

    def _require_conversation(self):
        if self._conversation is None:
            raise RuntimeError("realtime provider is not connected")
        return self._conversation

    def send_audio(self, pcm_s16le: bytes) -> None:
        self._require_conversation().append_audio(
            base64.b64encode(pcm_s16le).decode("ascii")
        )

    def create_response(self, instructions: str | None = None) -> None:
        self._require_conversation().create_response(instructions=instructions)

    def cancel_response(self) -> None:
        self._require_conversation().cancel_response()

    def send_tool_result(self, call_id: str, value: dict) -> None:
        self._require_conversation().create_item({
            "id": "item_" + uuid.uuid4().hex,
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(value, ensure_ascii=False),
        })

    def send_image(self, jpeg: bytes) -> None:
        self._require_conversation().append_video(base64.b64encode(jpeg).decode("ascii"))

    def close(self) -> None:
        conversation, self._conversation = self._conversation, None
        if conversation is not None:
            conversation.close()


def create_realtime_provider(
    name: str,
    config: RealtimeProviderConfig,
    generation: int,
    emit: Callable[[RealtimeEvent], None],
) -> QwenRealtimeProvider:
    validate_realtime_provider(name)
    return QwenRealtimeProvider(config, generation, emit)


def validate_realtime_provider(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized != "qwen":
        raise ProviderConfigurationError(
            f"unsupported realtime_provider: {name!r}; supported: qwen"
        )
    return normalized
