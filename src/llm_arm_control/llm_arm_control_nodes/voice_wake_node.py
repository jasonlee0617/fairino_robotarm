#!/usr/bin/env python3
"""Local Sherpa-ONNX wake detector; command audio never leaves this node."""

from __future__ import annotations

import json
import os
import re
import signal
import tempfile
import threading
import time
from collections import Counter

import numpy as np
import rclpy
from audio_common_msgs.msg import AudioDataStamped, AudioInfo
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from llm_arm_control_nodes.voice_logic import classify_wake, compact_speech


LISTEN_MODES = frozenset({"WAKE_ONLY", "MUTED"})


class VoiceWakeNode(Node):
    def __init__(self):
        super().__init__("voice_wake_node")
        self._declare_parameters()
        self._mode = "WAKE_ONLY"
        self._audio_valid = False
        self._audio_info_seen = False
        self._audio_format = {}
        self._lock = threading.RLock()
        self._runtime_keywords_path = ""
        self._audio_frames = 0
        self._audio_total_frames = 0
        self._audio_samples = 0
        self._audio_energy = 0.0
        self._audio_peak = 0.0
        self._audio_clipped = 0
        self._audio_silent = 0
        self._audio_metrics_started_at = time.monotonic()
        self._last_audio_at = None
        self._last_wake_at = 0.0
        self._kws_hits = Counter()
        self._last_audio_issue = ""
        self._health_reported = False

        self.wake_pub = self.create_publisher(String, "/voice_control/wake_event", 10)
        self.create_subscription(String, "/voice_control/listen_mode", self._on_mode, 10)
        info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            AudioInfo, "/voice_control/audio_info", self._on_info, info_qos
        )
        self.create_subscription(AudioDataStamped, "/voice_control/audio", self._on_audio, 20)

        self._sherpa = self._load_sherpa()
        self._spotter = self._create_spotter()
        self._kws_stream = self._spotter.create_stream()
        self.create_timer(5.0, self._report_audio_diagnostics)
        self.get_logger().info(
            "Local wake detector ready: 小鹏同学 / 小鹏小鹏 / Hi Robot."
        )

    def _declare_parameters(self):
        defaults = {
            "sample_rate": 16000,
            "num_threads": 2,
            "provider": "cpu",
            "kws_tokens": "",
            "kws_encoder": "",
            "kws_decoder": "",
            "kws_joiner": "",
            "kws_keywords_file": "",
            "kws_keywords_score": 1.0,
            "kws_keywords_threshold": 0.25,
            "kws_score_wake_zh": 3.0,
            "kws_threshold_wake_zh": 0.05,
            "kws_score_wake_en": 1.0,
            "kws_threshold_wake_en": 0.15,
            "kws_num_trailing_blanks": 1,
            "wake_dedup_sec": 2.0,
            "voice_diagnostics_enabled": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _param(self, name):
        return self.get_parameter(name).value

    def _load_sherpa(self):
        try:
            import sherpa_onnx
        except Exception as exc:
            raise RuntimeError(
                "sherpa_onnx is unavailable; run setup_voice_runtime.sh first"
            ) from exc
        return sherpa_onnx

    def _create_spotter(self):
        s = self._sherpa
        keywords_file = self._thresholded_keywords_file()
        return s.KeywordSpotter(
            tokens=str(self._param("kws_tokens")),
            encoder=str(self._param("kws_encoder")),
            decoder=str(self._param("kws_decoder")),
            joiner=str(self._param("kws_joiner")),
            keywords_file=keywords_file,
            num_threads=int(self._param("num_threads")),
            sample_rate=int(self._param("sample_rate")),
            keywords_score=float(self._param("kws_keywords_score")),
            keywords_threshold=float(self._param("kws_keywords_threshold")),
            num_trailing_blanks=int(self._param("kws_num_trailing_blanks")),
            provider=str(self._param("provider")),
        )

    def _thresholded_keywords_file(self):
        source_path = str(self._param("kws_keywords_file"))
        profiles = {
            "小鹏同学": (
                float(self._param("kws_score_wake_zh")),
                float(self._param("kws_threshold_wake_zh")),
            ),
            "小鹏小鹏": (
                float(self._param("kws_score_wake_zh")),
                float(self._param("kws_threshold_wake_zh")),
            ),
            "hirobot": (
                float(self._param("kws_score_wake_en")),
                float(self._param("kws_threshold_wake_en")),
            ),
        }
        with open(source_path, "r", encoding="utf-8") as source:
            lines = source.readlines()
        rendered = []
        for line in lines:
            match = re.search(r"@(\S+)\s*$", line)
            alias = compact_speech(match.group(1)) if match else ""
            if alias not in profiles:
                rendered.append(line.rstrip())
                continue
            score, threshold = profiles[alias]
            body = re.sub(
                r"\s+[:#](?:\d+(?:\.\d*)?|\.\d+)(?=\s|$)",
                "",
                line[:match.start()],
            ).rstrip()
            boost = "" if score is None else f" :{score:.2f}"
            rendered.append(f"{body}{boost} #{threshold:.2f} @{match.group(1)}")
        runtime = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="llm-kws-", suffix=".txt", delete=False
        )
        with runtime:
            runtime.write("\n".join(rendered) + "\n")
        self._runtime_keywords_path = runtime.name
        return runtime.name

    def _on_mode(self, msg):
        mode = str(msg.data).strip().upper()
        if mode in LISTEN_MODES:
            with self._lock:
                previous = self._mode
                self._mode = mode
                if mode == "WAKE_ONLY" and previous != "WAKE_ONLY":
                    self._kws_stream = self._spotter.create_stream()

    def _on_info(self, msg):
        audio_format = {
            "channels": int(msg.channels),
            "sample_rate": int(msg.sample_rate),
            "sample_format": str(msg.sample_format),
        }
        valid = (
            int(msg.channels) == 1
            and int(msg.sample_rate) == int(self._param("sample_rate"))
            and str(msg.sample_format).upper() == "S16LE"
        )
        with self._lock:
            changed = not self._audio_info_seen or audio_format != self._audio_format
            self._audio_info_seen = True
            self._audio_format = audio_format
            self._audio_valid = valid
        if valid:
            if changed:
                self.get_logger().info("Microphone format ready: 16 kHz mono S16LE.")
        else:
            self.get_logger().error(
                "Unsupported microphone format; expected 16 kHz mono S16LE."
            )

    @staticmethod
    def _result_text(result) -> str:
        if result is None:
            return ""
        return str(getattr(result, "text", result)).strip()

    def _publish_wake(self, token: str) -> bool:
        canonical, language = classify_wake(token)
        if not language or self._mode != "WAKE_ONLY":
            return False
        now = time.monotonic()
        if now - self._last_wake_at < float(self._param("wake_dedup_sec")):
            return False
        self._last_wake_at = now
        normalized = compact_speech(token)
        self._kws_hits[normalized] += 1
        self.get_logger().info("WAKE_TRACE " + json.dumps({
            "source": "kws",
            "wake": normalized,
            "canonical": canonical,
            "language": language,
        }, ensure_ascii=False))
        self.wake_pub.publish(String(data=json.dumps({
            "wake": canonical, "language": language, "source": "kws",
            "monotonic_ns": time.monotonic_ns(),
        }, ensure_ascii=False)))
        self._kws_stream = self._spotter.create_stream()
        return True

    def _decode_kws(self, samples) -> bool:
        self._kws_stream.accept_waveform(int(self._param("sample_rate")), samples)
        while self._spotter.is_ready(self._kws_stream):
            self._spotter.decode_stream(self._kws_stream)
        text = self._result_text(self._spotter.get_result(self._kws_stream))
        return self._publish_wake(text)

    def _update_audio_diagnostics(self, samples):
        absolute = np.abs(samples)
        self._audio_frames += 1
        self._audio_samples += int(samples.size)
        self._audio_energy += float(np.dot(samples.astype(np.float64), samples))
        self._audio_peak = max(self._audio_peak, float(absolute.max(initial=0.0)))
        self._audio_clipped += int(np.count_nonzero(absolute >= 0.999))
        self._audio_silent += int(np.count_nonzero(absolute < 0.005))

    def _report_audio_diagnostics(self):
        with self._lock:
            now = time.monotonic()
            duration = now - self._audio_metrics_started_at
            rms = (
                (self._audio_energy / self._audio_samples) ** 0.5
                if self._audio_samples else 0.0
            )
            last_audio_age_ms = (
                None if self._last_audio_at is None
                else round((now - self._last_audio_at) * 1000.0, 1)
            )
            payload = {
                "window_sec": round(duration, 2),
                "window_frames": self._audio_frames,
                "total_frames": self._audio_total_frames,
                "samples": self._audio_samples,
                "last_audio_age_ms": last_audio_age_ms,
                "audio_publishers": self.count_publishers("/voice_control/audio"),
                "audio_info_publishers": self.count_publishers(
                    "/voice_control/audio_info"
                ),
                "audio_info_seen": self._audio_info_seen,
                "format_valid": self._audio_valid,
                "audio_format": dict(self._audio_format),
                "listen_mode": self._mode,
                "rms": round(rms, 6),
                "peak": round(self._audio_peak, 6),
                "clipped_ratio": round(
                    self._audio_clipped / self._audio_samples, 6
                ) if self._audio_samples else 0.0,
                "silence_ratio": round(
                    self._audio_silent / self._audio_samples, 6
                ) if self._audio_samples else 1.0,
                "kws_hits": dict(self._kws_hits),
            }
            self._audio_frames = 0
            self._audio_samples = 0
            self._audio_clipped = 0
            self._audio_silent = 0
            self._audio_energy = 0.0
            self._audio_peak = 0.0
            self._audio_metrics_started_at = now
        issue = (
            "audio_publisher_missing" if not payload["audio_publishers"]
            else "audio_info_missing" if (
                not payload["audio_info_publishers"]
                or not payload["audio_info_seen"]
            )
            else "audio_format_invalid" if not payload["format_valid"]
            else "audio_frames_missing" if not payload["window_frames"]
            else "digital_silence" if (
                payload["listen_mode"] == "WAKE_ONLY"
                and payload["samples"]
                and payload["peak"] <= round(1.0 / 32768.0, 6)
            )
            else "input_level_low" if (
                payload["listen_mode"] == "WAKE_ONLY"
                and payload["samples"]
                and payload["rms"] < 0.003
                and payload["peak"] < 0.03
                and payload["silence_ratio"] > 0.95
            )
            else ""
        )
        if issue and issue != self._last_audio_issue:
            self.get_logger().warning(
                "VOICE_AUDIO_UNHEALTHY "
                + json.dumps({"reason": issue, **payload}, ensure_ascii=False)
            )
        self._last_audio_issue = issue
        if not self._health_reported or bool(
            self._param("voice_diagnostics_enabled")
        ):
            self.get_logger().info(
                "VOICE_AUDIO_HEALTH " + json.dumps(payload, ensure_ascii=False)
            )
        self._health_reported = True

    def _on_audio(self, msg):
        with self._lock:
            self._audio_total_frames += 1
            self._last_audio_at = time.monotonic()
        if not self._audio_valid:
            return
        raw = bytes(msg.audio.data)
        if len(raw) < 2 or len(raw) % 2:
            return
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        with self._lock:
            self._update_audio_diagnostics(samples)
            if self._mode == "WAKE_ONLY":
                self._decode_kws(samples)

    def destroy_node(self):
        path, self._runtime_keywords_path = self._runtime_keywords_path, ""
        if path:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceWakeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
