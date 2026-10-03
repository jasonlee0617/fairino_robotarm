"""Pure wake-speech normalization rules."""

from __future__ import annotations

import re


def compact_speech(text) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text).strip().lower())


_WAKE_ALIASES = {
    "小鹏同学": ("小鹏同学", "zh"),
    "小鹏小鹏": ("小鹏小鹏", "zh"),
    "hirobot": ("Hi Robot", "en"),
}


def classify_wake(text):
    return _WAKE_ALIASES.get(compact_speech(text), ("", ""))
