#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_PACKAGE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SOURCE_LAYOUT=0
if [[ -f "$SOURCE_PACKAGE_DIR/config/voice_models.lock.yaml" ]]; then
  PACKAGE_DIR=$SOURCE_PACKAGE_DIR
  SOURCE_LAYOUT=1
else
  PREFIX_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
  PACKAGE_DIR="$PREFIX_DIR/share/llm_arm_control"
fi
LOCK_FILE="$PACKAGE_DIR/config/voice_models.lock.yaml"
MODEL_DIR="$PACKAGE_DIR/model/kws"
CHECK_ONLY=0
CHECK_CLOUD=0
INSTALL_SYSTEM_DEPS=0

usage() {
  echo "Usage: $0 [--install-system-deps] [--check-only|--check-cloud|--refresh-keywords]"
}

REFRESH_KEYWORDS=0
while (($#)); do
  case "$1" in
    --install-system-deps) INSTALL_SYSTEM_DEPS=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --check-cloud) CHECK_ONLY=1; CHECK_CLOUD=1; shift ;;
    --refresh-keywords) CHECK_ONLY=1; REFRESH_KEYWORDS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if ((INSTALL_SYSTEM_DEPS)); then
  sudo apt-get update
  sudo apt-get install -y ros-humble-audio-common ros-humble-audio-capture \
    ros-humble-audio-common-msgs alsa-utils pulseaudio-utils
fi

if ((CHECK_ONLY == 0)); then
  python3 -m pip install --user \
    sherpa-onnx==1.13.8 sherpa-onnx-bin==1.13.8 sherpa-onnx-core==1.13.8 \
    sentencepiece==0.2.2 pypinyin==0.55.0 dashscope==1.26.5
fi

python3 - <<'PY'
import importlib.metadata
import sherpa_onnx
assert sherpa_onnx.__version__ == "1.13.8", sherpa_onnx.__version__
version = tuple(map(int, importlib.metadata.version("dashscope").split(".")[:3]))
assert version >= (1, 26, 5), version
print("sherpa_onnx", sherpa_onnx.__version__, "dashscope", importlib.metadata.version("dashscope"))
PY
command -v aplay >/dev/null
command -v pactl >/dev/null
compgen -G '/usr/lib/pulse-*/modules/module-echo-cancel.so' >/dev/null || {
  echo "PulseAudio module-echo-cancel is missing" >&2; exit 1;
}

read -r MODEL_NAME MODEL_URL MODEL_DIRECTORY MODEL_SHA256 MODEL_ARCHIVE_SIZE < <(
python3 - "$LOCK_FILE" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
value = data["models"]["kws"]
print("\t".join(str(value[name]) for name in (
    "name", "url", "directory", "sha256", "archive_size"
)))
PY
)

TEMP_DIR=""
cleanup() {
  [[ -z "$TEMP_DIR" ]] || rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

MANAGED_ASSETS=(
  encoder.onnx decoder.onnx joiner.onnx tokens.txt keywords.txt en.phone keywords_raw.txt
)
asset_ready() {
  [[ -f "$MODEL_DIR/$1" ]] && ((SOURCE_LAYOUT == 0)) || {
    [[ -f "$MODEL_DIR/$1" && ! -L "$MODEL_DIR/$1" ]]
  }
}
missing=0
for asset in "${MANAGED_ASSETS[@]}"; do
  asset_ready "$asset" || missing=1
done

if ((missing)); then
  ((CHECK_ONLY == 0)) || {
    echo "Missing or symlinked KWS asset in $MODEL_DIR" >&2
    exit 1
  }
  mkdir -p "$MODEL_DIR"
  TEMP_DIR=$(mktemp -d)
  archive="$TEMP_DIR/${MODEL_NAME}.tar.bz2"
  curl --fail --location --output "$archive" "$MODEL_URL"
  [[ $(stat -c %s "$archive") == "$MODEL_ARCHIVE_SIZE" ]] || {
    echo "Size mismatch: $archive" >&2
    exit 1
  }
  echo "$MODEL_SHA256  $archive" | sha256sum --check --status || {
    echo "SHA256 mismatch: $archive" >&2
    exit 1
  }
  tar -xjf "$archive" -C "$TEMP_DIR"
  extracted="$TEMP_DIR/$MODEL_DIRECTORY"
  install -m 0644 "$extracted/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx" "$MODEL_DIR/encoder.onnx"
  install -m 0644 "$extracted/decoder-epoch-13-avg-2-chunk-8-left-64.onnx" "$MODEL_DIR/decoder.onnx"
  install -m 0644 "$extracted/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx" "$MODEL_DIR/joiner.onnx"
  install -m 0644 "$extracted/tokens.txt" "$MODEL_DIR/tokens.txt"
  install -m 0644 "$extracted/en.phone" "$MODEL_DIR/en.phone"
  install -m 0644 "$extracted/keywords_raw.txt" "$MODEL_DIR/keywords_raw.txt"
  install -m 0644 "$extracted/keywords.txt" "$MODEL_DIR/keywords.txt"
fi

if ((CHECK_ONLY == 0 || REFRESH_KEYWORDS)); then
  if ((CHECK_ONLY == 0)); then
    mkdir -p "$MODEL_DIR"
  fi
  printf '%s\n' '小鹏同学 @小鹏同学' '小鹏小鹏 @小鹏小鹏' 'HI ROBOT @HI_ROBOT' >"$MODEL_DIR/keywords_raw.txt"
  sherpa-onnx-cli text2token --tokens "$MODEL_DIR/tokens.txt" \
    --tokens-type phone+ppinyin --lexicon "$MODEL_DIR/en.phone" \
    "$MODEL_DIR/keywords_raw.txt" "$MODEL_DIR/keywords.txt"
fi

for required in "${MANAGED_ASSETS[@]}"; do
  asset_ready "$required" || {
    echo "Missing or symlinked model asset: $required" >&2
    exit 1
  }
done

if ((CHECK_CLOUD)); then
  : "${DASHSCOPE_API_KEY:?DASHSCOPE_API_KEY is required}"
  : "${DASHSCOPE_WORKSPACE_ID:?DASHSCOPE_WORKSPACE_ID is required}"
  python3 - <<'PY'
import os
from dashscope.audio.qwen_omni import OmniRealtimeCallback, OmniRealtimeConversation
class Check(OmniRealtimeCallback):
    def on_open(self): pass
    def on_event(self, _message): pass
    def on_close(self, _code, _message): pass
workspace = os.environ["DASHSCOPE_WORKSPACE_ID"]
conversation = OmniRealtimeConversation(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    url=f"wss://{workspace}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
    model="qwen3.8-omni-flash-realtime", callback=Check(),
)
conversation.connect()
print("Qwen Realtime authentication and WebSocket connection succeeded")
conversation.close()
PY
fi

echo "Wake and Qwen Realtime runtime ready: $MODEL_DIR"
