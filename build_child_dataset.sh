#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"

OUTPUT_DIR="outputs/child_dataset"
TRACE_DIR="${OUTPUT_DIR}/trace"
DEFAULT_FILLED="${OUTPUT_DIR}/child_labels.filled.jsonl"
LABELS_PATH="${MAIN_CHILD_LABELS_PATH:-$DEFAULT_FILLED}"

if [ -n "${MAIN_BUILD_STEP:-}" ]; then
  STEP="${MAIN_BUILD_STEP}"
  if [ "${STEP}" != "1" ] && [ "${STEP}" != "2" ]; then
    echo "MAIN_BUILD_STEP must be 1 or 2, got: ${STEP}" >&2
    exit 1
  fi
else
  if [ -f "${LABELS_PATH}" ]; then
    STEP=2
  else
    STEP=1
  fi
fi

if ! "${PYTHON}" scripts/bootstrap_assets.py --check-only >/dev/null 2>&1; then
  echo "Offline assets are missing."
  echo "Run the bootstrap step first:"
  echo "  export HF_TOKEN=your_token"
  echo "  ${PYTHON} scripts/bootstrap_assets.py"
  exit 1
fi

# Fixed strongest path:
# Demucs htdemucs_ft -> ClearerVoice MossFormer2_SE_48K -> pyannote turns
# --step 1：仅 audios/ + child_labels.template.jsonl（儿童侧不做 API ASR）
# --step 2：读已填 JSONL，BGE + 聚链 + manifest（家长间隙 API ASR）
# 自动推断：存在 MAIN_CHILD_LABELS_PATH 或默认 child_labels.filled.jsonl 则 step 2，否则 step 1。
# 显式覆盖：MAIN_BUILD_STEP=1 或 MAIN_BUILD_STEP=2

STEP_ARGS=(--step "${STEP}")
if [ "${STEP}" = "2" ]; then
  if [ -z "${GEMINI_PROXY_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "请设置 GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY（--step 2 需家长间隙 API ASR）。" >&2
    exit 1
  fi
  STEP_ARGS+=(--labels-path "${LABELS_PATH}")
fi

"${PYTHON}" scripts/build_dataset.py \
  --input-dir data/audio \
  --output-dir "${OUTPUT_DIR}" \
  "${STEP_ARGS[@]}" \
  --seed 42 \
  --num-threads 8 \
  --max-turn-sec 30.0 \
  --min-turn-sec 1.25 \
  --turn-merge-gap-sec 0.5 \
  --turn-glitch-max-sec 0.25 \
  --turn-glitch-gap-sec 0.2 \
  --child-threshold 0.6 \
  --max-gap-seconds 30 \
  --multi-link-threshold 0.7 \
  --max-turns 6 \
  --trace-dir "${TRACE_DIR}"
