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
TEMPLATE_PATH="${OUTPUT_DIR}/child_labels.template.jsonl"
ASR_PATH="${OUTPUT_DIR}/child_labels.asr.jsonl"
DEFAULT_FILLED="${OUTPUT_DIR}/child_labels.filled.jsonl"
LABELS_PATH_EXPLICIT="${MAIN_CHILD_LABELS_PATH:-}"

# 人工校验 ASR：1=在 Qwen 后退出，等待将校对结果保存为 child_labels.filled.jsonl（或 MAIN_CHILD_LABELS_PATH）后再跑
MAIN_MANUAL_ASR_REVIEW="${MAIN_MANUAL_ASR_REVIEW:-0}"

# 无显式指定时，优先已人工文件，否则在无人工关时使用 ASR 机转文件
if [ -n "${LABELS_PATH_EXPLICIT}" ]; then
  LABELS_PATH_RESOLVED="${LABELS_PATH_EXPLICIT}"
else
  if [ -f "${DEFAULT_FILLED}" ]; then
    LABELS_PATH_RESOLVED="${DEFAULT_FILLED}"
  elif [ -f "${ASR_PATH}" ] && [ "${MAIN_MANUAL_ASR_REVIEW}" != "1" ]; then
    LABELS_PATH_RESOLVED="${ASR_PATH}"
  else
    LABELS_PATH_RESOLVED="${DEFAULT_FILLED}"
  fi
fi

if [ -n "${MAIN_BUILD_STEP:-}" ]; then
  STEP="${MAIN_BUILD_STEP}"
  if [ "${STEP}" != "1" ] && [ "${STEP}" != "2" ]; then
    echo "MAIN_BUILD_STEP must be 1 or 2, got: ${STEP}" >&2
    exit 1
  fi
else
  # 自动：有可用 labels -> step2；机转+人工开、尚无已填 -> 不推断为 step1（避免重复切段）
  if [ -n "${LABELS_PATH_EXPLICIT}" ] && [ -f "${LABELS_PATH_EXPLICIT}" ]; then
    STEP=2
  elif [ -f "${DEFAULT_FILLED}" ]; then
    STEP=2
  elif [ -f "${ASR_PATH}" ] && [ "${MAIN_MANUAL_ASR_REVIEW}" != "1" ]; then
    STEP=2
  elif [ -f "${ASR_PATH}" ] && [ "${MAIN_MANUAL_ASR_REVIEW}" = "1" ] && [ ! -f "${DEFAULT_FILLED}" ]; then
    STEP=blocked
  else
    STEP=1
  fi
fi

# 开人工但仅有模板+机转、尚无已填：等待校对
if [ "${STEP:-}" = "blocked" ]; then
  echo "" >&2
  echo "已存在 ${ASR_PATH} 且 MAIN_MANUAL_ASR_REVIEW=1。" >&2
  echo "请人工校对后保存为: ${DEFAULT_FILLED}" >&2
  echo "（或设置 MAIN_CHILD_LABELS_PATH 指向校对后的文件），再重新执行本脚本或 main.sh。" >&2
  exit 1
fi

if ! "${PYTHON}" scripts/bootstrap_assets.py --check-only >/dev/null 2>&1; then
  echo "Offline assets are missing."
  echo "Run the bootstrap step first:"
  echo "  export HF_TOKEN=your_token"
  echo "  ${PYTHON} scripts/bootstrap_assets.py"
  exit 1
fi

if [ "${STEP}" = "1" ]; then
  "${PYTHON}" scripts/build_dataset.py \
    --input-dir data/audio \
    --output-dir "${OUTPUT_DIR}" \
    --step 1 \
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
  if [ ! -f "${TEMPLATE_PATH}" ]; then
    echo "未生成 ${TEMPLATE_PATH}" >&2
    exit 1
  fi
  echo "==> Qwen ASR (child_labels.asr.jsonl)"
  "${PYTHON}" scripts/dataset/apply_qwen_asr_to_labels.py \
    --template "${TEMPLATE_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --output "${ASR_PATH}"
  if [ "${MAIN_MANUAL_ASR_REVIEW}" = "1" ]; then
    echo "" >&2
    echo "MAIN_MANUAL_ASR_REVIEW=1：请校对 ${ASR_PATH} 并另存为 ${DEFAULT_FILLED}" >&2
    echo "然后重新执行（将自动从 --step 2 使用已填文件）。" >&2
    exit 1
  fi
  echo "==> --step 2 (manifest) using ${ASR_PATH}"
  "${PYTHON}" scripts/build_dataset.py \
    --input-dir data/audio \
    --output-dir "${OUTPUT_DIR}" \
    --step 2 \
    --labels-path "${ASR_PATH}" \
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
  exit 0
fi

# --step 2
if [ ! -f "${LABELS_PATH_RESOLVED}" ]; then
  echo "找不到 labels 文件: ${LABELS_PATH_RESOLVED}" >&2
  echo "请先跑 MAIN_BUILD_STEP=1 或提供 MAIN_CHILD_LABELS_PATH。" >&2
  exit 1
fi

echo "==> --step 2 (manifest) using ${LABELS_PATH_RESOLVED}"
"${PYTHON}" scripts/build_dataset.py \
  --input-dir data/audio \
  --output-dir "${OUTPUT_DIR}" \
  --step 2 \
  --labels-path "${LABELS_PATH_RESOLVED}" \
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
