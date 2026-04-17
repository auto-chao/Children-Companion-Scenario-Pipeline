#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"

# 模块 1 转写走 GEMINI 兼容 HTTP（见 src/ccs_audio_pipeline/asr_gemini_proxy.py）
if [ -z "${GEMINI_PROXY_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "请设置 GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY（数据集 ASR 与代理一致）。" >&2
  exit 1
fi

# 减轻 CPU 占满：可调小 --num-threads（如 2～4），并与 BLAS 对齐，例如：
#   export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUTPUT_DIR="outputs/child_dataset"
TRACE_DIR="${OUTPUT_DIR}/trace"

if ! "${PYTHON}" scripts/bootstrap_assets.py --check-only >/dev/null 2>&1; then
  echo "Offline assets are missing."
  echo "Run the bootstrap step first:"
  echo "  export HF_TOKEN=your_token"
  echo "  ${PYTHON} scripts/bootstrap_assets.py"
  exit 1
fi

# Fixed strongest path:
# Demucs htdemucs_ft -> ClearerVoice MossFormer2_SE_48K -> pyannote turns
"${PYTHON}" scripts/build_dataset.py \
  --input-dir data/audio \
  --output-dir "${OUTPUT_DIR}" \
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
