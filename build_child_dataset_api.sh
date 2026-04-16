#!/usr/bin/env bash
# 模块一 · 仅多模态 API + ffmpeg 切分（不加载 pyannote / 本地 ASR / GPU 模型权重）。
# 无需 HF bootstrap；需要 GEMINI_PROXY_API_KEY、ffmpeg、ffprobe。

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"

if [ -z "${GEMINI_PROXY_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "请设置 GEMINI_PROXY_API_KEY（第三方 Gemini 兼容代理密钥）" >&2
  exit 1
fi

exec "${PYTHON}" scripts/build_dataset_api.py \
  --input-dir data/audio \
  --output-dir outputs/child_dataset \
  "$@"
