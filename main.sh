#!/usr/bin/env bash
# 仓库根目录一键端到端流水线（Git Bash / WSL / Linux / macOS）
# 1) API 儿童数据集  2) 助手回复  3) CosyVoice（按需部署） 4) TTS  5) Demo 页
#
# 使用前请激活与本项目一致的 conda 环境（如 ccs），并设置：
#   GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY（助手步骤）
#   HF_TOKEN（仅在首次部署 CosyVoice 需要拉权重时）
# TTS 默认用 GPU；仅 CPU 推理： COSYVOICE_FORCE_CPU=1 ./main.sh
#
# 可选维护者环境变量（一般不用于日常）：
#   PYTHON=python3   MAIN_SKIP_ASSISTANT=1   MAIN_SKIP_TTS=1   ASSISTANT_WORKERS=4

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"

echo "==> [1/5] Child dataset (API + ffmpeg only)"
bash build_child_dataset_api.sh

if [ -n "${MAIN_SKIP_ASSISTANT:-}" ]; then
  echo "==> Skipping assistant / TTS / demo (MAIN_SKIP_ASSISTANT is set)"
  exit 0
fi

echo "==> [2/5] Assistant responses (Gemini-compatible proxy)"
if [ -z "${GEMINI_PROXY_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "请设置环境变量 GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY（第三方代理密钥）。" >&2
  exit 1
fi
bash run_assistant_responses.sh --workers "${ASSISTANT_WORKERS:-4}"

if [ -n "${MAIN_SKIP_TTS:-}" ]; then
  echo "==> Skipping TTS and demo (MAIN_SKIP_TTS is set)"
  exit 0
fi

VENV_UNIX="${SCRIPT_DIR}/artifacts/cosyvoice/.venv/bin/python"
VENV_WIN="${SCRIPT_DIR}/artifacts/cosyvoice/.venv/Scripts/python.exe"
echo "==> [3/5] CosyVoice"
if [[ ! -f "${VENV_UNIX}" && ! -f "${VENV_WIN}" ]]; then
  echo "    未检测到 artifacts/cosyvoice/.venv ，正在运行 deploy_cosyvoice.py（首次可能较久）…"
  "${PYTHON}" scripts/deploy_cosyvoice.py
fi

echo "==> [4/5] TTS (run_tts.sh; GPU 默认，CPU 请设 COSYVOICE_FORCE_CPU=1)"
bash run_tts.sh \
  --input outputs/assistant_responses_multiturn.jsonl \
  --output outputs/assistant_responses_with_tts.jsonl

echo "==> [5/5] Demo page"
"${PYTHON}" scripts/demo/generate_demo_page.py \
  --input outputs/assistant_responses_with_tts.jsonl

echo "Done. Open demo_page/index.html in a browser (audio paths are relative to that file)."
