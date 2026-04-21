#!/usr/bin/env bash
# 仓库根目录一键端到端流水线（Git Bash / WSL / Linux / macOS）
# 1) 离线资产检查  2) 儿童数据集  3) 助手回复  4) CosyVoice（按需部署） 5) TTS  6) Demo 页
#
# 使用前请激活与本项目一致的 conda 环境（如 ccs），并设置：
#   HF_TOKEN（若需首次 bootstrap 或首次部署 CosyVoice 拉权重）
#   GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY（数据集 --step 2 家长间隙 ASR + 助手步骤）
# TTS 默认用 GPU；仅 CPU 推理： COSYVOICE_FORCE_CPU=1 ./main.sh
#
# 可选维护者环境变量（一般不用于日常）：
#   PYTHON=python3   MAIN_SKIP_DATASET=1   MAIN_SKIP_ASSISTANT=1   MAIN_SKIP_TTS=1   ASSISTANT_WORKERS=4
#   MAIN_CHILD_LABELS_PATH=path/to/child_labels.filled.jsonl  已填标签（默认 outputs/child_dataset/child_labels.filled.jsonl）
#   MAIN_BUILD_STEP=1|2  强制数据集子步骤（覆盖自动推断）

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"

echo "==> [1/6] Offline assets check"
if ! "${PYTHON}" scripts/bootstrap_assets.py --check-only; then
  echo "请先下载离线资产，例如: export HF_TOKEN=你的token && ${PYTHON} scripts/bootstrap_assets.py" >&2
  exit 1
fi

if [ -z "${MAIN_SKIP_DATASET:-}" ]; then
  echo "==> [2/6] Child dataset (build_child_dataset.sh)"
  bash build_child_dataset.sh
  if [ ! -f "${SCRIPT_DIR}/outputs/child_dataset/manifest.jsonl" ]; then
    echo "" >&2
    echo "未生成 outputs/child_dataset/manifest.jsonl（本次为数据集 --step 1：已写出 child_labels.template.jsonl 与 audios/）。" >&2
    echo "请人工填写模板中的 content，另存为 outputs/child_dataset/child_labels.filled.jsonl" >&2
    echo "（或通过 MAIN_CHILD_LABELS_PATH 指定已填文件），并设置 GEMINI 密钥后重新执行 bash main.sh。" >&2
    exit 1
  fi
else
  echo "==> Skipping child dataset (MAIN_SKIP_DATASET is set; 请确认已有 outputs/child_dataset/manifest.jsonl)"
fi

if [ -n "${MAIN_SKIP_ASSISTANT:-}" ]; then
  echo "==> Skipping assistant / TTS / demo (MAIN_SKIP_ASSISTANT is set)"
  exit 0
fi

echo "==> [3/6] Assistant responses (Gemini-compatible proxy)"
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
echo "==> [4/6] CosyVoice"
if [[ ! -f "${VENV_UNIX}" && ! -f "${VENV_WIN}" ]]; then
  echo "    未检测到 artifacts/cosyvoice/.venv ，正在运行 deploy_cosyvoice.py（首次可能较久）…"
  "${PYTHON}" scripts/deploy_cosyvoice.py
fi

echo "==> [5/6] TTS (run_tts.sh; GPU 默认，CPU 请设 COSYVOICE_FORCE_CPU=1)"
bash run_tts.sh \
  --input outputs/assistant_responses_multiturn.jsonl \
  --output outputs/assistant_responses_with_tts.jsonl

echo "==> [6/6] Demo page"
"${PYTHON}" scripts/demo/generate_demo_page.py \
  --input outputs/assistant_responses_with_tts.jsonl

echo "Done. Demo: bash demo_page/local_http.sh start — then open the printed http:// URL (needs HTTP for samples_embed.json + outputs/ audio; do not rely on file://)."
