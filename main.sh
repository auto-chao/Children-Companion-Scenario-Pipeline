#!/usr/bin/env bash
# 仓库根目录一键端到端：资产检查 → Stage1 数据集 → Stage2 助手 + 2.5 质检 → CosyVoice → Stage3 TTS + 3.5 质检
#
# 环境变量（示例）：
#   HF_TOKEN、GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY（Stage2/3.5 等）
#   MAIN_RUN_STAGE1/2/3=1|0  三阶段总开关（默认 1）
#   MAIN_MANUAL_ASR_REVIEW=1  在 Qwen 后断点，见 build_child_dataset.sh
#   ASSISTANT_WORKERS、PYTHON、MAIN_CHILD_LABELS_PATH、MAIN_BUILD_STEP 等
#   COSYVOICE_FORCE_CPU=1  强制 TTS 走 CPU

export GEMINI_PROXY_API_KEY="sk-ragFTLE01dU6dZPTgOKSfhPdW66jKVRY2PDfX7QQCLX4uo0F"

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"

RUN1=0
RUN2=1
RUN3=0
[ "${MAIN_RUN_STAGE1:-1}" = "0" ] && RUN1=0
[ "${MAIN_RUN_STAGE2:-1}" = "0" ] && RUN2=0
[ "${MAIN_RUN_STAGE3:-1}" = "0" ] && RUN3=0

echo "==> Offline assets check"
if ! "${PYTHON}" scripts/bootstrap_assets.py --check-only; then
  echo "请先下载离线资产，例如: export HF_TOKEN=你的token && ${PYTHON} scripts/bootstrap_assets.py" >&2
  exit 1
fi

if [ "${RUN1}" = "1" ]; then
  echo "==> Stage 1: child dataset (build_child_dataset.sh)"
  bash build_child_dataset.sh
  if [ ! -f "${SCRIPT_DIR}/outputs/child_dataset/manifest.jsonl" ]; then
    echo "" >&2
    echo "未生成 manifest.jsonl（可能为 MAIN_MANUAL_ASR_REVIEW=1 等待人工校对，或需先跑通 step1+ASR）。" >&2
    exit 1
  fi
else
  echo "==> Skipping Stage 1 (set MAIN_RUN_STAGE1=1 以启用；需已有 outputs/child_dataset/manifest.jsonl)"
fi

if [ "${RUN2}" = "1" ]; then
  echo "==> Stage 2: assistant responses (Gemini-compatible proxy)"
  if [ -z "${GEMINI_PROXY_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "请设置环境变量 GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY。" >&2
    exit 1
  fi
  bash run_assistant_responses.sh --workers "${ASSISTANT_WORKERS:-4}"

  echo "==> Stage 2.5: QC (GPT-5.4) — 仅通过样本进入 TTS"
  ec2=0
  "${PYTHON}" scripts/qc/verify_assistant_responses_gpt54.py || ec2=$?
  if [ "${ec2}" -ne 0 ]; then
    if [ "${ec2}" -eq 2 ]; then
      echo "Stage 2.5: 有质检样本但 0 条通过，终止流水线（不进入 TTS）。" >&2
    fi
    exit "${ec2}"
  fi
  if [ ! -s "outputs/assistant_responses_multiturn.qc_passed.jsonl" ]; then
    echo "未生成或非空的 outputs/assistant_responses_multiturn.qc_passed.jsonl，无法进入 TTS。" >&2
    exit 1
  fi
else
  echo "==> Skipping Stage 2 / 2.5 (assistant + GPT-5.4 质检)"
fi

if [ "${RUN3}" != "1" ]; then
  echo "==> Skipping Stage 3 / 3.5 (TTS + Gemini 质检)"
  exit 0
fi

VENV_UNIX="${SCRIPT_DIR}/artifacts/cosyvoice/.venv/bin/python"
VENV_WIN="${SCRIPT_DIR}/artifacts/cosyvoice/.venv/Scripts/python.exe"
echo "==> CosyVoice venv (Stage 3 依赖)"
if [[ ! -f "${VENV_UNIX}" && ! -f "${VENV_WIN}" ]]; then
  echo "    未检测到 artifacts/cosyvoice/.venv ，正在运行 deploy_cosyvoice.py（首次可能较久）…"
  "${PYTHON}" scripts/deploy_cosyvoice.py
fi

echo "==> Stage 3: TTS (run_tts.sh; GPU 默认，CPU 请设 COSYVOICE_FORCE_CPU=1)"
bash run_tts.sh \
  --input outputs/assistant_responses_multiturn.qc_passed.jsonl \
  --output outputs/assistant_responses_with_tts.jsonl

if [ -z "${GEMINI_PROXY_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "未设置 GEMINI 密钥，跳过 Stage 3.5。" >&2
else
  echo "==> Stage 3.5: QC (gemini-3.1-pro-preview, s2s)"
  "${PYTHON}" scripts/qc/verify_tts_s2s_gemini.py
fi

echo "Done."
