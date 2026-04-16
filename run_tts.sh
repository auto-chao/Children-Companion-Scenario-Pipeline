#!/usr/bin/env bash
# 仓库根目录一键批量 CosyVoice TTS（每轮仅使用 JSON 中的 plain_text，zero-shot 合成）。
# 依赖：已运行 python scripts/deploy_cosyvoice.py，使 artifacts/cosyvoice/.venv 存在并装好依赖。
#
# 用法（Git Bash / WSL / Linux / macOS）:
#   ./run_tts.sh
#   bash run_tts.sh
#   sh run_tts.sh   # 会自动改用 bash 重新执行（dash 不支持 pipefail / [[）
#   ./run_tts.sh --mock --limit 2
#   COSYVOICE_FORCE_CPU=1 ./run_tts.sh
#
# 环境变量:
#   COSYVOICE_FORCE_CPU=1|true|yes  追加 --cpu（与 batch 脚本内 COSYVOICE_FORCE_CPU 一致）
#
# 说明：本脚本固定使用 artifacts/cosyvoice/.venv 中的 Python（CosyVoice 依赖），
# 与当前是否激活 conda「ccs」无关；请勿依赖 conda 里的 site-packages 跑 CosyVoice。

# dash/sh 不支持 pipefail 与 [[；未在 bash 下则换用 bash 重新执行本脚本
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

VENV_UNIX="${REPO_ROOT}/artifacts/cosyvoice/.venv/bin/python"
VENV_WIN="${REPO_ROOT}/artifacts/cosyvoice/.venv/Scripts/python.exe"

PY=""
if [[ -f "${VENV_UNIX}" ]]; then
  PY="${VENV_UNIX}"
elif [[ -f "${VENV_WIN}" ]]; then
  PY="${VENV_WIN}"
fi

if [[ -z "${PY}" ]]; then
  echo "未找到 CosyVoice 虚拟环境: artifacts/cosyvoice/.venv" >&2
  echo "请先执行: python scripts/deploy_cosyvoice.py" >&2
  exit 1
fi

cpu_args=()
case "${COSYVOICE_FORCE_CPU:-}" in
  1|true|True|yes|YES) cpu_args=(--cpu) ;;
esac

echo "Using CosyVoice Python: ${PY}"
"${PY}" -c "import torch; print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}'); print('cuda_device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))" || true

exec "${PY}" scripts/tts/batch_cosyvoice_tts.py \
  "${cpu_args[@]}" "$@"
