#!/usr/bin/env bash
# 从仓库根目录运行：批量生成儿童陪伴助手回复（第三方 Gemini 代理）
# 默认 --mode multi（读 manifest.jsonl，写 assistant_responses_multiturn.jsonl）
# 单轮: --mode single --input 你的单轮.jsonl（须显式指定输入，不再使用 manifest_single_turn.jsonl）
# 须先设置 GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY（勿将密钥写入本文件或提交到 Git）
# 用法: export GEMINI_PROXY_API_KEY=... && sh run_assistant_responses.sh [-- 额外参数传给 Python]
# 例如并发: sh run_assistant_responses.sh --workers 4
#
# 可选环境变量（与 main.sh 一致；CLI 参数若重复传入一般由 argparse 后者覆盖）:
#   ASSISTANT_WORKERS（默认 4）、ASSISTANT_RESUME=1|true、ASSISTANT_MAX_PASSES=N
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON="${PYTHON:-python}"

if [ -z "${GEMINI_PROXY_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "请先设置环境变量 GEMINI_PROXY_API_KEY（或 GEMINI_API_KEY）为第三方代理密钥。" >&2
  exit 1
fi

_pre=(--workers "${ASSISTANT_WORKERS:-4}")
case "${ASSISTANT_RESUME:-}" in 1|true|True|yes|YES) _pre+=(--resume) ;; esac
[ -n "${ASSISTANT_MAX_PASSES:-}" ] && _pre+=(--max-passes "${ASSISTANT_MAX_PASSES}")

exec "${PYTHON}" scripts/assistant/generate_assistant_responses.py --with-google-search "${_pre[@]}" "$@"
