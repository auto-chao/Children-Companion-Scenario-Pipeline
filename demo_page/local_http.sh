#!/usr/bin/env bash
# 本地 HTTP 提供 demo_page/index.html 与 outputs/ 相对路径（勿用 file:// 直接打开）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$SCRIPT_DIR/.http_server.pid"
LOG_FILE="$SCRIPT_DIR/.http_server.log"
PORT="${PORT:-8765}"
BIND="${BIND:-127.0.0.1}"

url_demo() {
  echo "http://${BIND}:${PORT}/demo_page/index.html"
}

# 可执行的 Python 调用（数组，供 "${HTTP_PY[@]}" 展开）；避免 Windows 上 command -v 命中 Store 占位 python。
HTTP_PY=()

_py_ok() {
  "$@" -c "import sys" >/dev/null 2>&1
}

resolve_python() {
  HTTP_PY=()
  if [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
    if _py_ok "$PYTHON"; then
      HTTP_PY=("$PYTHON")
      return 0
    fi
    echo "错误: 环境变量 PYTHON 指向的解释器无法运行（请换为真实 python.exe 路径）。" >&2
    return 1
  fi
  # Windows：优先 py -3，跳过仅存在于 PATH 的 Store 别名 python
  if command -v py >/dev/null 2>&1 && py -3 -c "import sys" >/dev/null 2>&1; then
    HTTP_PY=(py -3)
    return 0
  fi
  for c in python3 py python; do
    command -v "$c" >/dev/null 2>&1 || continue
    if [[ "$c" == "py" ]]; then
      _py_ok py || continue
      HTTP_PY=(py)
      return 0
    fi
    if _py_ok "$c"; then
      HTTP_PY=("$c")
      return 0
    fi
  done
  echo "错误: 未找到可用的 Python（已排除无法执行的解释器）。请安装 Python 或设置 PYTHON 为 python.exe 完整路径。" >&2
  return 1
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(tr -d ' \r\n' <"$PID_FILE" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

cmd_start() {
  if is_running; then
    echo "本地 HTTP 已在运行 (PID $(tr -d ' \r\n' <"$PID_FILE"))。"
    echo "打开: $(url_demo)"
    return 0
  fi
  resolve_python || exit 1
  cd "$ROOT"
  nohup "${HTTP_PY[@]}" -m http.server "$PORT" --bind "$BIND" --directory "$ROOT" >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  sleep 0.4
  if is_running; then
    echo "已启动 PID $(tr -d ' \r\n' <"$PID_FILE")（日志: $LOG_FILE）"
    echo "打开: $(url_demo)"
  else
    echo "启动失败，请查看: $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
}

cmd_stop() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "未在运行（无 PID 文件）。"
    return 0
  fi
  local pid
  pid="$(tr -d ' \r\n' <"$PID_FILE" 2>/dev/null || true)"
  rm -f "$PID_FILE"
  if [[ -z "${pid:-}" ]]; then
    echo "已清理无效的 PID 文件。"
    return 0
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 0.3
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "已停止 (PID $pid)。"
  else
    echo "进程 $pid 已不存在，已移除 PID 文件。"
  fi
}

cmd_status() {
  if is_running; then
    echo "运行中 PID $(tr -d ' \r\n' <"$PID_FILE")"
    echo "URL: $(url_demo)"
  else
    echo "未运行。"
    [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
  fi
}

cmd_restart() {
  cmd_stop || true
  cmd_start
}

usage() {
  cat <<'EOF'
用法: demo_page/local_http.sh <命令>

  start    后台启动 Python http.server（仓库根目录为站点根）
  stop     停止后台服务
  status   是否运行及访问 URL
  restart  先 stop 再 start

环境变量:
  PORT   监听端口（默认 8765）
  BIND   绑定地址（默认 127.0.0.1）
  PYTHON 优先使用的 Python 解释器（须为可执行的 python.exe 路径）

说明:
  Windows 上会优先尝试 py -3，避免 PATH 中的 Microsoft Store「python」占位程序。

示例:
  bash demo_page/local_http.sh start
  PORT=9000 bash demo_page/local_http.sh start
  bash demo_page/local_http.sh stop
EOF
}

main() {
  case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    restart) cmd_restart ;;
    help|-h|--help)
      usage
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
