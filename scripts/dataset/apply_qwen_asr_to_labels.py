#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读 --step 1 产出的 child_labels.template.jsonl，对儿童片段调 Qwen ASR，写出 child_labels.asr.jsonl。

通过将仓库 api_call/ 加入 sys.path 以 import api_call_qwen（不修改 api_call 内文件；日志在 api_call/api_logs）。

失败容忍：单条 ASR 在 per-request 重试耗尽后仍失败则记索引、写空 content，不中断全量。
顺序：按模板行顺序（0..N-1）预分配结果槽位，写出时严格按序。

状态文件（默认与 --output 同目录、同名加后缀 .asr_state.json）记录 failed_indices，供 --resume 与多轮补全使用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[2]


_ROOT = _repo_root()
_DEFAULT_TEMPLATE = _ROOT / "outputs" / "child_dataset" / "child_labels.template.jsonl"
_DEFAULT_OUT = _ROOT / "outputs" / "child_dataset" / "child_labels.asr.jsonl"


def _default_state_path(output: Path) -> Path:
    return output.with_name(output.stem + ".asr_state.json")


def _load_template_rows(template: Path, limit: int) -> tuple[list[dict[str, Any]], list[int]]:
    """返回 (rows, line_numbers)：仅非空行；line_numbers 为源文件行号（用于报错）。"""
    rows: list[dict[str, Any]] = []
    line_nos: list[int] = []
    with template.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {line_no} 行 JSON 无效: {e}", file=sys.stderr)
                raise SystemExit(1) from e
            rows.append(row)
            line_nos.append(line_no)
            if limit and len(rows) >= limit:
                break
    return rows, line_nos


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"{path} 第 {line_no} 行 JSON 无效: {e}", file=sys.stderr)
                raise SystemExit(1) from e
    return out


def _fsync_textio(f: TextIO) -> None:
    try:
        f.flush()
        os.fsync(f.fileno())
    except OSError:
        try:
            f.flush()
        except OSError:
            pass


def _write_state(path: Path, *, failed_indices: list[int], last_pass: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "failed_indices": sorted(failed_indices),
        "last_pass": last_pass,
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        _fsync_textio(f)


def _state_failed_snapshot(failed: set[int], batch: list[int], batch_pos: int) -> list[int]:
    """含本 pass 尚未处理的 batch 后缀，便于中断后 resume 继续跑。"""
    not_done = set(batch[batch_pos + 1 :])
    return sorted(failed | not_done)


def _read_state(path: Path) -> tuple[list[int], int]:
    if not path.is_file():
        return [], 0
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"状态文件损坏: {path}: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    raw = obj.get("failed_indices") or []
    if not isinstance(raw, list):
        print(f"状态文件格式错误: {path} failed_indices", file=sys.stderr)
        raise SystemExit(1)
    indices: list[int] = []
    for x in raw:
        if isinstance(x, int) and x >= 0:
            indices.append(x)
        elif isinstance(x, str) and x.isdigit():
            indices.append(int(x))
    last_pass = obj.get("last_pass", 0)
    if not isinstance(last_pass, int):
        last_pass = 0
    return sorted(set(indices)), last_pass


def _resolve_clip(row: dict[str, Any], out_dir: Path) -> tuple[Path | None, str | None]:
    fp = str(row.get("file_path") or "")
    if not fp:
        return None, "缺少 file_path"
    clip = Path(fp)
    if not clip.is_absolute():
        clip = (out_dir / clip).resolve()
    if not clip.is_file():
        return None, f"片段不存在: {clip}"
    return clip, None


def _transcribe_qwen_with_retries(
    clip: Path,
    *,
    max_retries: int,
    base_sleep: float,
    user: str,
    transcribe_qwen: Any,
) -> tuple[str | None, str | None]:
    """成功返回 (text, None)；彻底失败返回 (None, err_msg)。"""
    sleep_s = base_sleep
    last_msg = "unknown"
    for attempt in range(max_retries):
        try:
            text = transcribe_qwen(clip, user=user)
            return (text or "").strip(), None
        except Exception as e:
            last_msg = f"{type(e).__name__}: {e}"
            msg = str(e).lower()
            retryable = (
                "timeout" in msg
                or "timed out" in msg
                or "connection" in msg
                or "connect" in msg
                or "network" in msg
                or "429" in msg
                or "503" in msg
                or "502" in msg
                or "500" in msg
                or "rate" in msg
                or "quota" in msg
                or "broken pipe" in msg
                or "reset" in msg
                or "stream" in msg
            )
            if attempt < max_retries - 1 and retryable:
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, 60.0)
                continue
            break
    return None, last_msg


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        _fsync_textio(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Qwen3.5-omni-plus ASR: template JSONL -> child_labels.asr.jsonl"
    )
    p.add_argument(
        "--template",
        type=Path,
        default=_DEFAULT_TEMPLATE,
        help="child_labels.template.jsonl 路径",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="儿童数据集 output-dir（与 pipeline 一致；用于解析 file_path 相对 audios/）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUT,
        help="输出 JSONL（默认同目录 child_labels.asr.jsonl）",
    )
    p.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="记录 failed_indices 的 JSON（默认: <output>.asr_state.json）",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="仅根据状态文件中的 failed_indices 补跑（需已有与模板等长的 --output）",
    )
    p.add_argument(
        "--max-passes",
        type=int,
        default=1,
        metavar="N",
        help="最多执行 N 轮：第 1 轮处理本轮待处理全集，之后每轮仅重试上一轮仍失败的索引，"
        "直至无失败或达到 N（默认 1，即单轮全量）",
    )
    p.add_argument(
        "--per-item-retries",
        type=int,
        default=5,
        help="单条 ASR 请求的最大重试次数（指数退避，默认 5）",
    )
    p.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="单条首次重试前等待秒数（默认 1.0）",
    )
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 行，0=全部")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示 tqdm 进度条（便于重定向日志）",
    )
    args = p.parse_args()

    if args.max_passes < 1:
        print("--max-passes 须 >= 1", file=sys.stderr)
        return 1

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = args.template.parent

    state_path = args.state_file if args.state_file is not None else _default_state_path(args.output)

    if not args.template.is_file():
        print(f"未找到模板: {args.template}", file=sys.stderr)
        return 1

    rows_template, line_nos = _load_template_rows(args.template, args.limit)
    n = len(rows_template)
    if n == 0:
        print("模板无有效数据行", file=sys.stderr)
        return 1

    _api = str(_ROOT / "api_call")
    if _api not in sys.path:
        sys.path.insert(0, _api)
    from api_call_qwen import transcribe_qwen

    # 结果槽位：与模板顺序一一对应
    results: list[dict[str, Any]] = [dict(r) for r in rows_template]

    if args.resume:
        if not args.output.is_file():
            print(f"--resume 需要已存在的输出: {args.output}", file=sys.stderr)
            return 1
        prev_rows = _read_jsonl_rows(args.output)
        if len(prev_rows) != n:
            print(
                f"--resume 时输出行数 {len(prev_rows)} 与模板有效行数 {n} 不一致",
                file=sys.stderr,
            )
            return 1
        results = [dict(r) for r in prev_rows]
        failed_from_disk, _ = _read_state(state_path)
        pending = [i for i in failed_from_disk if i < n]
        if not pending:
            print("状态文件中无待补跑索引，退出")
            _write_state(state_path, failed_indices=[], last_pass=0)
            return 0
        print(f"--resume: 待补跑索引 {len(pending)} 个 -> {pending[:20]}{'...' if len(pending) > 20 else ''}")
        pending_set = set(pending)
        pass_no = 1
        failed: set[int] = set()
        while pass_no <= args.max_passes and pending_set:
            batch = sorted(pending_set)
            failed = set()
            print(f"== pass {pass_no}/{args.max_passes}（补跑 {len(batch)} 条）")
            bar_kw: dict[str, Any] = {
                "disable": args.no_progress,
                "unit": "seg",
                "desc": f"Qwen ASR resume {pass_no}/{args.max_passes}",
            }
            for p, i in enumerate(tqdm(batch, **bar_kw)):
                row = rows_template[i]
                clip, err = _resolve_clip(row, out_dir)
                if err:
                    print(f"[{i}] 跳过 ASR: {err}（模板行 {line_nos[i]}）", file=sys.stderr)
                    results[i] = dict(row)
                    results[i]["content"] = ""
                    failed.add(i)
                else:
                    text, terr = _transcribe_qwen_with_retries(
                        clip,
                        max_retries=args.per_item_retries,
                        base_sleep=args.retry_sleep,
                        user="apply_qwen_asr_to_labels",
                        transcribe_qwen=transcribe_qwen,
                    )
                    if text is not None:
                        results[i] = dict(row)
                        results[i]["content"] = text
                        failed.discard(i)
                    else:
                        print(f"[{i}] ASR 失败（模板行 {line_nos[i]}）: {terr}", file=sys.stderr)
                        results[i] = dict(row)
                        results[i]["content"] = ""
                        failed.add(i)
                _write_output(args.output, results)
                _write_state(
                    state_path,
                    failed_indices=_state_failed_snapshot(failed, batch, p),
                    last_pass=pass_no,
                )
            if not failed:
                print(f"补跑完成，全部成功。Wrote {n} rows -> {args.output}")
                return 0
            pending_set = failed
            pass_no += 1
        print(
            f"Wrote {n} rows -> {args.output}；仍有 {len(failed)} 条失败，见 {state_path}",
            file=sys.stderr,
        )
        return 0

    # 非 resume：第 1 轮全量，其后仅重试失败索引（若 max_passes>1）
    failed: set[int] = set()
    pass_no = 1
    while pass_no <= args.max_passes:
        if pass_no == 1:
            batch = list(range(n))
        else:
            batch = sorted(failed)
            if not batch:
                break
            failed = set()
        print(f"== pass {pass_no}/{args.max_passes}（本批 {len(batch)} 条）")
        bar_kw = {
            "disable": args.no_progress,
            "unit": "seg",
            "desc": f"Qwen ASR {pass_no}/{args.max_passes}",
        }
        for p, i in enumerate(tqdm(batch, **bar_kw)):
            row = rows_template[i]
            clip, err = _resolve_clip(row, out_dir)
            if err:
                print(f"[{i}] 跳过 ASR: {err}（模板行 {line_nos[i]}）", file=sys.stderr)
                results[i] = dict(row)
                results[i]["content"] = ""
                failed.add(i)
            else:
                text, terr = _transcribe_qwen_with_retries(
                    clip,
                    max_retries=args.per_item_retries,
                    base_sleep=args.retry_sleep,
                    user="apply_qwen_asr_to_labels",
                    transcribe_qwen=transcribe_qwen,
                )
                if text is not None:
                    results[i] = dict(row)
                    results[i]["content"] = text
                    failed.discard(i)
                else:
                    print(f"[{i}] ASR 失败（模板行 {line_nos[i]}）: {terr}", file=sys.stderr)
                    results[i] = dict(row)
                    results[i]["content"] = ""
                    failed.add(i)
            _write_output(args.output, results)
            _write_state(
                state_path,
                failed_indices=_state_failed_snapshot(failed, batch, p),
                last_pass=pass_no,
            )
        if not failed:
            print(f"Wrote {n} rows -> {args.output}")
            return 0
        pass_no += 1

    print(
        f"Wrote {n} rows -> {args.output}；仍有 {len(failed)} 条失败，见 {state_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
