#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读 --step 1 产出的 child_labels.template.jsonl，对儿童片段调 Qwen ASR，写出 child_labels.asr.jsonl。

通过将仓库 api_call/ 加入 sys.path 以 import api_call_qwen（不修改 api_call 内文件；日志在 api_call/api_logs）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[2]


_ROOT = _repo_root()
_DEFAULT_TEMPLATE = _ROOT / "outputs" / "child_dataset" / "child_labels.template.jsonl"
_DEFAULT_OUT = _ROOT / "outputs" / "child_dataset" / "child_labels.asr.jsonl"


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
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 行，0=全部")
    args = p.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = args.template.parent

    if not args.template.is_file():
        print(f"未找到模板: {args.template}", file=sys.stderr)
        return 1

    _api = str(_ROOT / "api_call")
    if _api not in sys.path:
        sys.path.insert(0, _api)
    from api_call_qwen import transcribe_qwen

    rows_out: list[dict[str, Any]] = []
    n = 0
    with args.template.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {line_no} 行 JSON 无效: {e}", file=sys.stderr)
                return 1
            fp = str(row.get("file_path") or "")
            if not fp:
                print(f"第 {line_no} 行缺少 file_path", file=sys.stderr)
                return 1
            clip = Path(fp)
            if not clip.is_absolute():
                clip = (out_dir / clip).resolve()
            if not clip.is_file():
                print(f"第 {line_no} 行片段不存在: {clip}", file=sys.stderr)
                return 1
            text = transcribe_qwen(clip, user="apply_qwen_asr_to_labels")
            row2 = dict(row)
            row2["content"] = (text or "").strip()
            rows_out.append(row2)
            n += 1
            if args.limit and n >= args.limit:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for r in rows_out:
            out.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(rows_out)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
