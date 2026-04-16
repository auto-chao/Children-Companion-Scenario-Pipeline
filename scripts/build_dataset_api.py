#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块一 · 仅 API + ffmpeg 切分：不加载 pyannote/torch/FireRed/ST/前端增强。

依赖：ffmpeg/ffprobe、GEMINI_PROXY_API_KEY、网络。

用法：
  python scripts/build_dataset_api.py --input-dir data/audio --output-dir outputs/child_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[2]


_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ccs_audio_pipeline.api_child_segmentation import run_file  # noqa: E402


def _resolve_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    k = os.environ.get("GEMINI_PROXY_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not k:
        raise SystemExit(
            "请设置 GEMINI_PROXY_API_KEY，或使用 --api-key"
        )
    return k


def main() -> int:
    p = argparse.ArgumentParser(
        description="儿童数据集 manifest（API 分段 + ffmpeg 切片，无本地模型）"
    )
    p.add_argument("--input-dir", type=Path, default=Path("data/audio"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/child_dataset"))
    p.add_argument(
        "--audio-mime",
        type=str,
        default="audio/mp4",
        help="inline_data MIME（m4a 常用 audio/mp4）",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("GEMINI_SEGMENT_MODEL", "gemini-3-flash-preview"),
    )
    p.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("GEMINI_PROXY_BASE", "http://azpro.xunxkj.cn"),
    )
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-sleep", type=float, default=1.0)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 个 m4a 文件",
    )
    args = p.parse_args()

    try:
        api_key = _resolve_api_key(args.api_key)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1

    inp = args.input_dir
    if not inp.is_dir():
        print(f"输入目录不存在: {inp}", file=sys.stderr)
        return 1

    files = sorted(inp.glob("*.m4a"), key=lambda x: x.name)
    if args.limit is not None:
        files = files[: max(0, args.limit)]
    if not files:
        print(f"未找到 m4a 文件: {inp}", file=sys.stderr)
        return 1

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    all_lines: list[dict] = []
    for fp in files:
        print(f"处理: {fp.name} ...", flush=True)
        try:
            lines = run_file(
                audio_path=fp,
                output_dir=out,
                audio_mime=args.audio_mime,
                api_base=args.api_base,
                api_key=api_key,
                model_name=args.model,
                max_turns=args.max_turns,
                max_retries=args.max_retries,
                base_sleep=args.retry_sleep,
            )
        except Exception as e:
            print(f"  失败: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if not lines:
            print(f"  跳过（无儿童段）: {fp.name}", file=sys.stderr)
            continue
        all_lines.extend(lines)
        print(f"  写入 {len(lines)} 行 manifest 片段", flush=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in all_lines:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"完成: {manifest_path}（共 {len(all_lines)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
