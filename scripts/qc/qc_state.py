"""Shared QC checkpoint helpers (aligned with Qwen ASR / assistant_state pattern)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, TextIO


def default_qc_state_path(output: Path) -> Path:
    return output.with_name(output.stem + ".qc_state.json")


def fsync_textio(f: TextIO) -> None:
    try:
        f.flush()
        os.fsync(f.fileno())
    except OSError:
        try:
            f.flush()
        except OSError:
            pass


def write_qc_state(path: Path, *, failed_indices: list[int], last_pass: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"failed_indices": sorted(failed_indices), "last_pass": last_pass}
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        fsync_textio(f)


def read_qc_state(path: Path) -> tuple[list[int], int]:
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


def state_failed_snapshot(failed: set[int], batch: set[int], finished: set[int]) -> list[int]:
    return sorted(failed | (batch - finished))


def load_jsonl_by_manifest_line(path: Path) -> dict[int, dict[str, Any]]:
    """Last wins if duplicate manifest_line in file."""
    out: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ml = obj.get("manifest_line")
            if isinstance(ml, int):
                out[ml] = obj
    return out


def rewrite_jsonl_store(
    path: Path,
    store: dict[int, dict[str, Any]],
    *,
    fsync_fn: Callable[[TextIO], None] = fsync_textio,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ml in sorted(store.keys()):
            f.write(json.dumps(store[ml], ensure_ascii=False) + "\n")
        fsync_fn(f)
