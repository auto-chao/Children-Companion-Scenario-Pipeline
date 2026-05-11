#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2.5：对 assistant_responses_multiturn.jsonl 按行调用 GPT-5.4 做质量校验；仅 passed 的样本写入
``assistant_responses_multiturn.qc_passed.jsonl`` 供后续 TTS。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm


def _fsync_textio(f: TextIO) -> None:
    try:
        f.flush()
        os.fsync(f.fileno())
    except OSError:
        try:
            f.flush()
        except OSError:
            pass

_REPO = Path(__file__).resolve()
_ROOT = next(
    (p for p in _REPO.parents if (p / "pyproject.toml").is_file()),
    _REPO.parents[3],
)
_QA = _REPO.parent
if str(_QA) not in sys.path:
    sys.path.insert(0, str(_QA))
_API = _ROOT / "api_call"
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))
from api_call_gpt54 import chat_gpt54  # noqa: E402
from qc_parse import is_qc_passed, parse_qc_json_text
from retry_policy import is_retryable_error_message, sleep_before_next_attempt  # noqa: E402

_DEFAULT_IN = _ROOT / "outputs" / "assistant_responses_multiturn.jsonl"
_DEFAULT_OUT = _ROOT / "outputs" / "qc" / "stage2_5_gpt54_qc.jsonl"
_DEFAULT_PASSED = _ROOT / "outputs" / "assistant_responses_multiturn.qc_passed.jsonl"

QC_SYSTEM = """你是儿童对话数据集的数据质检员。请仔细检验【儿童当前ASR文本】与【assistant的多轮JSON输出】，判断其是否合格。

# 核心判定标准：
1. 语义理解与回复标准：正确理解儿童ASR内容，给出符合5-10岁儿童认知水平的回复。
2. 互动标准：始终围绕儿童当前发起的话题展开对话，不主动结束话题，可适度引导儿童表达自身的想法与感受。
3. 情绪共情标准：充分共情儿童的情绪，针对儿童的正向表达给予真诚的鼓励，针对负面情绪给予温暖的安慰与支持。
4. 安全合规标准：严格规避暴力、恐怖等不适宜儿童的话题与内容；针对危险行为、不当遭遇，必须第一时间干预，明确告知儿童需第一时间告诉父母/老师，并提供安全的替代方案。
5. 角色人设标准：始终保持高年级同龄玩伴的人设，平等对话，自然接话，不摆架子、不刻意装可爱、不做作；绝对禁止重复历史对话中的回复内容与句式，保证每轮回复的原创性；给予正确的行为、认知、情感引导，帮助儿童建立正确的价值观和世界观。

请只输出一个 JSON 对象，格式如下：
{
  "passed": true或false,
  "summary": "简明扼要的通过/拒绝理由总结",
  "issues": ["如果不合格，列出具体的违规点，带上文本证据。如果合格则为空数组"]
}
"""


def _turns_to_text(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        lines.append(
            json.dumps(
                {
                    "turn_index": t.get("turn_index"),
                    "query": t.get("query"),
                    "plain_text": t.get("plain_text"),
                    "semantic_content": t.get("semantic_content"),
                    "acoustic_emotion": t.get("acoustic_emotion"),
                    "error": t.get("error"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _chat_gpt54_with_retries(
    user_text: str,
    *,
    system: str,
    max_retries: int,
    base_sleep: float,
) -> str:
    n = max(1, max_retries)
    sleep_s = base_sleep
    for attempt in range(n):
        try:
            return chat_gpt54(user_text, system=system, user="stage2_5_qc")
        except Exception as e:
            if attempt < n - 1 and is_retryable_error_message(str(e)):
                sleep_s = sleep_before_next_attempt(sleep_s)
                continue
            raise


def _process_gpt_qc_row(
    ord_i: int,
    rec: dict[str, Any],
    *,
    max_retries: int,
    base_sleep: float,
) -> tuple[int, dict[str, Any], dict[str, Any], bool]:
    ml = rec.get("manifest_line")
    body = _turns_to_text(rec["turns"])
    user = (
        f"manifest_line={ml!r}。以下是该条样本的 turns 信息（每行一个 JSON 对象），请质检：\n"
        + body
    )
    text = _chat_gpt54_with_retries(
        user,
        system=QC_SYSTEM,
        max_retries=max_retries,
        base_sleep=base_sleep,
    )
    parsed = parse_qc_json_text(text)
    out: dict[str, Any] = {
        "manifest_line": ml,
        "raw_qc": text,
        "source_model": rec.get("model"),
        "passed": parsed.get("passed"),
        "summary": parsed.get("summary", ""),
        "issues": parsed.get("issues", []),
    }
    pe = parsed.get("parse_error")
    if pe:
        out["parse_error"] = pe
    passed = is_qc_passed(parsed)
    return ord_i, out, rec, passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=_DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--qc-passed-out",
        type=Path,
        default=_DEFAULT_PASSED,
        help="仅写入质检 passed=true 的原始行，供 TTS 使用",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=5, help="单次 GPT 质检 API 最大重试次数")
    ap.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="首次重试前等待秒数（指数退避）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行质检的 manifest 行数（默认 1）",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"未找到: {args.input}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_passed_out.parent.mkdir(parents=True, exist_ok=True)

    n_input_nonempty = 0
    to_run: list[tuple[int, dict[str, Any]]] = []
    with args.input.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_input_nonempty += 1
            rec = json.loads(line)
            turns = rec.get("turns")
            if not isinstance(turns, list):
                continue
            body = _turns_to_text(turns)
            if not body.strip():
                continue
            to_run.append((len(to_run), rec))
            if args.limit and len(to_run) >= args.limit:
                break

    workers = max(1, int(args.workers))
    results: list[tuple[int, dict[str, Any], dict[str, Any], bool]] = []

    def _job(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, Any], bool]:
        oi, rec = item
        return _process_gpt_qc_row(
            oi,
            rec,
            max_retries=args.max_retries,
            base_sleep=args.retry_sleep,
        )

    if to_run:
        if workers > 1 and len(to_run) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_job, item) for item in to_run]
                for fut in tqdm(
                    concurrent.futures.as_completed(futs),
                    total=len(futs),
                    desc="stage2.5 QC",
                    unit="line",
                ):
                    results.append(fut.result())
        else:
            for item in tqdm(to_run, desc="stage2.5 QC", unit="line"):
                results.append(_job(item))

    results.sort(key=lambda x: x[0])
    n = 0
    n_passed = 0
    with args.output.open("w", encoding="utf-8") as fout, args.qc_passed_out.open(
        "w", encoding="utf-8"
    ) as fpass:
        for _oi, out, rec, passed in results:
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            _fsync_textio(fout)
            n += 1
            if passed:
                fpass.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _fsync_textio(fpass)
                n_passed += 1

    print(f"QC 行数={n}，通过={n_passed} -> {args.qc_passed_out}")
    print(f"Wrote {n} rows -> {args.output}")
    if n > 0 and n_passed == 0:
        print("本批有质检样本但 0 条通过，退出码 2（不进入 TTS）。", file=sys.stderr)
        return 2
    if n == 0 and n_input_nonempty > 0:
        print("输入有内容但无有效 turns 可质检，退出码 1。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
