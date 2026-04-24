#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3.5：对 TTS 后 JSONL 中（孩子 query + 玩伴 plain_text）做 s2s 质量与适宜性校验（Gemini，不修改 api_call/ 下文件）。
仅 passed 的样本写入 ``assistant_responses_with_tts.qc_passed.jsonl``，与输入行一一对应（含 manifest_line）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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

import local_api_logger.logger as _lm  # noqa: E402

_lm.set_log_dir(str(_API / "api_logs"))
import local_api_logger.tracker as _tr  # noqa: E402

_tr._default_tracker.logger = _lm._default_logger
from local_api_logger import wrap_requests_call  # noqa: E402
from qc_parse import is_qc_passed, parse_qc_json_text  # noqa: E402

_DEFAULT_IN = _ROOT / "outputs" / "assistant_responses_with_tts.jsonl"
_DEFAULT_OUT = _ROOT / "outputs" / "qa" / "stage3_5_gemini_qc.jsonl"
_DEFAULT_PASSED = _ROOT / "outputs" / "assistant_responses_with_tts.qc_passed.jsonl"
_DEFAULT_BASE = "http://azpro.xunxkj.cn"
_MODEL = "gemini-3.1-pro-preview"
HEADERS = {"Content-Type": "application/json"}

QC_INSTRUCTION = """你是端到端儿童语音陪伴系统的最终质检员（Stage 3.5）。输入为一段多轮「孩子 query文本」与「AI 玩伴回复 plain_text」的对话（即将或已转为TTS语音）。

请站在“通过语音听取”的场景下，评估 AI 的回复是否合格：
1) 听感与语感：文本转成语音后是否自然口语化？句子是否过长导致孩子失去耐心？是否存在像读课文、读百科书一样的书面语生硬感？
2) 角色一致性：AI 是否始终维持一个真实、平等的10岁左右“大哥哥/大姐姐”形象，而没有变成居高临下的“小老师”或“父母替身”？
3) 倾听与节奏：AI 是否说得太多（抢戏）？是否给儿童留出了接话的口子（通过自然的好奇追问）？
4) 安全与红线：是否诱导儿童进行危险动作、暴露隐私、或谈论恐怖不适内容？

请只输出一个 JSON 对象：
{
  "passed": true或false,
  "summary": "基于语音交互场景的总体评估评价",
  "issues":["如果不合格，列出导致听感差、不自然或越界的具体原句或问题。如果合格则为空数组"]
}"""


def _proxy_key() -> str:
    k = os.environ.get("GEMINI_PROXY_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not k:
        raise RuntimeError("请设置 GEMINI_PROXY_API_KEY 或 GEMINI_API_KEY")
    return k


def _extract_text(resp_json: dict[str, Any]) -> str:
    try:
        candidates = resp_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p["text"] for p in parts if "text" in p]
            return "\n".join(texts)
    except (KeyError, TypeError, IndexError):
        pass
    return ""


def _s2s_block(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        q = (t.get("query") or t.get("transcript_ref") or "").strip()
        r = (t.get("plain_text") or "").strip()
        if not q and not r:
            continue
        lines.append(f"孩子：{q}\n玩伴：{r}")
    return "\n---\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=_DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--qc-passed-out",
        type=Path,
        default=_DEFAULT_PASSED,
        help="仅写入 passed=true 的 TTS 输出行（与 --input 同 schema）",
    )
    ap.add_argument("--base", type=str, default=_DEFAULT_BASE)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"未找到: {args.input}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_passed_out.parent.mkdir(parents=True, exist_ok=True)
    base = args.base.rstrip("/")
    api_key = _proxy_key()
    url = f"{base}/v1beta/models/{_MODEL}:generateContent?key={api_key}"
    n = 0
    n_passed = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout, args.qc_passed_out.open(
        "w", encoding="utf-8"
    ) as fpass:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ml = rec.get("manifest_line")
            turns = rec.get("turns")
            if not isinstance(turns, list):
                continue
            block = _s2s_block(turns)
            if not block.strip():
                continue
            user_text = QC_INSTRUCTION + "\n\n【待检对话】\n" + block
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_text}],
                    }
                ],
                "stream": False,
            }
            resp = wrap_requests_call(
                model=_MODEL,
                url=url,
                headers=HEADERS,
                payload=payload,
                user="stage3_5_qc",
                verify=False,
            )
            text = _extract_text(resp)
            parsed = parse_qc_json_text(text)
            out: dict[str, Any] = {
                "manifest_line": ml,
                "raw_qc": text,
                "passed": parsed.get("passed"),
                "summary": parsed.get("summary", ""),
                "issues": parsed.get("issues", []),
            }
            pe = parsed.get("parse_error")
            if pe:
                out["parse_error"] = pe
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if is_qc_passed(parsed):
                fpass.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_passed += 1
            if args.limit and n >= args.limit:
                break

    print(f"Stage 3.5 QC 行数={n}，通过={n_passed} -> {args.qc_passed_out}")
    print(f"Wrote {n} rows -> {args.output}")
    if n > 0 and n_passed == 0:
        print(
            "\n"
            "================================================================\n"
            "【警告】Stage 3.5：本批有质检样本但 0 条通过。\n"
            f"  请检查: {args.output}\n"
            f"  若需可交付子集: {args.qc_passed_out}（当前为空或应忽略）\n"
            "================================================================\n",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
