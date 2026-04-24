#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2.5：对 assistant_responses_multiturn.jsonl 按行调用 GPT-5.4 做质量校验（复用 api_call_gpt54，不修改 api_call/）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve()
_ROOT = next(
    (p for p in _REPO.parents if (p / "pyproject.toml").is_file()),
    _REPO.parents[3],
)
_API = _ROOT / "api_call"
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))
from api_call_gpt54 import chat_gpt54  # noqa: E402

_DEFAULT_IN = _ROOT / "outputs" / "assistant_responses_multiturn.jsonl"
_DEFAULT_OUT = _ROOT / "outputs" / "qa" / "stage2_5_gpt54_qc.jsonl"

QC_SYSTEM = """你是儿童对话数据集的数据质检员。请仔细比对【儿童当前ASR文本】与【assistant的多轮JSON输出】，判断其是否合格。

# 核心判定标准（触碰任意一条即为不合格 passed: false）：
1. 幻觉与过度解读：当ASR片段混乱或语义不清时，assistant是否无中生有地编造了儿童根本没提到的具体场景、剧情或人物行为？semantic_content 是否有主观臆测（如孩子没哭写嚎啕大哭，没说原因却脑补了原因）？
2. 违规澄清问法：plain_text 中是否出现了“你刚才说的是XX吗？”、“你的意思是XX对吗？”等反问澄清句式？
3. 语气违和（爹妈味/幼教风）：plain_text 是否出现了长篇大论的说教、过度做作的拟人化安抚（如“玩具也需要休息”），或者使用“我会一直陪着你的”等像长辈/幼教老师的话？
4. 句式复读机：是否频繁使用“哇！你也太厉害了吧！”、“太酷了吧！”等套路化夸张开场白？

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=_DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"未找到: {args.input}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ml = rec.get("manifest_line")
            turns = rec.get("turns")
            if not isinstance(turns, list):
                continue
            body = _turns_to_text(turns)
            if not body.strip():
                continue
            user = (
                f"manifest_line={ml!r}。以下是该条样本的 turns 信息（每行一个 JSON 对象），请质检：\n"
                + body
            )
            text = chat_gpt54(
                user,
                system=QC_SYSTEM,
                user="stage2_5_qc",
            )
            out = {
                "manifest_line": ml,
                "raw_qc": text,
                "source_model": rec.get("model"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"Wrote {n} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
