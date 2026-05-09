#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3.5：对 TTS 后 JSONL 中每轮 ``tts_audio`` 做多模态听音质检（Gemini：system + 用户说明 + 音频 inline_data）。

通过规则：同一条 manifest 的 **所有** 含 ``tts_audio`` 的轮次均 ``is_pass: true`` 且 JSON 可解析，且每轮均存在可读的 TTS 文件；否则该 manifest **整行** 不通过。
仅 ``line_passed`` 的样本整行写入 ``assistant_responses_with_tts.qc_passed.jsonl``。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
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
_QC = _REPO.parent
if str(_QC) not in sys.path:
    sys.path.insert(0, str(_QC))
_API = _ROOT / "api_call"
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

import local_api_logger.logger as _lm  # noqa: E402

_lm.set_log_dir(str(_API / "api_logs"))
import local_api_logger.tracker as _tr  # noqa: E402

_tr._default_tracker.logger = _lm._default_logger
from local_api_logger import wrap_requests_call  # noqa: E402
from qc_parse import is_tts_s2s_qc_passed, parse_tts_s2s_qc_json_text  # noqa: E402

_DEFAULT_IN = _ROOT / "outputs" / "assistant_responses_with_tts.jsonl"
_DEFAULT_OUT = _ROOT / "outputs" / "qc" / "stage3_5_gemini_qc.jsonl"
_DEFAULT_PASSED = _ROOT / "outputs" / "assistant_responses_with_tts.qc_passed.jsonl"
_DEFAULT_BASE = "http://azpro.xunxkj.cn"
_MODEL = "gemini-3.1-pro-preview"
HEADERS = {"Content-Type": "application/json"}

S2S_SYSTEM = """你是一位资深的专业配音导演和音频质量检测专家。你的任务是听一段由 TTS（语音合成）生成的音频，并评估其表现力与音频质量。
受众是儿童，所以声音必须自然、有感情、不机械。"""

S2S_USER_TEMPLATE = """请仔细聆听我上传的语音合成音频，并对照以下目标进行评估：

【预期目标】
- 应该朗读的文本：\"{plain_text}\"
- 预期的情感基调：\"{acoustic_emotion}\"

【评估维度】
1. 语音清晰度与无瑕疵 (Audio Clarity & Artifacts) [1-5分]：是否有明显的机器电音、杂音、不自然的长时间停顿、断音？
2. 情感表现力 (Emotional Expressiveness) [1-5分]：声音是否听起来像设定的情感基调？是否生动、像真人在对话？
3. 语调与自然度 (Prosody & Naturalness) [1-5分]：重音是否正确？断句是否符合自然人类的说话习惯？

【输出要求】
请仅输出合法 JSON 格式，不要包含 // 注释或代码 fence，结构如下（字段名与类型须一致）：
{{
  "scores": {{
    "clarity": 4,
    "emotion": 5,
    "naturalness": 4
  }},
  "is_pass": true,
  "audio_issues": ["如果有具体瑕疵，例如'第3秒有杂音'，列在这里，否则为空数组"],
  "review_summary": "总体听感评价"
}}
规则：只要有任何一项分数低于3分，或听出明显的电音/破音，则 is_pass 必须为 false。"""


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
    if "error" in resp_json:
        return f"[error] {resp_json.get('error')!r}"
    return ""


def _mime_for_path(p: Path) -> str:
    s = p.suffix.lower()
    if s == ".wav":
        return "audio/wav"
    if s in (".m4a", ".mp4", ".m4b"):
        return "audio/mp4"
    if s in (".mp3", ".mpga"):
        return "audio/mpeg"
    if s == ".flac":
        return "audio/flac"
    if s == ".ogg":
        return "audio/ogg"
    return "audio/mpeg"


def _resolve_tts_path(rel: str) -> Path:
    p = Path(rel.strip())
    if p.is_file():
        return p.resolve()
    r = _ROOT / p
    if r.is_file():
        return r.resolve()
    return r


def _system_field_recoverable(msg: str) -> bool:
    m = msg.lower()
    return "400" in m or "unknown" in m or "invalid" in m or "field" in m or "system" in m


def _call_gemini_tts_qc(
    url: str,
    model: str,
    system_instruction: str,
    user_text: str,
    audio_b64: str,
    mime_type: str,
) -> dict[str, Any]:
    gen_cfg = {"response_mime_type": "application/json"}
    payload_sys: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_text},
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                ],
            }
        ],
        "generation_config": gen_cfg,
        "stream": False,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }
    try:
        return wrap_requests_call(
            model=model,
            url=url,
            headers=HEADERS,
            payload=payload_sys,
            user="stage3_5_qc",
            verify=False,
        )
    except Exception as e_sys:  # noqa: BLE001
        if not _system_field_recoverable(str(e_sys)):
            raise
        combined = f"System:\n{system_instruction}\n\nUser:\n{user_text}"
        payload_fb: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": combined},
                        {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    ],
                }
            ],
            "generation_config": gen_cfg,
            "stream": False,
        }
        return wrap_requests_call(
            model=model,
            url=url,
            headers=HEADERS,
            payload=payload_fb,
            user="stage3_5_qc",
            verify=False,
        )


def _build_user_prompt(plain: str, emotion: str) -> str:
    return S2S_USER_TEMPLATE.format(
        plain_text=plain if plain else "（无）",
        acoustic_emotion=emotion if emotion else "（无）",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=_DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--qc-passed-out",
        type=Path,
        default=_DEFAULT_PASSED,
        help="仅写入 line_passed=true 的 TTS 整行（与 --input 同 schema）",
    )
    ap.add_argument("--base", type=str, default=_DEFAULT_BASE)
    ap.add_argument(
        "--limit", type=int, default=0, help="仅处理前 N 条有内容的 manifest 行（0 为不限制）"
    )
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
        for line in tqdm(fin, desc="stage3.5 QC", unit="line"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ml = rec.get("manifest_line")
            turns = rec.get("turns")
            if not isinstance(turns, list) or not turns:
                continue

            turns_qc: list[dict[str, Any]] = []
            line_ok = True

            for t in turns:
                if not isinstance(t, dict):
                    line_ok = False
                    continue
                ti = t.get("turn_index")
                rel = t.get("tts_audio")
                plain = (t.get("plain_text") or "").strip()
                emo = (t.get("acoustic_emotion") or "").strip()

                if not rel or not str(rel).strip():
                    turns_qc.append(
                        {
                            "turn_index": ti,
                            "is_pass": False,
                            "scores": None,
                            "raw_qc": "",
                            "error": "missing tts_audio",
                        }
                    )
                    line_ok = False
                    continue

                apath = _resolve_tts_path(str(rel).strip())
                if not apath.is_file():
                    turns_qc.append(
                        {
                            "turn_index": ti,
                            "is_pass": False,
                            "scores": None,
                            "raw_qc": "",
                            "error": f"TTS 文件不存在: {apath}",
                        }
                    )
                    line_ok = False
                    continue

                audio_b64 = base64.standard_b64encode(apath.read_bytes()).decode("ascii")
                user_text = _build_user_prompt(plain, emo)
                try:
                    resp = _call_gemini_tts_qc(
                        url, _MODEL, S2S_SYSTEM, user_text, audio_b64, _mime_for_path(apath)
                    )
                except Exception as exc:  # noqa: BLE001
                    turns_qc.append(
                        {
                            "turn_index": ti,
                            "is_pass": False,
                            "scores": None,
                            "raw_qc": "",
                            "error": f"api_error: {exc}",
                        }
                    )
                    line_ok = False
                    continue

                text = _extract_text(resp)
                if re.search(r"^\[error\]", text.strip()[:80]):
                    turns_qc.append(
                        {
                            "turn_index": ti,
                            "is_pass": False,
                            "scores": None,
                            "raw_qc": text,
                            "error": "gemini error payload",
                        }
                    )
                    line_ok = False
                    continue

                parsed = parse_tts_s2s_qc_json_text(text)
                t_out: dict[str, Any] = {
                    "turn_index": ti,
                    "is_pass": parsed.get("is_pass"),
                    "scores": parsed.get("scores"),
                    "audio_issues": parsed.get("audio_issues", []),
                    "review_summary": parsed.get("review_summary", ""),
                    "raw_qc": text,
                }
                pe = parsed.get("parse_error")
                if pe:
                    t_out["parse_error"] = pe
                turns_qc.append(t_out)
                if not is_tts_s2s_qc_passed(parsed):
                    line_ok = False

            if not turns_qc:
                line_ok = False

            out_row: dict[str, Any] = {
                "manifest_line": ml,
                "turns_qc": turns_qc,
                "line_passed": line_ok,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            _fsync_textio(fout)
            n += 1
            if line_ok and turns_qc:
                fpass.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _fsync_textio(fpass)
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
