#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3.5：对 TTS 后 JSONL 中每轮 ``tts_audio`` 做多模态听音质检（Gemini：system + 用户说明 + 音频 inline_data）。

通过规则：同一条 manifest 的 **所有** 含 ``tts_audio`` 的轮次均 ``is_pass: true`` 且 JSON 可解析，且每轮均存在可读的 TTS 文件；否则该 manifest **整行** 不通过。
仅 ``line_passed`` 的样本整行写入 ``assistant_responses_with_tts.qc_passed.jsonl``。

续跑与 Qwen ASR / generate_assistant_responses 对齐：默认写 ``<output>.qc_state.json``（``failed_indices`` + ``last_pass``），
支持 ``--state-file`` / ``--resume`` / ``--max-passes``；每条完成后持锁更新内存并按 ``manifest_line`` 排序重写双 JSONL。
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
from qc_state import (  # noqa: E402
    default_qc_state_path,
    load_jsonl_by_manifest_line,
    read_qc_state,
    rewrite_jsonl_store,
    state_failed_snapshot,
    write_qc_state,
)
from retry_policy import is_retryable_error_message, sleep_before_next_attempt  # noqa: E402

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


def _call_gemini_tts_qc_with_retries(
    url: str,
    model: str,
    system_instruction: str,
    user_text: str,
    audio_b64: str,
    mime_type: str,
    *,
    max_retries: int,
    base_sleep: float,
) -> dict[str, Any]:
    n = max(1, max_retries)
    sleep_s = base_sleep
    for attempt in range(n):
        try:
            return _call_gemini_tts_qc(
                url, model, system_instruction, user_text, audio_b64, mime_type
            )
        except Exception as e:
            if attempt < n - 1 and is_retryable_error_message(str(e)):
                sleep_s = sleep_before_next_attempt(sleep_s)
                continue
            raise


def _build_user_prompt(plain: str, emotion: str) -> str:
    return S2S_USER_TEMPLATE.format(
        plain_text=plain if plain else "（无）",
        acoustic_emotion=emotion if emotion else "（无）",
    )


def _process_tts_manifest_record(
    rec: dict[str, Any],
    url: str,
    *,
    max_retries: int,
    base_sleep: float,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """返回 (out_row, 原始 rec, 是否写入 qc_passed)。"""
    ml = rec.get("manifest_line")
    turns = rec.get("turns")
    if not isinstance(turns, list) or not turns:
        out_row: dict[str, Any] = {
            "manifest_line": ml,
            "turns_qc": [],
            "line_passed": False,
        }
        return out_row, rec, False

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
            resp = _call_gemini_tts_qc_with_retries(
                url,
                _MODEL,
                S2S_SYSTEM,
                user_text,
                audio_b64,
                _mime_for_path(apath),
                max_retries=max_retries,
                base_sleep=base_sleep,
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

    out_row = {
        "manifest_line": ml,
        "turns_qc": turns_qc,
        "line_passed": line_ok,
    }
    return out_row, rec, bool(line_ok and turns_qc)


def _safe_tts_job(
    rec: dict[str, Any],
    url: str,
    *,
    max_retries: int,
    base_sleep: float,
) -> tuple[int, dict[str, Any], dict[str, Any], bool, bool]:
    """返回 (ml, out_row, rec, write_passed, technical_failed)。"""
    ml = rec.get("manifest_line")
    if not isinstance(ml, int):
        out_row: dict[str, Any] = {
            "manifest_line": ml,
            "turns_qc": [],
            "line_passed": False,
            "error": "manifest_line 缺失或非 int",
        }
        return -1, out_row, rec, False, True
    try:
        out_row, src, write_pass = _process_tts_manifest_record(
            rec,
            url,
            max_retries=max_retries,
            base_sleep=base_sleep,
        )
        return ml, out_row, src, write_pass, False
    except Exception as e:  # noqa: BLE001
        out_row = {
            "manifest_line": ml,
            "turns_qc": [],
            "line_passed": False,
            "error": f"{type(e).__name__}: {e}",
        }
        return ml, out_row, rec, False, True


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
    ap.add_argument("--max-retries", type=int, default=5, help="单次听音 API 最大重试次数")
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
        help="并行处理的 manifest 行数（默认 1，行内各 turn 仍串行）",
    )
    ap.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="记录 failed_indices 的 JSON（默认: <output_stem>.qc_state.json）",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="仅根据状态文件中的 failed_indices 补跑（需已有 --output）",
    )
    ap.add_argument(
        "--max-passes",
        type=int,
        default=1,
        help="最大轮数；首轮处理缺 detail 的行，后续轮仅重试上一轮仍技术失败的 manifest_line",
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

    state_path = (
        args.state_file if args.state_file is not None else default_qc_state_path(args.output)
    )

    detail_store: dict[int, dict[str, Any]] = load_jsonl_by_manifest_line(args.output)
    passed_store: dict[int, dict[str, Any]] = load_jsonl_by_manifest_line(args.qc_passed_out)

    n_input_nonempty = 0
    rec_by_ml: dict[int, dict[str, Any]] = {}
    with args.input.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_input_nonempty += 1
            rec = json.loads(line)
            ml = rec.get("manifest_line")
            if not isinstance(ml, int):
                print(
                    f"跳过 manifest_line 非 int 的行（输入第 {n_input_nonempty} 条非空行）",
                    file=sys.stderr,
                )
                continue
            turns = rec.get("turns")
            if not isinstance(turns, list) or not turns:
                continue
            rec_by_ml[ml] = rec
            if args.limit and len(rec_by_ml) >= args.limit:
                break

    resume_start_batch: list[int] | None = None
    if args.resume:
        if not args.output.is_file():
            print(f"--resume 需要已存在的输出: {args.output}", file=sys.stderr)
            return 1
        pending_raw, _ = read_qc_state(state_path)
        pending_list = sorted({p for p in pending_raw if p in rec_by_ml})
        if not pending_list:
            print("状态文件中无待补 manifest 行，退出")
            write_qc_state(state_path, failed_indices=[], last_pass=0)
            return 0
        print(
            f"--resume: 待补 {len(pending_list)} 行 -> {pending_list[:20]}{'...' if len(pending_list) > 20 else ''}"
        )
        resume_start_batch = pending_list

    workers = max(1, int(args.workers))
    write_lock = threading.Lock()

    def _run_pass(batch_mls: list[int], pass_no: int) -> set[int]:
        if not batch_mls:
            return set()
        batch_set = set(batch_mls)
        finished: set[int] = set()
        pass_failed: set[int] = set()

        def _accumulate(
            ml: int,
            out_row: dict[str, Any],
            rec: dict[str, Any],
            write_passed: bool,
            technical_failed: bool,
        ) -> None:
            nonlocal detail_store, passed_store
            if ml < 0:
                return
            with write_lock:
                finished.add(ml)
                detail_store[ml] = out_row
                if write_passed:
                    passed_store[ml] = rec
                else:
                    passed_store.pop(ml, None)
                if technical_failed:
                    pass_failed.add(ml)
                else:
                    pass_failed.discard(ml)
                snap = state_failed_snapshot(pass_failed, batch_set, finished)
                write_qc_state(state_path, failed_indices=snap, last_pass=pass_no)
                rewrite_jsonl_store(args.output, detail_store)
                rewrite_jsonl_store(args.qc_passed_out, passed_store)

        def _dispatch(rec: dict[str, Any]) -> None:
            ml, out_row, src, write_pass, technical_failed = _safe_tts_job(
                rec,
                url,
                max_retries=args.max_retries,
                base_sleep=args.retry_sleep,
            )
            _accumulate(ml, out_row, src, write_pass, technical_failed)

        inputs = [rec_by_ml[ml] for ml in sorted(batch_mls) if ml in rec_by_ml]
        if not inputs:
            return set()

        if workers > 1 and len(inputs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_dispatch, rec) for rec in inputs]
                for fut in tqdm(
                    concurrent.futures.as_completed(futs),
                    total=len(futs),
                    desc=f"stage3.5 QC pass {pass_no}/{args.max_passes}",
                    unit="line",
                ):
                    fut.result()
        else:
            for rec in tqdm(
                inputs,
                desc=f"stage3.5 QC pass {pass_no}/{args.max_passes}",
                unit="line",
            ):
                _dispatch(rec)
        return set(pass_failed)

    max_passes = max(1, int(args.max_passes))
    last_failed: set[int] = set()

    if resume_start_batch is not None:
        pass_no = 1
        current_batch = list(resume_start_batch)
        while pass_no <= max_passes:
            if not current_batch:
                break
            last_failed = _run_pass(current_batch, pass_no)
            if not last_failed:
                break
            pass_no += 1
            current_batch = sorted(last_failed)
    else:
        pass_no = 1
        while pass_no <= max_passes:
            if pass_no == 1:
                current_batch = sorted(ml for ml in rec_by_ml if ml not in detail_store)
            else:
                current_batch = sorted(last_failed)
                if not current_batch:
                    break
            last_failed = _run_pass(current_batch, pass_no)
            if not last_failed:
                break
            pass_no += 1

    if not last_failed:
        write_qc_state(state_path, failed_indices=[], last_pass=0)

    n_scoped = len(rec_by_ml)
    n_passed_scoped = sum(1 for ml in rec_by_ml if ml in passed_store)

    print(f"Stage 3.5 QC 输入有效 manifest 行数={n_scoped}，其中通过={n_passed_scoped} -> {args.qc_passed_out}")
    print(f"详情行数（全量 store）={len(detail_store)} -> {args.output}")
    print(f"状态文件：{state_path}")

    if n_scoped > 0 and n_passed_scoped == 0:
        print(
            "\n"
            "================================================================\n"
            "【警告】Stage 3.5：本批有质检样本但 0 条通过。\n"
            f"  请检查: {args.output}\n"
            f"  若需可交付子集: {args.qc_passed_out}（当前为空或应忽略）\n"
            "================================================================\n",
            file=sys.stderr,
        )
    if n_scoped == 0 and n_input_nonempty > 0:
        print("输入有内容但无有效 turns 可质检。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
