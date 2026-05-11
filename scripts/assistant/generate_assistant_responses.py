#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
儿童陪伴 AI 助手回复批量生成（Gemini 兼容 HTTP + JSONL）

与 ``api_call/api_call_final.py`` 一致：请求体中带 **inline_data 音频**（m4a），由多模态模型
理解儿童语音并生成回复。manifest 中的 ``user`` 等为归档/对齐；**当前轮**仍以音频为主输入。

多轮模式（``--mode multi``）：每一轮 API 请求在 ``contents`` 末尾为单条 ``user``：**孩子侧文本历史**（与 manifest 转写一致）**+ 历史轮 API 的 plain_text（玩伴回复）**，一直到本轮孩子话，再拼任务说明与**本轮**儿童音频 ``inline_data``。
模型 JSON 含 ``semantic_content``、``acoustic_emotion``、``plain_text``。

调用经 ``local_api_logger.wrap_requests_call`` 记录到 ``api_call/api_logs/``。

环境变量
--------
GEMINI_PROXY_API_KEY（推荐）
    第三方代理提供的 API Key（勿提交到仓库）。
GEMINI_PROXY_BASE（可选）
    代理根地址，默认 ``http://azpro.xunxkj.cn``。

若未设置 ``GEMINI_PROXY_API_KEY``，会回退读取 ``GEMINI_API_KEY``（仅作别名，仍表示代理密钥）。

用法
----
    python scripts/assistant/generate_assistant_responses.py

    # 默认 --mode multi：读 manifest.jsonl、写 assistant_responses_multiturn.jsonl
    # 单轮：加 --mode single 且必须指定 --input（单轮 manifest JSONL），写 assistant_responses_single_turn.jsonl

可选参数见 --help。无参数时与 ``--mode multi`` 相同。

每行 JSON 为 **一个 manifest 样本**，顶层含 ``manifest_line``、``model``、``input_mode``、``turns``（每轮一条）、
``line_error``；单轮 ``len(turns)==1``，多轮为多元素数组。
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
from typing import Any, TextIO

from tqdm import tqdm

# 仓库根目录


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[2]


_ROOT = _repo_root()
_API_CALL_ROOT = _ROOT / "api_call"
if str(_API_CALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_CALL_ROOT))

# 日志目录与 api_call_final 一致：api_call/api_logs
import local_api_logger.logger as _lm

_lm.set_log_dir(str(_API_CALL_ROOT / "api_logs"))
import local_api_logger.tracker as _tr

_tr._default_tracker.logger = _lm._default_logger
from local_api_logger import wrap_requests_call
from retry_policy import is_retryable_error_message, sleep_before_next_attempt

_DEFAULT_MULTI_IN = _ROOT / "outputs" / "child_dataset" / "manifest.jsonl"
_DEFAULT_SINGLE_OUT = _ROOT / "outputs" / "assistant_responses_single_turn.jsonl"
_DEFAULT_MULTI_OUT = _ROOT / "outputs" / "assistant_responses_multiturn.jsonl"
# gemini-3.1-pro-preview才能完美结合acoustic和semantic信息，gemini-3.1-flash-lite-preview输出质量偏差
_DEFAULT_MODEL = "gemini-3-flash-preview"
_DEFAULT_BASE = "http://azpro.xunxkj.cn"
_DEFAULT_MIME = "audio/mp4"
_CHILD_DATASET_ROOT = _ROOT / "outputs" / "child_dataset"

HEADERS = {"Content-Type": "application/json"}

_RECORDING_CTX_HEADER = "【多轮对话上下文（孩子转写 + 历史玩伴回复）】"

_ASSIST_DIR = _ROOT / "scripts" / "assistant"
if str(_ASSIST_DIR) not in sys.path:
    sys.path.insert(0, str(_ASSIST_DIR))
from criteria_text import full_task_text as _criteria_full_task_text


def _full_task_text() -> str:
    return _criteria_full_task_text()


def _normalize_model_json_object(obj: dict[str, Any]) -> dict[str, Any]:
    """校验并规范化 API 返回的 JSON。"""
    pt = obj.get("plain_text")
    if not isinstance(pt, str) or not pt.strip():
        raise ValueError("plain_text missing or empty")
    sem = obj.get("semantic_content")
    ae = obj.get("acoustic_emotion")
    sem_s = sem.strip() if isinstance(sem, str) else ""
    ae_s = ae.strip() if isinstance(ae, str) else ""
    return {
        "plain_text": pt.strip(),
        "semantic_content": sem_s,
        "acoustic_emotion": ae_s,
    }


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t, count=1)
    return t.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = _strip_json_fence(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    return obj


def extract_text_from_generate_content(resp_json: dict[str, Any]) -> str:
    """从 Gemini 兼容 generateContent 响应中提取文本。"""
    try:
        candidates = resp_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p["text"] for p in parts if "text" in p]
            return "\n".join(texts)
    except (KeyError, TypeError, IndexError):
        pass
    return ""


def _resolve_proxy_key(args: argparse.Namespace) -> str:
    if getattr(args, "api_key", None):
        return args.api_key
    k = os.environ.get("GEMINI_PROXY_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not k:
        raise RuntimeError(
            "请设置环境变量 GEMINI_PROXY_API_KEY（第三方代理密钥），"
            "或传入 --api-key"
        )
    return k


def _resolve_audio_path(audio_rel: str) -> Path:
    p = Path(audio_rel)
    if p.is_absolute():
        return p
    return _CHILD_DATASET_ROOT / p


def _build_payload(
    *,
    contents: list[dict[str, Any]],
    use_google_search: bool,
    json_mode: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contents": contents,
        "stream": False,
    }
    if use_google_search:
        payload["tools"] = [{"google_search": {}}]
    if json_mode:
        payload["generation_config"] = {
            "temperature": 1.0,
            "response_mime_type": "application/json",
        }
    return payload


def _audio_key_for_turn(turn_1based: int) -> str:
    if turn_1based < 1:
        raise ValueError("turn_1based must be >= 1")
    return "audio" if turn_1based == 1 else f"audio_{turn_1based}"


def _user_transcripts_from_messages(row: dict[str, Any]) -> list[str]:
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return []
    out: list[str] = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            out.append(str(m.get("text") or ""))
    return out


def _turns_from_manifest_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    """每轮 (audio 相对路径, 转写归档)."""
    transcripts = _user_transcripts_from_messages(row)
    if not transcripts:
        audio = row.get("audio") or ""
        q = row.get("user") or ""
        if audio:
            return [(audio, q)]
        return []
    turns: list[tuple[str, str]] = []
    for i, tr in enumerate(transcripts):
        key = _audio_key_for_turn(i + 1)
        audio = row.get(key) or ""
        turns.append((audio, tr))
    return turns


def _child_transcript_for_turn(row: dict[str, Any], turn_1based: int) -> str:
    """返回指定轮次的孩子转写文本。"""
    if turn_1based < 1:
        raise ValueError("turn_1based must be >= 1")
    turns = _turns_from_manifest_row(row)
    idx = turn_1based - 1
    if idx >= len(turns):
        return ""
    return str(turns[idx][1] or "").strip()


def _multiturn_api_history_text(
    row: dict[str, Any], turn_1based: int, prior_plain_texts: list[str]
) -> str:
    """构造第 k 轮请求用历史文本：孩子1 + 回复1 + ... + 孩子k。"""
    if turn_1based < 1:
        raise ValueError("turn_1based must be >= 1")
    if len(prior_plain_texts) != max(0, turn_1based - 1):
        raise ValueError(
            f"prior_plain_texts length mismatch: got={len(prior_plain_texts)}, expected={turn_1based - 1}"
        )
    parts: list[str] = []
    for ti in range(1, turn_1based + 1):
        child = _child_transcript_for_turn(row, ti)
        parts.append(f"孩子：{child}")
        if ti < turn_1based:
            parts.append(f"【玩伴回复】{prior_plain_texts[ti - 1]}")
    return "\n".join(parts)


def _call_proxy_with_contents(
    *,
    base: str,
    api_key: str,
    model_name: str,
    contents: list[dict[str, Any]],
    max_retries: int,
    base_sleep: float,
    use_google_search: bool,
) -> dict[str, Any]:
    base = base.rstrip("/")
    url = f"{base}/v1beta/models/{model_name}:generateContent?key={api_key}"

    last_err: Exception | None = None
    sleep_s = base_sleep

    for attempt in range(max_retries):
        payload = _build_payload(
            contents=contents,
            use_google_search=use_google_search,
            json_mode=True,
        )

        try:
            resp_json = wrap_requests_call(
                model=model_name,
                url=url,
                headers=HEADERS,
                payload=payload,
                user="assistant_batch",
                verify=False,
            )
            text = extract_text_from_generate_content(resp_json).strip()
            if not text and "error" in resp_json:
                raise RuntimeError(f"API error: {resp_json.get('error')}")
            if not text:
                raise ValueError("empty model text in response")
            obj = _parse_json_object(text)
            return _normalize_model_json_object(obj)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if (
                attempt == 0
                and "generation_config" in payload
                and ("400" in msg or "unknown" in msg or "invalid" in msg or "field" in msg)
            ):
                try:
                    payload = _build_payload(
                        contents=contents,
                        use_google_search=use_google_search,
                        json_mode=False,
                    )
                    resp_json = wrap_requests_call(
                        model=model_name,
                        url=url,
                        headers=HEADERS,
                        payload=payload,
                        user="assistant_batch",
                        verify=False,
                    )
                    text = extract_text_from_generate_content(resp_json).strip()
                    if text:
                        obj = _parse_json_object(text)
                        try:
                            return _normalize_model_json_object(obj)
                        except ValueError:
                            pass
                except Exception:
                    pass

            if attempt < max_retries - 1 and is_retryable_error_message(msg):
                sleep_s = sleep_before_next_attempt(sleep_s)
                continue
            break

    assert last_err is not None
    raise last_err


def _call_proxy_audio_single_turn(
    *,
    base: str,
    api_key: str,
    model_name: str,
    audio_path: Path,
    audio_mime: str,
    max_retries: int,
    base_sleep: float,
    use_google_search: bool,
) -> dict[str, Any]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    audio_b64 = base64.standard_b64encode(audio_path.read_bytes()).decode("ascii")
    text_bits: list[str] = []
    text_bits.append(_full_task_text())
    full_text = "\n\n".join(text_bits)
    contents: list[dict[str, Any]] = [
        {
            "role": "user",
            "parts": [
                {"text": full_text},
                {"inline_data": {"mime_type": audio_mime, "data": audio_b64}},
            ],
        }
    ]
    return _call_proxy_with_contents(
        base=base,
        api_key=api_key,
        model_name=model_name,
        contents=contents,
        max_retries=max_retries,
        base_sleep=base_sleep,
        use_google_search=use_google_search,
    )


def _build_multiturn_contents(
    *,
    history_dialogue_text: str,
    current_audio_b64: str,
    audio_mime: str,
) -> list[dict[str, Any]]:
    """单条 user：孩子+历史玩伴回复文本（若有）+ 任务说明 + 本轮儿童音频。"""
    text_bits: list[str] = []
    fd = (history_dialogue_text or "").strip()
    if fd:
        text_bits.append(_RECORDING_CTX_HEADER + "\n" + fd)
    text_bits.append(_full_task_text())
    combined = "\n\n\n".join(text_bits)
    return [
        {
            "role": "user",
            "parts": [
                {"text": combined},
                {"inline_data": {"mime_type": audio_mime, "data": current_audio_b64}},
            ],
        }
    ]


def _load_done_single_skip(out_path: Path) -> tuple[set[int], set[str]]:
    """新格式：已写入的 manifest_line；旧格式（无 turns）：顶层 audio。"""
    done_ml: set[int] = set()
    done_audio: set[str] = set()
    if not out_path.is_file():
        return done_ml, done_audio
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ml = rec.get("manifest_line")
            if isinstance(ml, int) and isinstance(rec.get("turns"), list):
                done_ml.add(ml)
                continue
            a = rec.get("audio")
            if isinstance(a, str) and a and "turns" not in rec:
                done_audio.add(a)
    return done_ml, done_audio


def _load_multiturn_existing_records(out_path: Path) -> dict[int, dict[str, Any]]:
    """manifest 行号 -> 聚合行对象（含 turns）；兼容旧版每轮一行扁平记录。"""
    by_line: dict[int, dict[str, Any]] = {}
    legacy_flat: dict[int, list[dict[str, Any]]] = {}
    if not out_path.is_file():
        return by_line
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ml = rec.get("manifest_line")
            if not isinstance(ml, int):
                continue
            if isinstance(rec.get("turns"), list):
                by_line[ml] = rec
            elif rec.get("turn_index") is not None:
                legacy_flat.setdefault(ml, []).append(rec)

    for ml, flat_list in legacy_flat.items():
        if ml in by_line:
            continue
        flat_list.sort(key=lambda r: int(r.get("turn_index") or 0))
        turns: list[dict[str, Any]] = []
        for r in flat_list:
            q = r.get("query")
            tr = r.get("transcript_ref")
            turns.append(
                {
                    "turn_index": r.get("turn_index"),
                    "audio": r.get("audio"),
                    "query": q if isinstance(q, str) else tr,
                    "transcript_ref": tr if isinstance(tr, str) else q,
                    "plain_text": r.get("plain_text"),
                    "semantic_content": r.get("semantic_content"),
                    "acoustic_emotion": r.get("acoustic_emotion"),
                    "error": r.get("error"),
                }
            )
        by_line[ml] = {
            "manifest_line": ml,
            "model": flat_list[0].get("model"),
            "input_mode": flat_list[0].get("input_mode") or "audio_multiturn",
            "turns": turns,
            "line_error": None,
        }
    return by_line


def _multiturn_resume_state(
    existing: dict[str, Any] | None, total_turns: int
) -> tuple[int, list[str]] | None:
    """若该行已全部成功则返回 None；否则返回 (下一轮次, 历史 plain_text 列表)。"""
    if existing is None:
        return 1, []
    turns_list = existing.get("turns")
    if not isinstance(turns_list, list):
        return 1, []
    success: dict[int, str] = {}
    for t in turns_list:
        if not isinstance(t, dict):
            continue
        ti = t.get("turn_index")
        err = t.get("error")
        pt = t.get("plain_text")
        if isinstance(ti, int) and err is None and isinstance(pt, str) and pt.strip():
            success[ti] = pt.strip()
    if len(success) >= total_turns:
        return None
    next_t = 1
    prior_plain_texts: list[str] = []
    while next_t in success:
        prior_plain_texts.append(success[next_t])
        next_t += 1
    if next_t > total_turns:
        return None
    return next_t, prior_plain_texts


def main() -> int:
    p = argparse.ArgumentParser(
        description="从 manifest 读取片段，按音频调用代理生成儿童陪伴回复（与 api_call_final 同型）。"
    )
    p.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="multi",
        help="single：单轮 manifest（须同时指定 --input）；multi：多轮 manifest.jsonl，按轮构造上下文（默认 multi）",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="输入 manifest JSONL（multi 默认 outputs/child_dataset/manifest.jsonl；single 必须显式指定）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL（默认随 --mode：assistant_responses_single_turn / assistant_responses_multiturn）",
    )
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL, help="Gemini 兼容模型名（代理侧）")
    p.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("GEMINI_PROXY_BASE", _DEFAULT_BASE),
        help="代理根 URL（默认 env GEMINI_PROXY_BASE 或 azpro）",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="代理 API Key（默认从环境变量 GEMINI_PROXY_API_KEY 读取）",
    )
    p.add_argument(
        "--audio-mime",
        type=str,
        default=_DEFAULT_MIME,
        help="inline_data MIME（默认 audio/mp4；失败可试 audio/x-m4a）",
    )
    p.add_argument(
        "--with-google-search",
        action="store_true",
        help="与 api_call_final 一致，请求中带 google_search 工具（默认不带）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="仅处理前 N 条 manifest 行（省略则处理全部；可为 0 表示不处理）",
    )
    p.add_argument("--no-resume", action="store_true", help="不跳过输出文件中已有记录")
    p.add_argument("--max-retries", type=int, default=5, help="每条请求最大重试次数")
    p.add_argument("--retry-sleep", type=float, default=1.0, help="首次重试前等待秒数（指数退避）")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发 worker 数（默认 4）。按 manifest 行并发；多轮同一行内仍串行保证上下文。",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示 tqdm 进度条（便于重定向日志）",
    )
    args = p.parse_args()

    if args.mode == "single" and args.input is None:
        print(
            "错误: --mode single 时必须指定 --input（仓库不再生成 manifest_single_turn.jsonl）。"
            " 或改用默认的 --mode multi。",
            file=sys.stderr,
        )
        return 1

    in_path: Path = (
        args.input if args.input is not None else _DEFAULT_MULTI_IN
    )
    out_path: Path = (
        args.output
        if args.output is not None
        else (_DEFAULT_MULTI_OUT if args.mode == "multi" else _DEFAULT_SINGLE_OUT)
    )

    if not in_path.is_file():
        print(f"输入文件不存在: {in_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    multiturn_existing: dict[int, dict[str, Any]] = {}
    if args.mode == "multi" and not args.no_resume:
        multiturn_existing = _load_multiturn_existing_records(out_path)
    done_ml: set[int] = set()
    done_audio: set[str] = set()
    if args.mode == "single" and not args.no_resume:
        done_ml, done_audio = _load_done_single_skip(out_path)

    lines: list[str] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    if args.limit is not None:
        lines = lines[: args.limit]

    if not lines:
        print(f"无待处理行，退出。输出文件：{out_path}")
        return 0

    try:
        api_key = _resolve_proxy_key(args)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    n_ok = n_skip = n_err = 0
    n_api = 0
    workers = max(1, int(args.workers))
    write_lock = threading.Lock()

    def _fsync_textio(f: TextIO) -> None:
        try:
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            try:
                f.flush()
            except OSError:
                pass

    def _append_output_record(rec: dict[str, Any]) -> None:
        with write_lock:
            with out_path.open("a", encoding="utf-8") as outf:
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _fsync_textio(outf)

    def _accumulate_and_maybe_write(r: dict[str, Any]) -> None:
        nonlocal n_api, n_ok, n_skip, n_err
        if r.get("message"):
            print(str(r["message"]), file=sys.stderr)
        n_api += int(r.get("n_api", 0))
        st = r.get("status")
        if st == "skip":
            n_skip += 1
        elif st == "ok":
            n_ok += 1
            if isinstance(r.get("record"), dict):
                _append_output_record(r["record"])
        else:
            n_err += 1
            if isinstance(r.get("record"), dict):
                _append_output_record(r["record"])

    def _process_single_line(manifest_line: int, line: str) -> dict[str, Any]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            return {
                "manifest_line": manifest_line,
                "status": "err",
                "n_api": 0,
                "record": None,
                "message": f"跳过无效 JSON 行 {manifest_line}: {e}",
            }

        query = row.get("user") or ""
        if not query and row.get("messages"):
            msgs = row["messages"]
            if isinstance(msgs, list) and msgs:
                m0 = msgs[0]
                if isinstance(m0, dict) and m0.get("role") == "user":
                    query = m0.get("text") or ""
        audio = row.get("audio", "")
        audio_path = _resolve_audio_path(audio) if isinstance(audio, str) and audio else Path()

        if not args.no_resume and (
            manifest_line in done_ml
            or (isinstance(audio, str) and audio and audio in done_audio)
        ):
            return {
                "manifest_line": manifest_line,
                "status": "skip",
                "n_api": 0,
                "record": None,
                "message": None,
            }

        turn0: dict[str, Any] = {
            "turn_index": 1,
            "audio": audio,
            "query": query,
            "transcript_ref": query,
            "plain_text": None,
            "semantic_content": None,
            "acoustic_emotion": None,
            "error": None,
        }
        line_record: dict[str, Any] = {
            "manifest_line": manifest_line,
            "model": args.model,
            "input_mode": "audio",
            "turns": [turn0],
            "line_error": None,
        }

        try:
            if not audio or not audio_path.is_file():
                raise FileNotFoundError(f"无效或缺失音频路径: {audio!r} -> {audio_path}")
            out = _call_proxy_audio_single_turn(
                base=args.api_base,
                api_key=api_key,
                model_name=args.model,
                audio_path=audio_path,
                audio_mime=args.audio_mime,
                max_retries=args.max_retries,
                base_sleep=args.retry_sleep,
                use_google_search=args.with_google_search,
            )
            turn0["plain_text"] = out["plain_text"]
            turn0["semantic_content"] = out.get("semantic_content") or ""
            turn0["acoustic_emotion"] = out.get("acoustic_emotion") or ""
            return {
                "manifest_line": manifest_line,
                "status": "ok",
                "n_api": 1,
                "record": line_record,
                "message": None,
            }
        except Exception as e:
            turn0["error"] = f"{type(e).__name__}: {e}"
            line_record["line_error"] = turn0["error"]
            return {
                "manifest_line": manifest_line,
                "status": "err",
                "n_api": 0,
                "record": line_record,
                "message": None,
            }

    def _process_multi_line(manifest_line: int, line: str) -> dict[str, Any]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            return {
                "manifest_line": manifest_line,
                "status": "err",
                "n_api": 0,
                "record": None,
                "message": f"跳过无效 JSON 行 {manifest_line}: {e}",
            }

        turns = _turns_from_manifest_row(row)
        if not turns:
            return {
                "manifest_line": manifest_line,
                "status": "err",
                "n_api": 0,
                "record": None,
                "message": f"行 {manifest_line}：无可用用户轮次，跳过",
            }

        existing = multiturn_existing.get(manifest_line)
        if args.no_resume:
            start_turn = 1
            prior_plain_texts: list[str] = []
        else:
            rs = _multiturn_resume_state(existing, len(turns))
            if rs is None:
                return {
                    "manifest_line": manifest_line,
                    "status": "skip",
                    "n_api": 0,
                    "record": None,
                    "message": None,
                }
            start_turn, prior_plain_texts = rs

        turn_entries: list[dict[str, Any]] = []
        n_api_local = 0
        for turn_idx in range(start_turn, len(turns) + 1):
            audio_rel, query = turns[turn_idx - 1]
            audio_path = _resolve_audio_path(audio_rel)
            turn_d: dict[str, Any] = {
                "turn_index": turn_idx,
                "query": query,
                "transcript_ref": query,
                "audio": audio_rel,
                "plain_text": None,
                "semantic_content": None,
                "acoustic_emotion": None,
                "error": None,
            }
            try:
                if not audio_rel or not audio_path.is_file():
                    raise FileNotFoundError(f"无效或缺失音频路径: {audio_rel!r} -> {audio_path}")
                cur_b64 = base64.standard_b64encode(audio_path.read_bytes()).decode("ascii")
                hist_txt = _multiturn_api_history_text(row, turn_idx, prior_plain_texts)
                contents = _build_multiturn_contents(
                    history_dialogue_text=hist_txt,
                    current_audio_b64=cur_b64,
                    audio_mime=args.audio_mime,
                )
                out = _call_proxy_with_contents(
                    base=args.api_base,
                    api_key=api_key,
                    model_name=args.model,
                    contents=contents,
                    max_retries=args.max_retries,
                    base_sleep=args.retry_sleep,
                    use_google_search=args.with_google_search,
                )
                n_api_local += 1
                turn_d["plain_text"] = out["plain_text"]
                turn_d["semantic_content"] = out.get("semantic_content") or ""
                turn_d["acoustic_emotion"] = out.get("acoustic_emotion") or ""
                turn_entries.append(turn_d)
                prior_plain_texts.append(out["plain_text"])
            except Exception as e:
                turn_d["error"] = f"{type(e).__name__}: {e}"
                turn_entries.append(turn_d)
                break

        merged_turns: list[dict[str, Any]] = list(turn_entries)
        if existing and not args.no_resume:
            prev_turns = existing.get("turns")
            if isinstance(prev_turns, list) and start_turn > 1:
                prefix: list[dict[str, Any]] = []
                for t in prev_turns:
                    if isinstance(t, dict) and isinstance(t.get("turn_index"), int):
                        if int(t["turn_index"]) < start_turn:
                            prefix.append(t)
                prefix.sort(key=lambda x: int(x.get("turn_index") or 0))
                merged_turns = prefix + turn_entries

        line_err = next((t.get("error") for t in merged_turns if t.get("error")), None)
        line_record: dict[str, Any] = {
            "manifest_line": manifest_line,
            "model": args.model,
            "input_mode": "audio_multiturn",
            "turns": merged_turns,
            "line_error": line_err,
        }
        ok = (len(merged_turns) == len(turns)) and not any(
            t.get("error") for t in merged_turns
        )
        return {
            "manifest_line": manifest_line,
            "status": "ok" if ok else "err",
            "n_api": n_api_local,
            "record": line_record,
            "message": None,
        }

    if args.mode == "single":
        inputs = [(i + 1, line) for i, line in enumerate(lines)]
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_process_single_line, ml, ln) for ml, ln in inputs]
                for fut in tqdm(
                    concurrent.futures.as_completed(futs),
                    total=len(futs),
                    desc="assistant single",
                    unit="manifest",
                    disable=args.no_progress,
                ):
                    _accumulate_and_maybe_write(fut.result())
        else:
            for ml, ln in tqdm(
                inputs,
                desc="assistant single",
                unit="manifest",
                disable=args.no_progress,
            ):
                _accumulate_and_maybe_write(_process_single_line(ml, ln))

        print(
            f"完成：样本成功 {n_ok}，跳过 {n_skip}，样本失败/无效 {n_err}，API 调用 {n_api}，写入 {out_path}"
        )
        print(f"API 日志目录：{_API_CALL_ROOT / 'api_logs'}")
        if n_ok == 0 and n_err > 0:
            return 1
        return 0

    # multi：每个 manifest 样本一行 JSON（turns 数组），按行并发；行内轮次串行。
    inputs = [(i + 1, line) for i, line in enumerate(lines)]
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_multi_line, ml, ln) for ml, ln in inputs]
            for fut in tqdm(
                concurrent.futures.as_completed(futs),
                total=len(futs),
                desc="assistant multi",
                unit="manifest",
                disable=args.no_progress,
            ):
                _accumulate_and_maybe_write(fut.result())
    else:
        for ml, ln in tqdm(
            inputs,
            desc="assistant multi",
            unit="manifest",
            disable=args.no_progress,
        ):
            _accumulate_and_maybe_write(_process_multi_line(ml, ln))

    print(
        f"完成：样本成功 {n_ok}，跳过 {n_skip}，样本失败/无效 {n_err}，API 调用 {n_api}，写入 {out_path}"
    )
    print(f"API 日志目录：{_API_CALL_ROOT / 'api_logs'}")
    if n_ok == 0 and n_err > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
