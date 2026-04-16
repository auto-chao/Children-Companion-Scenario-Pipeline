"""
模块一 · API 分段：仅依赖多模态 API + ffmpeg 切片，不使用本地 ASR/嵌入/pyannote。

输出 manifest 行与 ``pipeline.write_manifest`` 同形（键名一致）。
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_API_CALL = _REPO_ROOT / "api_call"
if str(_API_CALL) not in sys.path:
    sys.path.insert(0, str(_API_CALL))

import local_api_logger.logger as _lm  # noqa: E402

_lm.set_log_dir(str(_API_CALL / "api_logs"))
import local_api_logger.tracker as _tr  # noqa: E402

_tr._default_tracker.logger = _lm._default_logger
from local_api_logger import wrap_requests_call  # noqa: E402

from ccs_audio_pipeline.ffmpeg_utils import build_segment_id, cut_audio, ffprobe_duration_seconds

HEADERS = {"Content-Type": "application/json"}

SEGMENTATION_PROMPT = """
你是纯音频亲子人声提取专家，核心准则：**宁缺毋滥**，听不清、音量过小、无把握的内容直接剔除，绝不臆造、绝不凑数。
# 刚性规则
1.  时间基准：以音频0秒为起点，先获取精确总时长`duration_sec`（秒，保留1位小数）。
2.  儿童片段提取&转写：
    - 仅保留5-10周岁儿童、≥1秒、清晰可懂、音量正常、无其他人声叠加的有效说话片段，排除所有非目标人声、杂音、无意义发声，听不清/音量过小的整段直接丢弃
    - 片段按start_sec升序排列，无重叠，同句≤2秒无其他人声的停顿合并为一段
    - 转写如实还原口语内容，不修改、不脑补、不增减
3.  成人内容提取（无内容则为空字符串""）：
    - `recording_prefix_adult`：首个儿童片段前的成人转写
    - `adult_between_children`：字符串数组，长度严格=max(0,儿童段数-1)，第n项为第n与n+1个儿童片段间的成人转写
    - `adult_suffix_after_last_child`：最后一个儿童片段结束至音频结尾的成人转写
# 输出铁律
仅输出纯标准JSON，无任何额外内容，键名与类型严格如下：
{
  "duration_sec": number,
  "recording_prefix_adult": string,
  "child_segments": [{"start_sec": number, "end_sec": number, "transcript": string}],
  "adult_between_children": [string],
  "adult_suffix_after_last_child": string
}
特殊情况：无符合要求的儿童片段时，child_segments、adult_between_children为空数组，recording_prefix_adult为全音频成人转写，片尾为空。
"""


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t, count=1)
    return t.strip()


def _extract_text_from_generate_content(resp_json: dict[str, Any]) -> str:
    try:
        candidates = resp_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p["text"] for p in parts if "text" in p]
            return "\n".join(texts)
    except (KeyError, TypeError, IndexError):
        pass
    return ""


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


def _call_gemini_json(
    *,
    base: str,
    api_key: str,
    model_name: str,
    contents: list[dict[str, Any]],
    max_retries: int,
    base_sleep: float,
) -> dict[str, Any]:
    base = base.rstrip("/")
    url = f"{base}/v1beta/models/{model_name}:generateContent?key={api_key}"
    last_err: Exception | None = None
    sleep_s = base_sleep
    for attempt in range(max_retries):
        payload: dict[str, Any] = {
            "contents": contents,
            "stream": False,
            "generation_config": {
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        }
        try:
            resp_json = wrap_requests_call(
                model=model_name,
                url=url,
                headers=HEADERS,
                payload=payload,
                user="child_segmentation",
                verify=False,
            )
            text = _extract_text_from_generate_content(resp_json).strip()
            if not text and "error" in resp_json:
                raise RuntimeError(f"API error: {resp_json.get('error')}")
            if not text:
                raise ValueError("empty model text")
            return _parse_json_object(text)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if (
                attempt == 0
                and "generation_config" in payload
                and ("400" in msg or "unknown" in msg or "invalid" in msg or "field" in msg)
            ):
                try:
                    payload2 = {"contents": contents, "stream": False}
                    resp_json = wrap_requests_call(
                        model=model_name,
                        url=url,
                        headers=HEADERS,
                        payload=payload2,
                        user="child_segmentation",
                        verify=False,
                    )
                    text = _extract_text_from_generate_content(resp_json).strip()
                    if text:
                        return _parse_json_object(text)
                except Exception:
                    pass
            retryable = (
                "429" in msg
                or "resource exhausted" in msg
                or "quota" in msg
                or "503" in msg
                or "timeout" in msg
                or "502" in msg
            )
            if attempt < max_retries - 1 and retryable:
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, 60.0)
                continue
            break
    assert last_err is not None
    raise last_err


def call_segmentation_api(
    *,
    audio_path: Path,
    audio_mime: str,
    api_base: str,
    api_key: str,
    model_name: str,
    max_retries: int,
    base_sleep: float,
) -> dict[str, Any]:
    if not audio_path.is_file():
        raise FileNotFoundError(str(audio_path))
    b64 = base64.standard_b64encode(audio_path.read_bytes()).decode("ascii")
    contents: list[dict[str, Any]] = [
        {
            "role": "user",
            "parts": [
                {"text": SEGMENTATION_PROMPT},
                {"inline_data": {"mime_type": audio_mime, "data": b64}},
            ],
        }
    ]
    return _call_gemini_json(
        base=api_base,
        api_key=api_key,
        model_name=model_name,
        contents=contents,
        max_retries=max_retries,
        base_sleep=base_sleep,
    )


def _validate_and_sort_segments(
    raw: list[Any], duration_sec: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            s = float(item["start_sec"])
            e = float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        tr = item.get("transcript")
        if not isinstance(tr, str):
            tr = ""
        s = max(0.0, min(s, duration_sec))
        e = max(0.0, min(e, duration_sec))
        if e - s < 0.05:
            continue
        if e <= s:
            continue
        out.append({"start_sec": s, "end_sec": e, "transcript": tr.strip()})
    out.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
    return out


def normalize_api_result(obj: dict[str, Any], ffprobe_duration: float) -> dict[str, Any]:
    """校验并规范化 API JSON；时间戳以 ffprobe 实测时长为上限。"""
    raw_segments = obj.get("child_segments")
    if not isinstance(raw_segments, list):
        raw_segments = []
    segments = _validate_and_sort_segments(raw_segments, ffprobe_duration)

    prefix = obj.get("recording_prefix_adult")
    prefix_s = prefix.strip() if isinstance(prefix, str) else ""

    between = obj.get("adult_between_children")
    if not isinstance(between, list):
        between = []
    between_strs: list[str] = []
    for x in between:
        between_strs.append(x.strip() if isinstance(x, str) else "")
    n = len(segments)
    need_between = max(0, n - 1)
    while len(between_strs) < need_between:
        between_strs.append("")
    between_strs = between_strs[:need_between]

    suf = obj.get("adult_suffix_after_last_child")
    suf_s = suf.strip() if isinstance(suf, str) else ""

    return {
        "duration_sec": ffprobe_duration,
        "recording_prefix_adult": prefix_s,
        "child_segments": segments,
        "adult_between_children": between_strs,
        "adult_suffix_after_last_child": suf_s,
    }


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        size = 1
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_manifest_line_for_chunk(
    *,
    segments: list[dict[str, Any]],
    global_start_index: int,
    global_total: int,
    prefix: str,
    between: list[str],
    suffix: str,
) -> dict[str, Any]:
    """
    segments: 本行包含的儿童段（全局连续下标 global_start_index .. global_start_index+len-1）。
    """
    line: dict[str, Any] = {"messages": []}
    m = len(segments)
    if m == 0:
        return line

    a = global_start_index

    if a == 0 and prefix:
        line["recording_prefix_adult"] = prefix

    for i in range(m):
        g = a + i
        turn = i + 1
        user_key = "user" if turn == 1 else f"user_{turn}"
        assistant_key = "assistant" if turn == 1 else f"assistant_{turn}"
        audio_key = "audio" if turn == 1 else f"audio_{turn}"

        seg = segments[i]
        line[user_key] = seg["transcript"]
        # audio 路径占位，由调用方在切片后填入
        line[audio_key] = ""

        if i < m - 1:
            ast = between[g] if g < len(between) else ""
        else:
            if g < global_total - 1:
                ast = between[g] if g < len(between) else ""
            else:
                ast = suffix
        line[assistant_key] = ast
        line["messages"].append({"role": "user", "text": seg["transcript"]})
        line["messages"].append({"role": "assistant", "text": ast})

    return line


def cut_segments_and_fill_manifest_audio(
    *,
    source_audio: Path,
    audio_id: str,
    audios_dir: Path,
    manifest_lines: list[dict[str, Any]],
    flat_segments: list[dict[str, Any]],
    global_indices_per_line: list[list[int]],
) -> None:
    """按全局段列表切片，并把每行 manifest 的 audio 键改为相对路径。"""
    seg_idx = 0
    for line_idx, line in enumerate(manifest_lines):
        indices = global_indices_per_line[line_idx]
        turns = len(indices)
        for t in range(turns):
            g = indices[t]
            seg = flat_segments[g]
            turn = t + 1
            audio_key = "audio" if turn == 1 else f"audio_{turn}"
            sid = build_segment_id(audio_id, seg_idx, seg["start_sec"], seg["end_sec"])
            seg_idx += 1
            out_m4a = audios_dir / f"{sid}.m4a"
            cut_audio(source_audio, out_m4a, seg["start_sec"], seg["end_sec"])
            line[audio_key] = str(Path("audios") / f"{sid}.m4a").replace("\\", "/")


def build_manifest_lines(
    norm: dict[str, Any], max_turns: int
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """返回 manifest 行与每行对应的全局段下标列表。"""
    segments: list[dict[str, Any]] = norm["child_segments"]
    prefix = norm["recording_prefix_adult"]
    between = norm["adult_between_children"]
    suffix = norm["adult_suffix_after_last_child"]
    n = len(segments)
    if n == 0:
        return [], []

    chunks = chunk_list(list(range(n)), max_turns)
    lines: list[dict[str, Any]] = []
    global_idx_map: list[list[int]] = []

    for chunk_indices in chunks:
        chunk_segs = [segments[i] for i in chunk_indices]
        a = chunk_indices[0]
        line = build_manifest_line_for_chunk(
            segments=chunk_segs,
            global_start_index=a,
            global_total=n,
            prefix=prefix,
            between=between,
            suffix=suffix,
        )
        # 非首块：片头应为上一段与当前段之间的成人转写
        if a > 0:
            gap_before = between[a - 1] if a - 1 < len(between) else ""
            if gap_before:
                line["recording_prefix_adult"] = gap_before
            elif "recording_prefix_adult" in line:
                del line["recording_prefix_adult"]

        lines.append(line)
        global_idx_map.append(chunk_indices)

    return lines, global_idx_map


def run_file(
    *,
    audio_path: Path,
    output_dir: Path,
    audio_mime: str,
    api_base: str,
    api_key: str,
    model_name: str,
    max_turns: int,
    max_retries: int,
    base_sleep: float,
) -> list[dict[str, Any]]:
    audio_id = audio_path.stem
    dur = ffprobe_duration_seconds(audio_path)
    raw = call_segmentation_api(
        audio_path=audio_path,
        audio_mime=audio_mime,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        max_retries=max_retries,
        base_sleep=base_sleep,
    )
    norm = normalize_api_result(raw, dur)
    lines, idx_map = build_manifest_lines(norm, max_turns)
    if not lines:
        return []

    audios_dir = output_dir / "audios"
    audios_dir.mkdir(parents=True, exist_ok=True)
    flat = norm["child_segments"]
    cut_segments_and_fill_manifest_audio(
        source_audio=audio_path,
        audio_id=audio_id,
        audios_dir=audios_dir,
        manifest_lines=lines,
        flat_segments=flat,
        global_indices_per_line=idx_map,
    )
    return lines
