# -*- coding: utf-8 -*-
"""从模型返回的纯文本中解析 Stage 2.5 / 3.5 约定的 JSON 质检结果。"""
from __future__ import annotations

import json
import re
from typing import Any

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t, count=1)
    return t.strip()


def _coerce_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "通过"):
            return True
        if s in ("false", "0", "no", "不通过", "不"):
            return False
    if isinstance(val, (int, float)) and val in (0, 1):
        return bool(val)
    return None


def _normalize_parsed_obj(obj: dict[str, Any]) -> dict[str, Any]:
    passed = _coerce_bool(obj.get("passed"))
    summary = obj.get("summary")
    summary_s = summary.strip() if isinstance(summary, str) else ""
    issues = obj.get("issues")
    issues_l: list[str] = []
    if isinstance(issues, list):
        issues_l = [str(x) for x in issues if x is not None]
    return {
        "passed": passed,
        "summary": summary_s,
        "issues": issues_l,
    }


def _try_json_loads(blob: str) -> dict[str, Any] | None:
    s = blob.strip()
    for candidate in (s, _fix_common_qc_json_typos(s)):
        try:
            o = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict):
            return _normalize_parsed_obj(o)
    m = _JSON_OBJ_RE.search(blob)
    if m:
        inner = m.group(0)
        for candidate in (inner, _fix_common_qc_json_typos(inner)):
            try:
                o = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict):
                return _normalize_parsed_obj(o)
    return None


def _fix_common_qc_json_typos(s: str) -> str:
    t = s
    t = re.sub(r'"passed"\s*:\s*true或false', '"passed": false', t, flags=re.IGNORECASE)
    t = re.sub(r'"passed"\s*:\s*true或true', '"passed": true', t, flags=re.IGNORECASE)
    t = re.sub(r'"passed"\s*:\s*false或true', '"passed": true', t, flags=re.IGNORECASE)
    t = re.sub(r'"passed"\s*:\s*false或false', '"passed": false', t, flags=re.IGNORECASE)
    return t


def _regex_passed_fallback(raw: str) -> bool | None:
    m = re.search(r'["\']?passed["\']?\s*:\s*(true|false)\b', raw, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def parse_qc_json_text(raw: str) -> dict[str, Any]:
    """
    返回 dict：passed (bool|None), summary, issues, parse_error (str|None)。
    passed 为 None 表示无法判定，**下游应按未通过处理**。
    """
    if not (raw or "").strip():
        return {
            "passed": None,
            "summary": "",
            "issues": [],
            "parse_error": "empty model text",
        }
    stripped = _strip_code_fence(raw)
    obj = _try_json_loads(stripped)
    if obj is not None and obj.get("passed") is not None:
        return {**obj, "parse_error": None}
    if obj is not None:
        return {
            **obj,
            "parse_error": "missing passed field",
        }
    fb = _regex_passed_fallback(stripped) or _regex_passed_fallback(raw)
    if fb is not None:
        return {
            "passed": fb,
            "summary": "",
            "issues": [],
            "parse_error": "regex passed fallback only",
        }
    return {
        "passed": None,
        "summary": "",
        "issues": [],
        "parse_error": "failed to parse QC JSON from model text",
    }


def is_qc_passed(parsed: dict[str, Any]) -> bool:
    return parsed.get("passed") is True
