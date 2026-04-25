"""Gemini-compatible HTTP ASR: send WAV bytes to proxy, return transcript text."""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_API_CALL_ROOT = _REPO_ROOT / "api_call"
if str(_API_CALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_CALL_ROOT))

import local_api_logger.logger as _lm

_lm.set_log_dir(str(_API_CALL_ROOT / "api_logs"))
import local_api_logger.tracker as _tr

_tr._default_tracker.logger = _lm._default_logger
from local_api_logger import wrap_requests_call  # noqa: E402

HEADERS = {"Content-Type": "application/json"}
_DEFAULT_BASE = "http://azpro.xunxkj.cn"
_DEFAULT_MODEL = "gemini-3-flash-preview"
_DEFAULT_PROMPT = (
"你是一位专业的语音转录专家。这是一段儿童说话的音频片段。"
"语言可能是中文、英文、方言，或两两混合、三者混合（code-switching）。"
"请准确转写。请注意儿童常见的说话模式"
"（例如：口齿不清、吞音、混合使用语言或方言）。"
"仅输出所说语言的转写文本，不要任何翻译或解释。"
)

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


def _parse_json_loose(raw: str) -> dict[str, Any]:
    raw = raw.strip()
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


def _resolve_proxy_key() -> str:
    k = os.environ.get("GEMINI_PROXY_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not k:
        raise RuntimeError(
            "ASR backend=api 需要环境变量 GEMINI_PROXY_API_KEY（或 GEMINI_API_KEY 作为别名）。"
        )
    return k


class GeminiProxyAsr:
    """Multimodal transcribe via same proxy as ``api_call/api_call_final.py``."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        prompt: str | None = None,
        max_retries: int = 5,
        base_sleep: float = 1.0,
    ) -> None:
        self._api_key = _resolve_proxy_key()
        self._base = (base_url or os.environ.get("GEMINI_PROXY_BASE") or _DEFAULT_BASE).rstrip("/")
        self._model = model_name or os.environ.get("GEMINI_ASR_MODEL") or _DEFAULT_MODEL
        self._qc_model = os.environ.get("GEMINI_QC_MODEL") or os.environ.get("GEMINI_QA_MODEL") or self._model
        self._prompt = prompt or os.environ.get("GEMINI_ASR_PROMPT") or _DEFAULT_PROMPT
        self._max_retries = max_retries
        self._base_sleep = base_sleep

    def transcribe(self, clip_wave: np.ndarray, sr: int) -> str:
        if sr <= 0 or len(clip_wave) == 0:
            return ""
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            sf.write(tmp_path, clip_wave, sr, subtype="PCM_16")
            wav_b64 = base64.standard_b64encode(tmp_path.read_bytes()).decode("ascii")
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

        url = f"{self._base}/v1beta/models/{self._model}:generateContent?key={self._api_key}"
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": self._prompt},
                        {"inline_data": {"mime_type": "audio/wav", "data": wav_b64}},
                    ],
                }
            ],
            "stream": False,
        }

        last_err: Exception | None = None
        sleep_s = self._base_sleep
        for attempt in range(self._max_retries):
            try:
                resp_json = wrap_requests_call(
                    model=self._model,
                    url=url,
                    headers=HEADERS,
                    payload=payload,
                    user="pipeline_asr",
                    verify=False,
                )
                text = _extract_text(resp_json).strip()
                if not text and "error" in resp_json:
                    raise RuntimeError(f"ASR API error: {resp_json.get('error')}")
                return text
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                retryable = (
                    "429" in msg
                    or "resource exhausted" in msg
                    or "quota" in msg
                    or "rate" in msg
                    or "503" in msg
                    or "timeout" in msg
                    or "502" in msg
                )
                if attempt < self._max_retries - 1 and retryable:
                    time.sleep(sleep_s)
                    sleep_s = min(sleep_s * 2, 60.0)
                    continue
                break
        assert last_err is not None
        raise last_err

    def generate_json_from_text(
        self,
        *,
        system_instruction: str,
        user_text: str,
    ) -> dict[str, Any]:
        """Text-only generateContent with JSON response; uses ``GEMINI_QC_MODEL`` (or legacy ``GEMINI_QA_MODEL``) when set, else ASR model."""
        model = self._qc_model
        url = f"{self._base}/v1beta/models/{model}:generateContent?key={self._api_key}"
        gen_cfg = {"response_mime_type": "application/json"}

        def _call(payload: dict[str, Any]) -> dict[str, Any]:
            resp_json = wrap_requests_call(
                model=model,
                url=url,
                headers=HEADERS,
                payload=payload,
                user="pipeline_qc_text",
                verify=False,
            )
            text = _extract_text(resp_json).strip()
            if not text and "error" in resp_json:
                raise RuntimeError(f"QC API error: {resp_json.get('error')}")
            if not text:
                raise ValueError("empty model text in QC response")
            return _parse_json_loose(text)

        def _system_field_recoverable(msg: str) -> bool:
            m = msg.lower()
            return (
                "400" in m
                or "unknown" in m
                or "invalid" in m
                or "field" in m
                or "system" in m
            )

        last_err: Exception | None = None
        sleep_s = self._base_sleep
        for attempt in range(self._max_retries):
            payload_sys: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "generation_config": gen_cfg,
                "stream": False,
                "systemInstruction": {"parts": [{"text": system_instruction}]},
            }
            try:
                return _call(payload_sys)
            except Exception as e_sys:
                last_err = e_sys
                if _system_field_recoverable(str(e_sys)):
                    combined = f"System:\n{system_instruction}\n\nUser:\n{user_text}"
                    payload_fb: dict[str, Any] = {
                        "contents": [{"role": "user", "parts": [{"text": combined}]}],
                        "generation_config": gen_cfg,
                        "stream": False,
                    }
                    try:
                        return _call(payload_fb)
                    except Exception as e_fb:
                        last_err = e_fb
                msg = str(last_err).lower()
                retryable = (
                    "429" in msg
                    or "resource exhausted" in msg
                    or "quota" in msg
                    or "rate" in msg
                    or "503" in msg
                    or "timeout" in msg
                    or "502" in msg
                )
                if attempt < self._max_retries - 1 and retryable:
                    time.sleep(sleep_s)
                    sleep_s = min(sleep_s * 2, 60.0)
                    continue
                break
        assert last_err is not None
        raise last_err
