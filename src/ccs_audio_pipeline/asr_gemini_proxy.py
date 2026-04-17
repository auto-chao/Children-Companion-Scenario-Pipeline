"""Gemini-compatible HTTP ASR: send WAV bytes to proxy, return transcript text."""

from __future__ import annotations

import base64
import os
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
_DEFAULT_PROMPT = "请转写这段音频中的语音内容，按时间顺序输出文字，不要添加无关说明。"


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
