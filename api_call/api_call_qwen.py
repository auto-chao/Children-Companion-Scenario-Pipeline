#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "httpx[socks]>=0.27.0",
#   "openai>=1.76.0",
#   "imageio-ffmpeg>=0.5.0",
# ]
# ///

import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openai import OpenAI

# 与脚本同目录，便于从任意 cwd 找到 local_api_logger
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from local_api_logger import log_completion, set_log_dir  # noqa: E402

# 与 api_call_final 一致：无论从哪里运行，日志都落在脚本目录下 api_call/api_logs
set_log_dir(str(_script_dir / "api_logs"))


def _fix_ssl_cert_env() -> None:
    """Git Bash 下 conda 可能把 SSL_CERT_FILE 设成不存在的路径（如 .../envs/ccs/ssl/cacert.pem），httpx 会因此失败。"""
    p = os.environ.get("SSL_CERT_FILE")
    if p and not os.path.isfile(p):
        os.environ.pop("SSL_CERT_FILE", None)


_DEFAULT_OPENAI_BASE_URL = "http://azpro.xunxkj.cn/v1"


def _api_key() -> str:
    key = (
        os.environ.get("QWEN_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_PROXY_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "请设置 QWEN_OPENAI_API_KEY / OPENAI_API_KEY，"
            "或复用 GEMINI_PROXY_API_KEY / GEMINI_API_KEY。"
        )
    return key


def _openai_base_url() -> str:
    return (
        os.environ.get("QWEN_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("GEMINI_PROXY_OPENAI_BASE")
        or _DEFAULT_OPENAI_BASE_URL
    )


def build_client() -> OpenAI:
    _fix_ssl_cert_env()
    return OpenAI(api_key=_api_key(), base_url=_openai_base_url())


def _m4a_to_wav16k_mono(m4a: Path) -> Path:
    """m4a 先转为 16k 单声道 WAV（与通义文档/解码端期望一致，避免网关报格式错误）。"""
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    out = Path(tempfile.gettempdir()) / f"ccs_asr_{m4a.stem}.wav"
    r = subprocess.run(
        [exe, "-y", "-i", str(m4a), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
        check=False,
        capture_output=True,
    )
    if r.returncode != 0 or not out.is_file():
        raise SystemExit(
            f"ffmpeg 无法将 m4a 转为 wav: {m4a}\n{r.stderr.decode('utf-8', errors='replace')}"
        )
    return out


def audio_to_input_audio(audio_path: Path) -> tuple[str, str]:
    """
    返回 (data, format) 给 input_audio。
    - gptplus5 等网关在 `data` 为纯 base64 时会报「URL 非法」，需使用 data:audio/...;base64,...
    - 原始 m4a 在部分网关上解码失败，先转为 16k 单声道 WAV 再传。
    """
    path = audio_path
    if path.suffix.lower() == ".m4a":
        path = _m4a_to_wav16k_mono(path)

    suffix = path.suffix.lower().lstrip(".")
    mime_map = {
        "mp3": "audio/mpeg",
        "mpga": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }
    fmt_map = {
        "mp3": "mp3",
        "mpga": "mp3",
        "wav": "wav",
        "flac": "flac",
        "ogg": "ogg",
    }
    mime = mime_map.get(suffix, "audio/mpeg")
    fmt = fmt_map.get(suffix, "mp3")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}", fmt


MODEL = "qwen3.5-omni-plus"
ASR_PROMPT = "请将这段音频的内容转为文字。"
audio_path = _script_dir / "demo.m4a"


def transcribe_qwen(
    path: Path,
    prompt: str = ASR_PROMPT,
    *,
    user: str = "api_call_qwen",
) -> str:
    client = build_client()
    audio_b64, audio_fmt = audio_to_input_audio(path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": audio_fmt},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    request_data = {
        "model": MODEL,
        "messages": messages,
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.time()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    last_usage = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
        u = getattr(chunk, "usage", None)
        if u is not None:
            last_usage = u
    text = "".join(parts)
    duration_ms = (time.time() - start) * 1000

    # 与 local_api_logger.tracker._handle_stream_response 中记录的流式响应结构一致
    usage_dict: dict = {}
    if last_usage is not None and hasattr(last_usage, "model_dump"):
        usage_dict = last_usage.model_dump()
    response_data: dict = {"content": text, "streaming": True}
    if usage_dict:
        response_data["usage"] = usage_dict
    log_completion(
        model=MODEL,
        request_data=request_data,
        response_data=response_data,
        user=user,
        duration_ms=duration_ms,
    )
    return text


def main() -> None:
    if not audio_path.is_file():
        raise SystemExit(f"未找到音频: {audio_path}")
    print(f"模型: {MODEL}")
    print(f"音频: {audio_path}")
    print("---")
    out = transcribe_qwen(audio_path, ASR_PROMPT)
    print("回复:\n", out)


if __name__ == "__main__":
    main()
