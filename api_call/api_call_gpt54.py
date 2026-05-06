#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "httpx[socks]>=0.27.0",
#   "openai>=1.76.0",
# ]
# ///

import os
import sys
import time
from pathlib import Path
from typing import List

from openai import OpenAI

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from local_api_logger import log_completion, set_log_dir  # noqa: E402

set_log_dir(str(_script_dir / "api_logs"))


def _fix_ssl_cert_env() -> None:
    p = os.environ.get("SSL_CERT_FILE")
    if p and not os.path.isfile(p):
        os.environ.pop("SSL_CERT_FILE", None)


_DEFAULT_OPENAI_BASE_URL = "http://azpro.xunxkj.cn/v1"


def _api_key() -> str:
    key = (
        os.environ.get("GPT54_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_PROXY_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "请设置 GPT54_OPENAI_API_KEY / OPENAI_API_KEY，"
            "或复用 GEMINI_PROXY_API_KEY / GEMINI_API_KEY。"
        )
    return key


def _openai_base_url() -> str:
    return (
        os.environ.get("GPT54_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("GEMINI_PROXY_OPENAI_BASE")
        or _DEFAULT_OPENAI_BASE_URL
    )

MODEL = "gpt-5.4"

USER_PROMPT = "用三句话向小朋友解释：为什么天空是蓝色的。"


def build_client() -> OpenAI:
    _fix_ssl_cert_env()
    return OpenAI(api_key=_api_key(), base_url=_openai_base_url())


def chat_gpt54(
    user_text: str,
    *,
    system: str | None = None,
    user: str = "api_call_gpt54",
) -> str:
    """
    纯文本入、纯文本出。底层与 api_call_qwen 相同，走 OpenAI 兼容 chat.completions 流式接口。
    """
    client = build_client()
    messages: List[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})

    request_data = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.time()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
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
    print(f"模型: {MODEL}")
    print(f"用户输入: {USER_PROMPT!r}")
    print("---")
    out = chat_gpt54(USER_PROMPT)
    print("回复:\n", out)


if __name__ == "__main__":
    main()
