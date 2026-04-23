#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "httpx[socks]>=0.27.0",
#   "openai>=1.76.0",
#   "python-dotenv>=1.0.1",
# ]
# ///

import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def getenv_clean(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def build_client() -> OpenAI:
    load_dotenv()
    api_key = getenv_clean("OPENAI_API_KEY")
    base_url = getenv_clean("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("Missing OPENAI_API_KEY or OPENAI_BASE_URL in .env")
    return OpenAI(api_key=api_key, base_url=base_url)


def audio_to_data_uri(audio_path: Path) -> tuple[str, str]:
    suffix = audio_path.suffix.lower().lstrip(".")
    mime_map = {"mp3": "audio/mpeg", "mpga": "audio/mpeg", "wav": "audio/wav",
                "flac": "audio/flac", "ogg": "audio/ogg", "m4a": "audio/mp4"}
    fmt_map = {"mp3": "mp3", "mpga": "mp3", "wav": "wav", "flac": "flac", "ogg": "ogg", "m4a": "m4a"}
    mime = mime_map.get(suffix, "audio/mpeg")
    fmt = fmt_map.get(suffix, "mp3")
    b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}", fmt


def transcribe(audio_path: Path, prompt: str = "请将这段音频的内容转为文字。") -> str:
    client = build_client()
    model = "qwen3.5-omni-plus"
    data_uri, audio_fmt = audio_to_data_uri(audio_path)

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data_uri, "format": audio_fmt},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )
    result = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            result.append(chunk.choices[0].delta.content)
    return "".join(result)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <音频文件路径> [提示词]")
        print(f"示例: {sys.argv[0]} test.wav")
        print(f"示例: {sys.argv[0]} test.mp3 '请总结这段音频的主要内容'")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        raise SystemExit(f"文件不存在: {audio_path}")

    prompt = sys.argv[2] if len(sys.argv) > 2 else "请将这段音频的内容转为文字。"

    print(f"模型: qwen3.5-omni-plus")
    print(f"音频: {audio_path}")
    print(f"提示: {prompt}")
    print("---")

    result = transcribe(audio_path, prompt)
    print(result)


if __name__ == "__main__":
    main()
