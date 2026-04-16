"""FFmpeg 辅助：无 torch/ML 依赖，供 API 分段流水线单独使用。"""
from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path


def _decode_bytes(data: bytes | None) -> str:
    if not data:
        return ""
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def resolve_ffmpeg_binary() -> str:
    env_ffmpeg = __import__("os").environ.get("FFMPEG_BINARY")
    if env_ffmpeg:
        return env_ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not available. Install system ffmpeg or add `imageio-ffmpeg`."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_duration_seconds(path: Path) -> float:
    """返回音频文件时长（秒）。优先 ffprobe，缺失时回退到 ffmpeg 日志解析。"""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            txt = _decode_bytes(proc.stdout).strip()
            try:
                return float(txt)
            except ValueError:
                pass

    ffmpeg = resolve_ffmpeg_binary()
    proc2 = subprocess.run(
        [ffmpeg, "-i", str(path), "-f", "null", "-"],
        capture_output=True,
    )
    text = _decode_bytes(proc2.stderr) + "\n" + _decode_bytes(proc2.stdout)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not m:
        raise RuntimeError("cannot detect audio duration: ffprobe unavailable and ffmpeg parse failed")
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600.0 + mm * 60.0 + ss


def build_segment_id(audio_id: str, idx: int, start: float, end: float) -> str:
    return f"{audio_id}_{idx:04d}_{int(start * 1000)}_{int(end * 1000)}"


def cut_audio(src: Path, dst: Path, start: float, end: float) -> None:
    duration = max(0.05, end - start)
    ffmpeg_binary = resolve_ffmpeg_binary()
    cmd = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(src),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg cut failed:\n{stderr}")
