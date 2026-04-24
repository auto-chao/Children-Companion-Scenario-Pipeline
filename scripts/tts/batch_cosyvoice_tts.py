#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从助手 JSONL 批量合成 TTS（CosyVoice 本地 / 或 --mock 试跑）。

每轮仅使用 JSON 中的 ``plain_text`` 作为朗读内容：有参考音频时走 CosyVoice **zero-shot**（``inference_zero_shot``），
有 ``--speaker-id`` 时走 SFT。

默认布局（deploy_cosyvoice.py 部署后）::
  artifacts/cosyvoice/CosyVoice          代码
  artifacts/cosyvoice/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B  权重
  artifacts/cosyvoice/CosyVoice/asset/zero_shot_prompt.wav  官方参考音频

环境变量（可选）
  COSYVOICE_ROOT          CosyVoice 仓库根目录
  COSYVOICE_MODEL_DIR     模型目录
  COSYVOICE_REFERENCE_AUDIO / COSYVOICE_PROMPT_TEXT  zero-shot
  COSYVOICE_FORCE_CPU=1   强制 CPU（需在 import torch 前生效；等价于 --cpu）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# 须在加载 CosyVoice / PyTorch 之前隐藏 GPU（如新显卡架构与当前 torch 的 CUDA 构建不兼容）
if "--cpu" in sys.argv or os.environ.get("COSYVOICE_FORCE_CPU", "").strip().lower() in (
    "1",
    "true",
    "yes",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import wave
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[3]


_ROOT = _repo_root()
_DEFAULT_CV_REPO = _ROOT / "artifacts" / "cosyvoice" / "CosyVoice"
_DEFAULT_MODEL_DIR = _DEFAULT_CV_REPO / "pretrained_models" / "Fun-CosyVoice3-0.5B"
_DEFAULT_REF_WAV = _DEFAULT_CV_REPO / "asset" / "zero_shot_prompt.wav"
# CosyVoice3 官方示例要求 prompt 含 <|endofprompt|>，否则 LLM 会 assert 失败
_CV3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
_DEFAULT_PROMPT_ZH = _CV3_PROMPT_PREFIX + "希望你以后能够做的比我还好呦。"
_ZS_SPK_ID = "ccs_batch_zero_shot"


def _normalize_cv3_prompt_text(raw: str) -> str:
    t = raw.strip()
    if "<|endofprompt|>" in t:
        return t
    return _CV3_PROMPT_PREFIX + t


def _save_generator_audio(gen, out_wav: Path, sample_rate: int) -> None:
    """CosyVoice 对长文本会按句多次 yield，需拼接后再落盘。"""
    import torch
    import torchaudio

    chunks: list = []
    for j in gen:
        chunks.append(j["tts_speech"])
    if not chunks:
        raise RuntimeError("CosyVoice returned no audio")
    full = torch.cat(chunks, dim=1)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_wav), full, sample_rate)


def _strip_strong_html(text: str) -> str:
    return re.sub(r"<strong>([^<]*)</strong>", r"\1", text, flags=re.IGNORECASE)


def _strip_cv_tags(text: str) -> str:
    t = _strip_strong_html(text)
    t = re.sub(r"\[breath\]|\[laughter\]|\[sigh\]|\[gasp\]", " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def _write_mock_wav(path: Path, duration_sec: float = 0.3, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)


class _CosyVoiceBackend:
    def __init__(self, cosyvoice_root: Path, model_dir: Path) -> None:
        cr = cosyvoice_root.resolve()
        sys.path.insert(0, str(cr / "third_party" / "Matcha-TTS"))
        sys.path.insert(0, str(cr))
        from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore  # noqa: E402

        self._model = AutoModel(model_dir=str(model_dir))
        self.sample_rate = getattr(self._model, "sample_rate", 24000)
        self._zs_registered = False

    def register_zero_shot(self, prompt_text: str, prompt_wav: Path, spk_id: str = _ZS_SPK_ID) -> None:
        ok = self._model.add_zero_shot_spk(prompt_text, str(prompt_wav), spk_id)
        if ok is not True and ok is not None:
            pass
        self._zs_registered = True

    def zero_shot_once(
        self, tts_text: str, prompt_text: str, prompt_wav: Path, out_wav: Path
    ) -> None:
        gen = self._model.inference_zero_shot(
            tts_text, prompt_text, str(prompt_wav), stream=False
        )
        _save_generator_audio(gen, out_wav, self.sample_rate)

    def zero_shot_cached(self, tts_text: str, out_wav: Path, spk_id: str = _ZS_SPK_ID) -> None:
        gen = self._model.inference_zero_shot(tts_text, "", "", spk_id, stream=False)
        _save_generator_audio(gen, out_wav, self.sample_rate)

    def sft(self, tts_text: str, spk_id: str, out_wav: Path) -> None:
        gen = self._model.inference_sft(tts_text, spk_id, stream=False)
        _save_generator_audio(gen, out_wav, self.sample_rate)

def _load_backend(
    cosyvoice_root: Path | None, model_dir: Path | None
) -> _CosyVoiceBackend:
    root = cosyvoice_root or Path(os.environ.get("COSYVOICE_ROOT", ""))
    if not root.is_dir():
        root = _DEFAULT_CV_REPO
    if not root.is_dir():
        raise RuntimeError(
            "未找到 CosyVoice 仓库。请先运行: python scripts/deploy_cosyvoice.py"
        )
    md = model_dir
    if md is None:
        env_md = os.environ.get("COSYVOICE_MODEL_DIR")
        md = Path(env_md) if env_md else _DEFAULT_MODEL_DIR
    if not md.is_dir():
        raise RuntimeError(f"模型目录不存在: {md}")
    return _CosyVoiceBackend(root, md)


def _cuda_health_check() -> None:
    """在加载 CosyVoice 前执行 GPU 轻量健康检查，给出更清晰的报错。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() == False。请确认 NVIDIA 驱动、CUDA 运行时与"
            " CosyVoice venv 中的 torch 安装。"
        )
    try:
        _ = torch.ones(1, device="cuda") + 1
    except Exception as e:
        raise RuntimeError(
            "CUDA 自检失败。常见原因是显卡架构与 torch wheel 不匹配（例如 RTX 50 系列）。"
            "请重新运行: python scripts/deploy_cosyvoice.py "
            "--skip-clone --skip-download "
            "--torch-index-url https://download.pytorch.org/whl/cu128"
        ) from e


def _plain_text_for_turn(
    turn: dict[str, Any],
    *,
    strip_cv_speech: bool,
) -> str | None:
    raw = turn.get("plain_text")
    if not isinstance(raw, str) or not raw.strip():
        return None
    t = raw.strip()
    if strip_cv_speech:
        t = _strip_cv_tags(t)
    return t.strip() or None


def _process_line(
    row: dict[str, Any],
    *,
    manifest_line: int,
    strip_cv_speech: bool,
    mock: bool,
    backend: _CosyVoiceBackend | None,
    reference_audio: Path | None,
    prompt_text: str | None,
    speaker_id: str | None,
    use_zs_cache: bool,
    max_retries: int,
    retry_sleep: float,
    resume: bool,
) -> dict[str, Any]:
    row = json.loads(json.dumps(row))
    turns = row.get("turns")
    if not isinstance(turns, list):
        return row

    for ti, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        if turn.get("error"):
            continue
        t_idx = int(turn.get("turn_index") or (ti + 1))
        tts_text = _plain_text_for_turn(turn, strip_cv_speech=strip_cv_speech)
        if not tts_text:
            turn["tts_error"] = "skip_no_text"
            continue

        rel_out = f"outputs/tts_generated/m{manifest_line}_t{t_idx}.wav"
        out_path = _ROOT / rel_out.replace("/", os.sep)
        if resume and out_path.is_file():
            turn["tts_audio"] = rel_out
            continue

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                if mock:
                    _write_mock_wav(out_path)
                elif reference_audio and reference_audio.is_file():
                    if backend is None:
                        raise RuntimeError("backend required")
                    pt = prompt_text or ""
                    if not pt.strip():
                        raise RuntimeError(
                            "使用 --reference-audio 时请提供 --prompt-text 或环境变量 "
                            "COSYVOICE_PROMPT_TEXT。"
                        )
                    if use_zs_cache and backend._zs_registered:
                        backend.zero_shot_cached(tts_text, out_path)
                    else:
                        backend.zero_shot_once(
                            tts_text, pt.strip(), reference_audio, out_path
                        )
                elif speaker_id:
                    if backend is None:
                        raise RuntimeError("backend required")
                    backend.sft(tts_text, speaker_id, out_path)
                else:
                    raise RuntimeError(
                        "请指定 --reference-audio + --prompt-text，或 --speaker-id，或使用 --mock"
                    )
                turn["tts_audio"] = rel_out
                turn.pop("tts_error", None)
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep * (2**attempt))
        else:
            turn["tts_error"] = f"{type(last_err).__name__}: {last_err}"

    return row


def main() -> int:
    p = argparse.ArgumentParser(description="批量 CosyVoice TTS，写入 turns[].tts_audio")
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "outputs" / "assistant_responses_multiturn.qc_passed.jsonl",
        help="输入 JSONL（默认与 Stage 2.5 筛子一致；全量见 assistant_responses_multiturn.jsonl）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "outputs" / "assistant_responses_with_tts.jsonl",
        help="输出 JSONL",
    )
    p.add_argument(
        "--tts-dir",
        type=Path,
        default=_ROOT / "outputs" / "tts_generated",
        help="TTS wav 物理目录",
    )
    p.add_argument(
        "--strip-cv-speech-tags",
        action="store_true",
        help="去掉 plain_text 中 [breath] 等副语言标签（默认保留）",
    )
    p.add_argument("--mock", action="store_true", help="只生成短静音 wav")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="强制 CPU 推理（避免不支持的 GPU / CUDA 架构导致崩溃；见文件头 COSYVOICE_FORCE_CPU）",
    )
    p.add_argument(
        "--cosyvoice-root",
        type=Path,
        default=None,
        help=f"CosyVoice 克隆路径（默认 {_DEFAULT_CV_REPO}）",
    )
    p.add_argument("--model-dir", type=Path, default=None, help="pretrained 模型目录")
    p.add_argument("--speaker-id", type=str, default=None, help="SFT 说话人 ID")
    p.add_argument("--reference-audio", type=Path, default=None, help="zero-shot 参考 wav")
    p.add_argument("--prompt-text", type=str, default=None, help="参考音频对应文本")
    p.add_argument(
        "--no-zs-cache",
        action="store_true",
        help="每轮重新完整 zero-shot（慢）；默认注册一次说话人后复用",
    )
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条 manifest 行")
    p.add_argument("--offset", type=int, default=0, help="跳过前 offset 条 manifest 行")
    p.add_argument("--resume", action="store_true", help="跳过已存在 wav 的轮次")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=1.0)
    args = p.parse_args()

    ref = args.reference_audio
    if ref is None:
        env_ref = os.environ.get("COSYVOICE_REFERENCE_AUDIO")
        if env_ref:
            ref = Path(env_ref)
        elif _DEFAULT_REF_WAV.is_file():
            ref = _DEFAULT_REF_WAV

    prompt = args.prompt_text or os.environ.get("COSYVOICE_PROMPT_TEXT")
    if prompt is None and ref is not None and ref.is_file():
        prompt = _DEFAULT_PROMPT_ZH
    if prompt is not None and prompt.strip():
        prompt = _normalize_cv3_prompt_text(prompt)

    inp = args.input
    if not inp.is_file():
        print(f"输入不存在: {inp}", file=sys.stderr)
        return 1

    args.tts_dir.mkdir(parents=True, exist_ok=True)

    backend: _CosyVoiceBackend | None = None
    use_zs_cache = not args.no_zs_cache
    if not args.mock:
        try:
            if not args.cpu:
                _cuda_health_check()
            cr = args.cosyvoice_root or _DEFAULT_CV_REPO
            md = args.model_dir or _DEFAULT_MODEL_DIR
            backend = _load_backend(cr, md)
            if (
                ref
                and ref.is_file()
                and prompt
                and prompt.strip()
                and use_zs_cache
            ):
                backend.register_zero_shot(prompt.strip(), ref, _ZS_SPK_ID)
            try:
                import torch

                if not args.cpu and torch.cuda.is_available():
                    print(
                        "CosyVoice 推理设备: CUDA ("
                        f"{torch.cuda.get_device_name(0)})"
                    )
                else:
                    print("CosyVoice 推理设备: CPU")
            except Exception:
                pass
        except Exception as e:
            print(f"无法加载 CosyVoice: {e}", file=sys.stderr)
            print("提示: python scripts/deploy_cosyvoice.py ；或 --mock", file=sys.stderr)
            return 1

    lines: list[str] = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    off = max(0, args.offset)
    lines = lines[off :]
    if args.limit is not None:
        lines = lines[: args.limit]

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with out_path.open("w", encoding="utf-8") as outf:
        for i, line in enumerate(tqdm(lines, desc="manifest lines")):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ml = off + i + 1
            processed = _process_line(
                row,
                manifest_line=ml,
                strip_cv_speech=args.strip_cv_speech_tags,
                mock=args.mock,
                backend=backend,
                reference_audio=ref,
                prompt_text=prompt,
                speaker_id=args.speaker_id,
                use_zs_cache=use_zs_cache,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                resume=args.resume,
            )
            outf.write(json.dumps(processed, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"完成：写入 {out_path}（行数 {n_ok}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
