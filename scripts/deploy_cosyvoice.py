#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 CosyVoice 代码克隆到 artifacts，并从 Hugging Face 拉取 Fun-CosyVoice3 权重。
可选：创建独立 venv 并安装 CosyVoice/requirements.txt（与主项目依赖隔离）。

用法（仓库根目录）::
    conda activate ccs
    python scripts/deploy_cosyvoice.py
    python scripts/deploy_cosyvoice.py --skip-venv

说明：请用 **ccs**（或 Python 3.10/3.11）运行本脚本创建 ``artifacts/cosyvoice/.venv``；
若用 base 的 Python 3.13 创建 venv，CosyVoice 的 requirements 中部分包无预编译 wheel。

环境变量：HF_TOKEN（若模型需授权）
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[2]


ROOT = _repo_root()
DEFAULT_COSYVOICE_GIT = "https://github.com/FunAudioLLM/CosyVoice.git"
HF_MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
LOCAL_MODEL_NAME = "Fun-CosyVoice3-0.5B"


def main() -> int:
    ap = argparse.ArgumentParser(description="部署 CosyVoice3 到 artifacts/cosyvoice")
    ap.add_argument(
        "--git-url",
        type=str,
        default=os.environ.get("COSYVOICE_GIT_URL", DEFAULT_COSYVOICE_GIT),
        help="CosyVoice 仓库 URL（也可用环境变量 COSYVOICE_GIT_URL，便于镜像站）",
    )
    ap.add_argument(
        "--clone-retries",
        type=int,
        default=3,
        help="git clone 失败时的重试次数",
    )
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "artifacts" / "cosyvoice",
        help="部署根目录（默认 artifacts/cosyvoice）",
    )
    ap.add_argument("--skip-clone", action="store_true", help="若已有 CosyVoice 目录则跳过 git clone")
    ap.add_argument("--skip-download", action="store_true", help="跳过 Hugging Face 权重下载")
    ap.add_argument("--skip-venv", action="store_true", help="不创建 venv / 不 pip install")
    ap.add_argument(
        "--skip-torch-upgrade",
        action="store_true",
        help="跳过在 CosyVoice venv 中升级 torch/torchaudio（默认会升级以提升新显卡兼容性）",
    )
    ap.add_argument(
        "--torch-version",
        type=str,
        default=os.environ.get("COSYVOICE_TORCH_VERSION", "2.8.0"),
        help="CosyVoice venv 中 torch 版本（默认 2.8.0；可用环境变量 COSYVOICE_TORCH_VERSION）",
    )
    ap.add_argument(
        "--torchaudio-version",
        type=str,
        default=os.environ.get("COSYVOICE_TORCHAUDIO_VERSION", "2.8.0"),
        help="CosyVoice venv 中 torchaudio 版本（默认 2.8.0；可用环境变量 COSYVOICE_TORCHAUDIO_VERSION）",
    )
    ap.add_argument(
        "--torch-index-url",
        type=str,
        default=os.environ.get("COSYVOICE_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu128"),
        help="torch wheel 源（默认 cu128；可用环境变量 COSYVOICE_TORCH_INDEX_URL）",
    )
    ap.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()

    art = args.artifacts_dir.resolve()
    art.mkdir(parents=True, exist_ok=True)
    cv_dir = art / "CosyVoice"
    model_dir = cv_dir / "pretrained_models" / LOCAL_MODEL_NAME

    if not args.skip_clone:
        if cv_dir.is_dir() and (cv_dir / ".git").is_dir():
            print(f"已存在 {cv_dir}，拉取更新…")
            subprocess.run(
                ["git", "-C", str(cv_dir), "pull", "--ff-only"],
                check=False,
            )
            subprocess.run(
                ["git", "-C", str(cv_dir), "submodule", "update", "--init", "--recursive"],
                check=False,
            )
        else:
            git_url = args.git_url
            last_err: Exception | None = None
            for attempt in range(max(1, args.clone_retries)):
                try:
                    print(f"克隆 {git_url} -> {cv_dir}（尝试 {attempt + 1}/{args.clone_retries}）")
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "--recursive", git_url, str(cv_dir)],
                        check=True,
                    )
                    last_err = None
                    break
                except subprocess.CalledProcessError as e:
                    last_err = e
                    if cv_dir.is_dir():
                        shutil.rmtree(cv_dir, ignore_errors=True)
            if last_err is not None:
                raise last_err
    else:
        if not cv_dir.is_dir():
            print(f"缺少 {cv_dir}，勿使用 --skip-clone", file=sys.stderr)
            return 1

    if not args.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(
                "请先在当前 conda 环境中安装: pip install huggingface_hub",
                file=sys.stderr,
            )
            return 1
        token = args.hf_token or None
        print(f"下载权重 {HF_MODEL} -> {model_dir}（体积较大，请耐心等待）")
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            HF_MODEL,
            local_dir=str(model_dir),
            token=token,
        )
        print(f"权重已就绪: {model_dir}")

    ref_wav = cv_dir / "asset" / "zero_shot_prompt.wav"
    if not ref_wav.is_file():
        print(f"警告: 未找到参考音频 {ref_wav}（请确认 submodule asset 已拉取）", file=sys.stderr)

    venv_dir = art / ".venv"
    if not args.skip_venv:
        if sys.platform == "win32":
            py = venv_dir / "Scripts" / "python.exe"
        else:
            py = venv_dir / "bin" / "python"
        if not py.is_file():
            print(f"创建 venv（解释器: {sys.executable}）: {venv_dir}")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        req = cv_dir / "requirements.txt"
        if req.is_file():
            print("安装 CosyVoice 依赖（可能与主项目版本不同，故使用独立 venv）…")
            subprocess.run(
                [str(py), "-m", "pip", "install", "-U", "pip", "wheel"],
                check=True,
            )
            subprocess.run(
                [str(py), "-m", "pip", "install", "setuptools"],
                check=True,
            )
            # Windows：openai-whisper 等在隔离构建环境中缺 pkg_resources，需 --no-build-isolation
            subprocess.run(
                [
                    str(py),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(req),
                    "--no-build-isolation",
                ],
                check=False,
            )
        if not args.skip_torch_upgrade:
            print(
                "升级 CosyVoice venv 中的 torch/torchaudio（用于新显卡架构兼容）…"
            )
            subprocess.run(
                [
                    str(py),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    f"torch=={args.torch_version}",
                    f"torchaudio=={args.torchaudio_version}",
                    "--index-url",
                    args.torch_index_url,
                ],
                check=True,
            )
        print(f"\n请使用以下 Python 运行 TTS 脚本:\n  {py}\n")
        print("示例:")
        print(
            f'  "{py}" scripts/tts/batch_cosyvoice_tts.py '
            f"--cosyvoice-root {cv_dir} --model-dir {model_dir} "
            f'--reference-audio "{ref_wav}" --prompt-text "希望你以后能够做的比我还好呦。" '
            "--offset 1 --limit 1"
        )
    else:
        print("\n已跳过 venv。请自行在能 import cosyvoice 的环境中运行 batch_cosyvoice_tts.py。\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
