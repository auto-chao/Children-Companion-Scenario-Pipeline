"""Centralized CUDA device and throughput-oriented torch settings."""

from __future__ import annotations

import os

import torch


def resolve_torch_device() -> torch.device:
    """Prefer CUDA; honor CUDA_DEVICE (default 0) when multiple GPUs exist."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    idx = int(os.environ.get("CUDA_DEVICE", "0"))
    if 0 <= idx < torch.cuda.device_count():
        torch.cuda.set_device(idx)
    return torch.device("cuda", torch.cuda.current_device())


def configure_cuda_backends(*, gpu_fast: bool) -> None:
    """gpu_fast: enable cuDNN autotune + TF32 matmul (higher throughput, less strict determinism)."""
    if not torch.cuda.is_available():
        return
    if gpu_fast:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
