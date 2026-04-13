from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

DEMUCS_MODEL_NAME = "htdemucs_ft"
CLEARVOICE_MODEL_NAME = "MossFormer2_SE_48K"
CLEARVOICE_SAMPLE_RATE = 48000


@dataclass(frozen=True)
class AudioView:
    name: str
    waveform: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class ForegroundDialogueView:
    vocals_view: AudioView
    enhanced_view: AudioView


def summarize_audio_view(name: str, waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if waveform.ndim == 1:
        channels = 1
        mono_waveform = waveform
    else:
        channels = int(waveform.shape[0])
        mono_waveform = waveform.mean(axis=0)

    mono_waveform = np.asarray(mono_waveform, dtype=np.float32)
    peak = float(np.max(np.abs(mono_waveform))) if mono_waveform.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono_waveform, dtype=np.float64)))) if mono_waveform.size else 0.0
    return {
        "channels": channels,
        "duration_seconds": float(mono_waveform.shape[-1] / sample_rate) if sample_rate else 0.0,
        "name": name,
        "num_samples": int(mono_waveform.shape[-1]),
        "peak": peak,
        "rms": rms,
        "sample_rate": int(sample_rate),
    }


class DemucsMusicSuppressor:
    def __init__(
        self,
        model_repo: Path,
        ffmpeg_binary: str,
        device: torch.device,
        shifts: int = 2,
        overlap: float = 0.25,
    ) -> None:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        demucs_pretrained = importlib.import_module("demucs.pretrained")
        self._apply_model = getattr(importlib.import_module("demucs.apply"), "apply_model")
        self.model = demucs_pretrained.get_model(DEMUCS_MODEL_NAME, repo=model_repo)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.ffmpeg_binary = ffmpeg_binary
        self.shifts = shifts
        self.overlap = overlap
        self.source_to_index = {str(name): idx for idx, name in enumerate(self.model.sources)}
        if "vocals" not in self.source_to_index:
            raise RuntimeError("Demucs model does not expose a 'vocals' source.")

    def suppress_music(self, audio_path: Path) -> AudioView:
        mixture = self._decode_audio_tensor(
            audio_path,
            sample_rate=int(self.model.samplerate),
            channels=int(self.model.audio_channels),
        )
        ref = mixture.mean(0)
        mean_value = ref.mean()
        scale = ref.std().clamp_min(1e-8)
        mixture = (mixture - mean_value) / scale
        with torch.no_grad():
            sources = self._apply_model(
                self.model,
                mixture.unsqueeze(0),
                device=self.device,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                progress=False,
                num_workers=0,
            )[0]
        sources = sources * scale
        sources = sources + mean_value
        vocals = sources[self.source_to_index["vocals"]].mean(dim=0).detach().cpu().numpy()
        return AudioView(
            name="vocals_or_dialogue_view",
            waveform=_sanitize_waveform(vocals),
            sample_rate=int(self.model.samplerate),
        )

    def _decode_audio_tensor(self, audio_path: Path, sample_rate: int, channels: int) -> torch.Tensor:
        cmd = [
            self.ffmpeg_binary,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(audio_path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Demucs audio decode failed for {audio_path}:\n{stderr}")

        waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if waveform.size == 0:
            raise RuntimeError(f"Demucs audio decode produced empty waveform for {audio_path}.")
        waveform = waveform.reshape(-1, channels).T
        return torch.from_numpy(waveform)


class ClearVoiceEnhancer:
    def __init__(self, source_dir: Path, checkpoint_dir: Path) -> None:
        self._bootstrap_import_path(source_dir)
        network_wrapper_mod = importlib.import_module("clearvoice.network_wrapper")
        clearvoice_networks = importlib.import_module("clearvoice.networks")
        network_wrapper = getattr(network_wrapper_mod, "network_wrapper")
        model_cls = getattr(clearvoice_networks, "CLS_MossFormer2_SE_48K")

        wrapper = network_wrapper()
        wrapper.model_name = CLEARVOICE_MODEL_NAME
        wrapper.load_args_se()
        wrapper.args.task = "speech_enhancement"
        wrapper.args.network = CLEARVOICE_MODEL_NAME
        wrapper.args.checkpoint_dir = str(checkpoint_dir)

        self.model = model_cls(wrapper.args)
        self.sample_rate = int(wrapper.args.sampling_rate)

    def enhance(self, waveform: np.ndarray, sample_rate: int) -> AudioView:
        mono_waveform = np.asarray(waveform, dtype=np.float32)
        if mono_waveform.ndim != 1:
            mono_waveform = mono_waveform.mean(axis=0)
        if sample_rate != self.sample_rate:
            mono_waveform = librosa.resample(
                mono_waveform,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
            )
        batch = np.asarray(mono_waveform, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            enhanced = self.model.decode_data(batch)
        enhanced_waveform = np.asarray(enhanced[0], dtype=np.float32)
        return AudioView(
            name="foreground_dialogue_view",
            waveform=_sanitize_waveform(enhanced_waveform),
            sample_rate=self.sample_rate,
        )

    @staticmethod
    def _bootstrap_import_path(source_dir: Path) -> None:
        if not source_dir.exists():
            raise RuntimeError(f"ClearerVoice source is missing: {source_dir}")
        source_dir_str = str(source_dir)
        if source_dir_str not in sys.path:
            sys.path.insert(0, source_dir_str)


class DialogueFrontend:
    def __init__(
        self,
        demucs_model_repo: Path,
        clearvoice_source_dir: Path,
        clearvoice_checkpoint_dir: Path,
        ffmpeg_binary: str,
        device: torch.device,
    ) -> None:
        self.music_suppressor = DemucsMusicSuppressor(
            model_repo=demucs_model_repo,
            ffmpeg_binary=ffmpeg_binary,
            device=device,
        )
        self.speech_enhancer = ClearVoiceEnhancer(
            source_dir=clearvoice_source_dir,
            checkpoint_dir=clearvoice_checkpoint_dir,
        )

    def build_foreground_dialogue_view(self, audio_path: Path) -> ForegroundDialogueView:
        vocals_view = self.music_suppressor.suppress_music(audio_path)
        enhanced_view = self.speech_enhancer.enhance(
            vocals_view.waveform,
            sample_rate=vocals_view.sample_rate,
        )
        return ForegroundDialogueView(
            vocals_view=vocals_view,
            enhanced_view=enhanced_view,
        )


def _sanitize_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return waveform.copy()
