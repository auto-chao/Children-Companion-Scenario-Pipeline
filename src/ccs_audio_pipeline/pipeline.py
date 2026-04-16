from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import networkx as nx
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from ccs_audio_pipeline.asset_config import (
    CHILD_MODEL_DIR,
    CLEARVOICE_MODEL_DIR,
    CLEARVOICE_SOURCE_DIR,
    DEMUCS_MODEL_DIR,
    FIRERED_MODEL_DIR,
    FIRERED_SOURCE_DIR,
    PYANNOTE_DIR,
    SEMANTIC_MODEL_DIR,
    format_missing_assets,
    missing_assets,
)
from ccs_audio_pipeline.dialogue_frontend import DialogueFrontend, summarize_audio_view
from ccs_audio_pipeline.gpu_runtime import configure_cuda_backends, resolve_torch_device
from ccs_audio_pipeline.turn_extraction import extract_child_query_turns

DEFAULT_SAMPLE_RATE = 16000


@dataclass
class Segment:
    segment_id: str
    audio_id: str
    start: float
    end: float
    clip_path: Path
    transcript: str
    p_child: float
    speaker_label: str
    sem_embedding: np.ndarray


@dataclass(frozen=True)
class TracePaths:
    """Trace filenames follow pipeline step order (00 → … → summary)."""

    root: Path
    input_files_jsonl: Path
    frontend_views_jsonl: Path
    candidate_turns_jsonl: Path
    range_cleanup_jsonl: Path
    diarization_projection_jsonl: Path
    diarization_rttm_dir: Path
    child_scores_jsonl: Path
    asr_segments_jsonl: Path
    link_scores_jsonl: Path
    dialogs_jsonl: Path
    summary_json: Path


def set_deterministic(seed: int, num_threads: int, *, gpu_fast: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_cuda_backends(gpu_fast=gpu_fast)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, min(4, num_threads)))


class FireRedASR:
    """Strict FireRedASR adapter. No fallback."""

    def __init__(self, source_dir: Path, model_dir: Path, *, use_gpu: bool | None = None) -> None:
        self.source_dir = source_dir
        self.model_dir = model_dir
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self._use_gpu_flag = 1 if use_gpu else 0
        self._bootstrap_import_path()
        try:
            mod = importlib.import_module("fireredasr.models.fireredasr")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cannot import 'fireredasr'. Download the official FireRedASR repo and place it "
                "under 'FireRedASR/', 'vendor/FireRedASR/', or 'third_party/FireRedASR/' in the "
                "project root, or otherwise add it to PYTHONPATH before running the pipeline."
            ) from exc
        if not hasattr(mod, "FireRedAsr"):
            raise RuntimeError("Official FireRedASR source is present, but FireRedAsr class is missing.")
        self.model = mod.FireRedAsr.from_pretrained("aed", str(self.model_dir))

    def transcribe(self, clip_wave: np.ndarray, sr: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(tmp_path, clip_wave, sr, subtype="PCM_16")
            results = self.model.transcribe(
                ["segment_0000"],
                [str(tmp_path)],
                {
                    "use_gpu": self._use_gpu_flag,
                    "beam_size": 3,
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.0,
                    "aed_length_penalty": 0.0,
                    "eos_penalty": 1.0,
                },
            )
            return self._extract_text(results)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _bootstrap_import_path(self) -> None:
        if not self.source_dir.exists():
            raise RuntimeError(format_missing_assets())
        source_dir_str = str(self.source_dir)
        if source_dir_str not in sys.path:
            sys.path.insert(0, source_dir_str)

        path_entries = [
            str(self.source_dir / "fireredasr"),
            str(self.source_dir / "fireredasr" / "utils"),
        ]
        current_path = os.environ.get("PATH", "")
        current_parts = current_path.split(os.pathsep) if current_path else []
        for entry in reversed(path_entries):
            if entry not in current_parts:
                current_parts.insert(0, entry)
        os.environ["PATH"] = os.pathsep.join(current_parts)

    @staticmethod
    def _extract_text(results: Any) -> str:
        def normalize(value: Any) -> str:
            return " ".join(str(value).split())

        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                for key in ("text", "pred_txt", "transcript", "transcription"):
                    if key in first and first[key]:
                        return normalize(first[key])
                for key in ("nbest", "hyps", "hypotheses", "result"):
                    if key in first and first[key]:
                        value = first[key][0]
                        if isinstance(value, dict):
                            for nested_key in ("text", "pred_txt", "transcript", "transcription"):
                                if nested_key in value and value[nested_key]:
                                    return normalize(value[nested_key])
                        return normalize(value)
            if isinstance(first, (list, tuple)) and first:
                return normalize(first[-1])
            return normalize(first)
        if isinstance(results, dict):
            for key in ("text", "pred_txt", "transcript", "transcription"):
                if key in results and results[key]:
                    return normalize(results[key])
        return normalize(results)


class ModelHead(nn.Module):
    def __init__(self, config: Any, num_labels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


class ChildVoiceDetector:
    """Strict child-probability model. No heuristic fallback."""

    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self.device = device
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*clean_up_tokenization_spaces.*",
                category=FutureWarning,
            )
            self.processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Some weights of AgeGenderModel were not initialized.*",
                category=UserWarning,
            )
            self.model = AgeGenderModel.from_pretrained(
                str(model_dir),
                local_files_only=True,
            )
        self.model.to(self.device)
        self.model.eval()
        label2id = getattr(self.model.config, "label2id", {})
        if "child" not in label2id:
            raise RuntimeError("Child label is missing in the age-gender model config.")
        self.child_label_index = int(label2id["child"])

    def predict_probability(self, clip_wave: np.ndarray, sr: int) -> float:
        inputs = self.processor(
            clip_wave.astype(np.float32),
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        with torch.no_grad():
            _, _, gender_probs = self.model(input_values)
        return float(gender_probs[0, self.child_label_index].detach().cpu().clamp(0.0, 1.0).item())


def run_pipeline() -> None:
    parser = argparse.ArgumentParser(
        description="Single-path reproducible SOTA pipeline for child speech dataset generation."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/child_dataset"))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--min-turn-sec", type=float, default=0.35)
    parser.add_argument("--max-turn-sec", type=float, default=20.0)
    parser.add_argument("--turn-merge-gap-sec", type=float, default=0.35)
    parser.add_argument("--turn-glitch-max-sec", type=float, default=0.25)
    parser.add_argument("--turn-glitch-gap-sec", type=float, default=0.2)
    parser.add_argument("--child-threshold", type=float, default=0.6)
    parser.add_argument("--max-gap-seconds", type=float, default=30.0)
    parser.add_argument("--multi-link-threshold", type=float, default=0.7)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--gap-asr-min-sec",
        type=float,
        default=0.35,
        help="儿童片段之间或片尾留白达到该时长才做大人语音 ASR（秒）。",
    )
    parser.add_argument(
        "--gap-asr-max-sec",
        type=float,
        default=45.0,
        help="相邻两轮儿童语音之间、单次大人 ASR 的最长截取（秒）。",
    )
    parser.add_argument(
        "--suffix-asr-max-sec",
        type=float,
        default=45.0,
        help="最后一轮儿童之后、单次大人 ASR 的最长截取（秒）。",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Optional directory used to preserve each pipeline stage output for backtracking.",
    )
    parser.add_argument(
        "--no-gpu-fast",
        action="store_true",
        help="Disable CUDA throughput tweaks (no pyannote on GPU, no cuDNN autotune/TF32). For debugging.",
    )
    args = parser.parse_args()

    missing = missing_assets()
    if missing:
        raise RuntimeError(format_missing_assets())

    device = resolve_torch_device()
    gpu_fast = not args.no_gpu_fast
    set_deterministic(seed=args.seed, num_threads=args.num_threads, gpu_fast=gpu_fast)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_paths = initialize_trace_dir(args.trace_dir) if args.trace_dir else None
    manifest_path = args.output_dir / "manifest.jsonl"
    audios_dir = args.output_dir / "audios"
    audios_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(args.input_dir.glob("*.m4a"), key=lambda p: p.name)
    if not audio_files:
        raise FileNotFoundError(f"No m4a files found in {args.input_dir}")

    summary: dict[str, Any] = {
        "input_files": len(audio_files),
        "candidate_turns": 0,
        "cleaned_candidate_ranges": 0,
        "child_kept_segments": 0,
        "dialogs": 0,
        "output_dir": args.output_dir,
        "trace_dir": trace_paths.root if trace_paths else None,
    }

    ffmpeg_binary = resolve_ffmpeg_binary()
    dialogue_frontend = DialogueFrontend(
        demucs_model_repo=DEMUCS_MODEL_DIR,
        clearvoice_source_dir=CLEARVOICE_SOURCE_DIR,
        clearvoice_checkpoint_dir=CLEARVOICE_MODEL_DIR,
        ffmpeg_binary=ffmpeg_binary,
        device=device,
    )
    diarization_pipeline = load_pyannote_pipeline(PYANNOTE_DIR, device, gpu_fast=gpu_fast)
    child_detector = ChildVoiceDetector(CHILD_MODEL_DIR, device=device)
    asr = FireRedASR(FIRERED_SOURCE_DIR, FIRERED_MODEL_DIR, use_gpu=device.type == "cuda")
    semantic_encoder = SentenceTransformer(
        str(SEMANTIC_MODEL_DIR),
        device=str(device),
        local_files_only=True,
    )

    all_segments: list[Segment] = []
    audio_waveforms: dict[str, tuple[np.ndarray, int]] = {}
    for audio_path in tqdm(audio_files, desc="Processing files"):
        audio_id = audio_path.stem
        waveform, sr = load_audio_mono(audio_path, args.sample_rate)
        audio_waveforms[audio_id] = (waveform, sr)

        foreground_view = dialogue_frontend.build_foreground_dialogue_view(audio_path)
        turn_waveform = foreground_view.enhanced_view.waveform
        turn_sr = foreground_view.enhanced_view.sample_rate
        if turn_sr != args.sample_rate:
            turn_waveform = librosa.resample(
                turn_waveform,
                orig_sr=turn_sr,
                target_sr=args.sample_rate,
            )
            turn_sr = args.sample_rate

        diarization_annotation, raw_turns, candidate_turns, cleanup_records = extract_child_query_turns(
            pipeline=diarization_pipeline,
            waveform=turn_waveform,
            sr=turn_sr,
            min_turn_sec=args.min_turn_sec,
            max_turn_sec=args.max_turn_sec,
            merge_gap_sec=args.turn_merge_gap_sec,
            glitch_max_sec=args.turn_glitch_max_sec,
            glitch_gap_sec=args.turn_glitch_gap_sec,
        )
        summary["candidate_turns"] += len(raw_turns)
        summary["cleaned_candidate_ranges"] += len(candidate_turns)

        if trace_paths:
            append_jsonl(
                trace_paths.input_files_jsonl,
                {
                    "audio_id": audio_id,
                    "cleaned_candidate_ranges": len(candidate_turns),
                    "duration_seconds": len(waveform) / sr if sr else 0.0,
                    "frontend_candidate_turns": len(raw_turns),
                    "num_samples": len(waveform),
                    "sample_rate": sr,
                    "source_audio": audio_path,
                },
            )
            for record in (
                summarize_audio_view("raw_audio", waveform, sr),
                summarize_audio_view(
                    foreground_view.vocals_view.name,
                    foreground_view.vocals_view.waveform,
                    foreground_view.vocals_view.sample_rate,
                ),
                summarize_audio_view(
                    foreground_view.enhanced_view.name,
                    foreground_view.enhanced_view.waveform,
                    foreground_view.enhanced_view.sample_rate,
                ),
            ):
                append_jsonl(
                    trace_paths.frontend_views_jsonl,
                    {
                        "audio_id": audio_id,
                        "source_audio": audio_path,
                        **record,
                    },
                )
            write_diarization_rttm(
                trace_paths.diarization_rttm_dir / f"{audio_id}.rttm",
                diarization_annotation,
            )
            for idx, turn in enumerate(raw_turns):
                append_jsonl(
                    trace_paths.candidate_turns_jsonl,
                    {
                        "audio_id": audio_id,
                        "candidate_index": idx,
                        "candidate_kind": "pyannote_turn",
                        "segment_id": build_segment_id(audio_id, idx, turn.start, turn.end),
                        "source_audio": audio_path,
                        **turn.to_record(),
                    },
                )
            for record in cleanup_records:
                append_jsonl(
                    trace_paths.range_cleanup_jsonl,
                    {
                        "audio_id": audio_id,
                        "source_audio": audio_path,
                        **record,
                    },
                )
            for idx, turn in enumerate(candidate_turns):
                append_jsonl(
                    trace_paths.diarization_projection_jsonl,
                    {
                        "audio_id": audio_id,
                        "duration_seconds": turn.duration,
                        "end": turn.end,
                        "segment_id": build_segment_id(audio_id, idx, turn.start, turn.end),
                        "source_audio": audio_path,
                        "speaker_label": turn.speaker_label,
                        "start": turn.start,
                    },
                )

        if not candidate_turns:
            continue

        for idx, turn in enumerate(candidate_turns):
            start = max(0.0, turn.start)
            end = max(start, turn.end)
            segment_id = build_segment_id(audio_id, idx, start, end)
            s = max(0, int(start * sr))
            e = min(len(waveform), int(end * sr))
            clip_wave = waveform[s:e]
            if len(clip_wave) == 0:
                continue

            speaker_label = turn.speaker_label
            p_child = child_detector.predict_probability(clip_wave, sr)
            passed = p_child >= args.child_threshold
            if trace_paths:
                append_jsonl(
                    trace_paths.child_scores_jsonl,
                    {
                        "audio_id": audio_id,
                        "child_threshold": args.child_threshold,
                        "duration_seconds": end - start,
                        "end": end,
                        "kept": passed,
                        "p_child": p_child,
                        "segment_id": segment_id,
                        "source_audio": audio_path,
                        "speaker_label": speaker_label,
                        "start": start,
                    },
                )
            if not passed:
                continue

            transcript = asr.transcribe(clip_wave, sr)
            sem_embedding = semantic_encoder.encode(
                [transcript if transcript else "[empty]"],
                normalize_embeddings=True,
            )[0]

            clip_name = f"{segment_id}.m4a"
            clip_out = audios_dir / clip_name
            cut_audio(audio_path, clip_out, start, end)
            clip_path = Path("audios") / clip_name
            all_segments.append(
                Segment(
                    segment_id=segment_id,
                    audio_id=audio_id,
                    start=start,
                    end=end,
                    clip_path=clip_path,
                    transcript=transcript.strip(),
                    p_child=p_child,
                    speaker_label=speaker_label,
                    sem_embedding=np.asarray(sem_embedding, dtype=np.float32),
                )
            )
            if trace_paths:
                append_jsonl(
                    trace_paths.asr_segments_jsonl,
                    {
                        "audio_id": audio_id,
                        "audio_clip": clip_path,
                        "candidate_source": "foreground_dialogue_view",
                        "end": end,
                        "p_child": p_child,
                        "segment_id": segment_id,
                        "speaker_label": speaker_label,
                        "start": start,
                        "transcript": transcript.strip(),
                    },
                )

    summary["child_kept_segments"] = len(all_segments)
    if not all_segments:
        if trace_paths:
            write_trace_summary(trace_paths, summary)
        raise RuntimeError("No valid child segments after threshold filtering.")

    dialogs = build_dialogs(
        all_segments=all_segments,
        max_gap_seconds=args.max_gap_seconds,
        link_threshold=args.multi_link_threshold,
        max_turns=args.max_turns,
        trace_paths=trace_paths,
    )
    summary["dialogs"] = len(dialogs)
    if trace_paths:
        write_dialog_trace(trace_paths.dialogs_jsonl, dialogs)
        write_trace_summary(trace_paths, summary)
    write_manifest(
        manifest_path,
        dialogs,
        audio_waveforms,
        asr,
        gap_min_sec=args.gap_asr_min_sec,
        gap_max_sec=args.gap_asr_max_sec,
        suffix_max_sec=args.suffix_asr_max_sec,
    )
    print(f"Done: {manifest_path}")
    print(f"Done: {audios_dir}")
    if trace_paths:
        print(f"Done: {trace_paths.root}")


def load_pyannote_pipeline(model_dir: Path, device: torch.device, *, gpu_fast: bool) -> Any:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*torchcodec is not installed correctly.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
            category=UserWarning,
        )
        pyannote_audio = importlib.import_module("pyannote.audio")
        pipeline_cls = getattr(pyannote_audio, "Pipeline")
        pipeline = pipeline_cls.from_pretrained(str(model_dir))
    if gpu_fast and device.type == "cuda":
        pipeline = pipeline.to(device)
    return pipeline


def initialize_trace_dir(root: Path) -> TracePaths:
    root.mkdir(parents=True, exist_ok=True)
    diarization_rttm_dir = root / "05_diarization_rttm"
    if diarization_rttm_dir.exists():
        shutil.rmtree(diarization_rttm_dir)
    diarization_rttm_dir.mkdir(parents=True, exist_ok=True)

    trace_paths = TracePaths(
        root=root,
        input_files_jsonl=root / "00_input_files.jsonl",
        frontend_views_jsonl=root / "01_frontend_views.jsonl",
        candidate_turns_jsonl=root / "02_pyannote_raw_turns.jsonl",
        range_cleanup_jsonl=root / "03_turn_cleanup.jsonl",
        diarization_projection_jsonl=root / "04_diarization_projection.jsonl",
        diarization_rttm_dir=diarization_rttm_dir,
        child_scores_jsonl=root / "06_child_scores.jsonl",
        asr_segments_jsonl=root / "07_asr_segments.jsonl",
        link_scores_jsonl=root / "08_link_scores.jsonl",
        dialogs_jsonl=root / "09_dialogs.jsonl",
        summary_json=root / "summary.json",
    )
    for path in (
        trace_paths.input_files_jsonl,
        trace_paths.frontend_views_jsonl,
        trace_paths.candidate_turns_jsonl,
        trace_paths.range_cleanup_jsonl,
        trace_paths.diarization_projection_jsonl,
        trace_paths.child_scores_jsonl,
        trace_paths.asr_segments_jsonl,
        trace_paths.link_scores_jsonl,
        trace_paths.dialogs_jsonl,
        trace_paths.summary_json,
    ):
        if path.exists():
            path.unlink()
    return trace_paths


def build_segment_id(audio_id: str, idx: int, start: float, end: float) -> str:
    return f"{audio_id}_{idx:04d}_{int(start * 1000)}_{int(end * 1000)}"


def to_json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(v) for v in value]
    return value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")


def write_diarization_rttm(path: Path, diarization: Any) -> None:
    diarization = get_diarization_annotation(diarization)
    with path.open("w", encoding="utf-8") as f:
        diarization.write_rttm(f)


def write_trace_summary(trace_paths: TracePaths, summary: dict[str, Any]) -> None:
    trace_paths.summary_json.write_text(
        json.dumps(to_json_ready(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_dialog_trace(path: Path, dialogs: list[list[Segment]]) -> None:
    for idx, dialog in enumerate(dialogs):
        append_jsonl(
            path,
            {
                "audio_clips": [seg.clip_path for seg in dialog],
                "audio_id": dialog[0].audio_id if dialog else "",
                "dialog_id": f"dialog_{idx:04d}",
                "num_turns": len(dialog),
                "segment_ids": [seg.segment_id for seg in dialog],
                "transcripts": [seg.transcript for seg in dialog],
            },
        )


def get_diarization_annotation(diarization: Any) -> Any:
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        return exclusive
    speaker_diarization = getattr(diarization, "speaker_diarization", None)
    if speaker_diarization is not None:
        return speaker_diarization
    return diarization


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


def load_audio_mono(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PySoundFile failed.*", category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="librosa.core.audio.__audioread_load.*",
                category=FutureWarning,
            )
            return librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception:  # noqa: BLE001
        ffmpeg_binary = resolve_ffmpeg_binary()
        cmd = [
            ffmpeg_binary,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"audio decode failed for {path}:\n{stderr}")
        waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if waveform.size == 0:
            raise RuntimeError(f"audio decode produced empty waveform for {path}")
        return waveform, sample_rate


def resolve_ffmpeg_binary() -> str:
    env_ffmpeg = os.environ.get("FFMPEG_BINARY")
    if env_ffmpeg:
        return env_ffmpeg

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not available. Install system ffmpeg or add `imageio-ffmpeg` to the environment."
        ) from exc

    return imageio_ffmpeg.get_ffmpeg_exe()


def build_dialogs(
    all_segments: list[Segment],
    max_gap_seconds: float,
    link_threshold: float,
    max_turns: int,
    trace_paths: TracePaths | None = None,
) -> list[list[Segment]]:
    by_audio: dict[str, list[Segment]] = {}
    for seg in all_segments:
        by_audio.setdefault(seg.audio_id, []).append(seg)

    dialogs: list[list[Segment]] = []
    for audio_id in sorted(by_audio.keys()):
        segs = sorted(by_audio[audio_id], key=lambda x: (x.start, x.end, x.clip_path.name))
        graph = build_link_graph(
            segs,
            max_gap_seconds=max_gap_seconds,
            link_threshold=link_threshold,
            trace_path=trace_paths.link_scores_jsonl if trace_paths else None,
        )
        components = sorted(nx.connected_components(graph), key=lambda c: min(c))
        for nodes in components:
            chain = sorted([segs[i] for i in nodes], key=lambda x: (x.start, x.end, x.clip_path.name))
            for chunk in chunk_list(chain, max_turns):
                dialogs.append(chunk)
    return dialogs


def build_link_graph(
    segs: list[Segment],
    max_gap_seconds: float,
    link_threshold: float,
    trace_path: Path | None = None,
) -> nx.Graph:
    graph = nx.Graph()
    for i in range(len(segs)):
        graph.add_node(i)

    embeddings = np.stack([s.sem_embedding for s in segs], axis=0) if segs else np.zeros((0, 1))
    sims = cosine_similarity(embeddings) if len(segs) > 1 else np.eye(len(segs))
    for i in range(len(segs) - 1):
        details = link_score_details(segs[i], segs[i + 1], float(sims[i, i + 1]), max_gap_seconds)
        if trace_path:
            append_jsonl(
                trace_path,
                {
                    "audio_id": segs[i].audio_id,
                    "cur_segment_id": segs[i + 1].segment_id,
                    "cur_speaker_label": segs[i + 1].speaker_label,
                    "cur_transcript": segs[i + 1].transcript,
                    "linked": details["score"] >= link_threshold,
                    "link_threshold": link_threshold,
                    "prev_segment_id": segs[i].segment_id,
                    "prev_speaker_label": segs[i].speaker_label,
                    "prev_transcript": segs[i].transcript,
                    **details,
                },
            )
        if details["score"] >= link_threshold:
            graph.add_edge(i, i + 1, score=details["score"])
    return graph


def link_score_details(prev: Segment, cur: Segment, sem_sim: float, max_gap_seconds: float) -> dict[str, Any]:
    gap = max(0.0, cur.start - prev.end)
    s_time = max(0.0, 1.0 - gap / max_gap_seconds)
    s_spk = 1.0 if prev.speaker_label == cur.speaker_label else 0.0
    s_sem = float(np.clip(sem_sim, 0.0, 1.0))
    return {
        "gap_seconds": gap,
        "same_speaker": prev.speaker_label == cur.speaker_label,
        "score": 0.40 * s_time + 0.35 * s_spk + 0.25 * s_sem,
        "semantic_score": s_sem,
        "speaker_score": s_spk,
        "time_score": s_time,
    }


def chunk_list(items: list[Segment], size: int) -> list[list[Segment]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _transcribe_waveform_interval(
    waveform: np.ndarray,
    sr: int,
    t_start: float,
    t_end: float,
    asr: Any,
    *,
    min_sec: float,
    max_sec: float,
) -> str:
    """在 [t_start, t_end] 上切片并 ASR；过短则返回空串；时长超过 max_sec 则截断。"""
    if sr <= 0:
        return ""
    span = max(0.0, t_end - t_start)
    if span < min_sec:
        return ""
    t1 = min(t_end, t_start + max_sec)
    s_idx = max(0, int(t_start * sr))
    e_idx = min(len(waveform), int(t1 * sr))
    if e_idx <= s_idx:
        return ""
    clip = waveform[s_idx:e_idx]
    if len(clip) / sr < min_sec:
        return ""
    return str(asr.transcribe(clip, sr)).strip()


def write_manifest(
    path: Path,
    dialogs: list[list[Segment]],
    audio_waveforms: dict[str, tuple[np.ndarray, int]],
    asr: Any,
    *,
    gap_min_sec: float,
    gap_max_sec: float,
    suffix_max_sec: float,
) -> None:
    """写入 manifest；assistant 槽位为相邻两轮儿童语音之间（或片尾）的大人语音 ASR。"""
    with path.open("w", encoding="utf-8") as f:
        for dialog in dialogs:
            if not dialog:
                continue
            audio_id = dialog[0].audio_id
            cached = audio_waveforms.get(audio_id)
            if cached is None:
                raise RuntimeError(f"Missing waveform cache for audio_id={audio_id!r}")
            waveform, sr = cached
            dur_sec = float(len(waveform)) / float(sr) if sr else 0.0

            line: dict[str, Any] = {"messages": []}

            prefix = _transcribe_waveform_interval(
                waveform,
                sr,
                0.0,
                dialog[0].start,
                asr,
                min_sec=gap_min_sec,
                max_sec=gap_max_sec,
            )
            if prefix:
                line["recording_prefix_adult"] = prefix

            for i, seg in enumerate(dialog, start=1):
                user_key = "user" if i == 1 else f"user_{i}"
                assistant_key = "assistant" if i == 1 else f"assistant_{i}"
                audio_key = "audio" if i == 1 else f"audio_{i}"

                line[user_key] = seg.transcript
                line[audio_key] = str(seg.clip_path).replace("\\", "/")

                if i < len(dialog):
                    nxt = dialog[i]
                    assistant_text = _transcribe_waveform_interval(
                        waveform,
                        sr,
                        seg.end,
                        nxt.start,
                        asr,
                        min_sec=gap_min_sec,
                        max_sec=gap_max_sec,
                    )
                else:
                    assistant_text = _transcribe_waveform_interval(
                        waveform,
                        sr,
                        seg.end,
                        dur_sec,
                        asr,
                        min_sec=gap_min_sec,
                        max_sec=suffix_max_sec,
                    )
                line[assistant_key] = assistant_text
                line["messages"].append({"role": "user", "text": seg.transcript})
                line["messages"].append({"role": "assistant", "text": assistant_text})

            f.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
