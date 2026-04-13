from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_label: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_record(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration,
            "end": self.end,
            "speaker_label": self.speaker_label,
            "start": self.start,
        }


def get_diarization_annotation(diarization: Any) -> Any:
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        return exclusive
    speaker_diarization = getattr(diarization, "speaker_diarization", None)
    if speaker_diarization is not None:
        return speaker_diarization
    return diarization


def extract_child_query_turns(
    pipeline: Any,
    waveform: np.ndarray,
    sr: int,
    min_turn_sec: float,
    max_turn_sec: float,
    merge_gap_sec: float,
    glitch_max_sec: float,
    glitch_gap_sec: float,
) -> tuple[Any, list[SpeakerTurn], list[SpeakerTurn], list[dict[str, Any]]]:
    diarization = pipeline(
        {
            "waveform": torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0),
            "sample_rate": sr,
        }
    )
    raw_turns = annotation_to_turns(get_diarization_annotation(diarization))
    cleaned_turns, cleanup_records = cleanup_turns(
        raw_turns,
        min_turn_sec=min_turn_sec,
        max_turn_sec=max_turn_sec,
        merge_gap_sec=merge_gap_sec,
        glitch_max_sec=glitch_max_sec,
        glitch_gap_sec=glitch_gap_sec,
    )
    return diarization, raw_turns, cleaned_turns, cleanup_records


def annotation_to_turns(annotation: Any) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end <= start:
            continue
        turns.append(
            SpeakerTurn(
                start=start,
                end=end,
                speaker_label=str(speaker).replace(" ", "_"),
            )
        )
    turns.sort(key=lambda item: (item.start, item.end, item.speaker_label))
    return turns


def cleanup_turns(
    raw_turns: list[SpeakerTurn],
    min_turn_sec: float,
    max_turn_sec: float,
    merge_gap_sec: float,
    glitch_max_sec: float,
    glitch_gap_sec: float,
) -> tuple[list[SpeakerTurn], list[dict[str, Any]]]:
    cleanup_records: list[dict[str, Any]] = []
    normalized_turns = [turn for turn in raw_turns if turn.duration > 0.0]

    merged_turns: list[SpeakerTurn] = []
    for turn in normalized_turns:
        if (
            merged_turns
            and turn.speaker_label == merged_turns[-1].speaker_label
            and max(0.0, turn.start - merged_turns[-1].end) <= merge_gap_sec
        ):
            previous = merged_turns[-1]
            merged_turns[-1] = SpeakerTurn(
                start=previous.start,
                end=max(previous.end, turn.end),
                speaker_label=previous.speaker_label,
            )
            cleanup_records.append(
                {
                    "action": "merge_same_speaker_gap",
                    "after": merged_turns[-1].to_record(),
                    "before": [previous.to_record(), turn.to_record()],
                    "gap_seconds": max(0.0, turn.start - previous.end),
                    "reason": "same speaker turns separated by a short pause",
                }
            )
        else:
            merged_turns.append(turn)

    bridged_turns = merged_turns[:]
    idx = 0
    while idx + 2 < len(bridged_turns):
        left = bridged_turns[idx]
        middle = bridged_turns[idx + 1]
        right = bridged_turns[idx + 2]
        left_gap = max(0.0, middle.start - left.end)
        right_gap = max(0.0, right.start - middle.end)
        if (
            left.speaker_label == right.speaker_label
            and left.speaker_label != middle.speaker_label
            and middle.duration <= glitch_max_sec
            and left_gap <= glitch_gap_sec
            and right_gap <= glitch_gap_sec
        ):
            merged = SpeakerTurn(
                start=left.start,
                end=right.end,
                speaker_label=left.speaker_label,
            )
            bridged_turns[idx : idx + 3] = [merged]
            cleanup_records.append(
                {
                    "action": "bridge_micro_turn",
                    "after": merged.to_record(),
                    "before": [left.to_record(), middle.to_record(), right.to_record()],
                    "left_gap_seconds": left_gap,
                    "reason": "short opposite-speaker blip between matching speakers",
                    "right_gap_seconds": right_gap,
                }
            )
            if idx > 0:
                idx -= 1
            continue
        idx += 1

    cleaned_turns: list[SpeakerTurn] = []
    for turn in bridged_turns:
        if turn.duration < min_turn_sec:
            cleanup_records.append(
                {
                    "action": "drop_short_turn",
                    "before": [turn.to_record()],
                    "reason": "turn shorter than minimum duration",
                }
            )
            continue

        if turn.duration <= max_turn_sec:
            cleaned_turns.append(turn)
            cleanup_records.append(
                {
                    "action": "keep_turn",
                    "after": turn.to_record(),
                    "reason": "turn kept after cleanup",
                }
            )
            continue

        num_chunks = max(2, int(math.ceil(turn.duration / max_turn_sec)))
        chunk_duration = turn.duration / num_chunks
        for chunk_idx in range(num_chunks):
            chunk_start = turn.start + chunk_idx * chunk_duration
            chunk_end = turn.end if chunk_idx == num_chunks - 1 else turn.start + (chunk_idx + 1) * chunk_duration
            chunk = SpeakerTurn(
                start=chunk_start,
                end=chunk_end,
                speaker_label=turn.speaker_label,
            )
            cleaned_turns.append(chunk)
            cleanup_records.append(
                {
                    "action": "split_long_turn",
                    "after": chunk.to_record(),
                    "before": [turn.to_record()],
                    "chunk_index": chunk_idx,
                    "num_chunks": num_chunks,
                    "reason": "turn longer than maximum duration",
                }
            )

    return cleaned_turns, cleanup_records
