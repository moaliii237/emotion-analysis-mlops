"""
Speaker diarization for EmotionScope.

Uses pyannote.audio's pretrained `speaker-diarization-3.1` pipeline to
identify WHO is speaking WHEN in an audio file. We then align Whisper's
transcribed segments to the diarization timeline so each line gets a
Speaker label, which the per-character analysis can use.

Setup (one-time):
    1. Get a HuggingFace token: https://huggingface.co/settings/tokens
       (read-only is enough). Set as HF_TOKEN env var.
    2. Accept the model licenses (click "Agree" once on each page):
       - https://huggingface.co/pyannote/speaker-diarization-3.1
       - https://huggingface.co/pyannote/segmentation-3.0

Runtime cost:
    - Model load: ~500 MB RAM, ~10 s on first call
    - Inference : ~5-10 s per minute of audio on CPU (much faster on GPU)
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_MODEL_NAME = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

# Lazy singletons, populated on first use.
_pipeline = None
_pipeline_lock = Lock()


def _get_pipeline():
    """Load the diarization pipeline once, cache it (thread-safe)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN not set. Speaker diarization needs a HuggingFace "
                "token. Get one at https://huggingface.co/settings/tokens, "
                "accept the licenses for pyannote/speaker-diarization-3.1 "
                "and pyannote/segmentation-3.0, then run: "
                'setx HF_TOKEN "hf_..." (Windows).'
            )

        logger.info("[Diarization] Loading model: %s", _MODEL_NAME)
        # Local import, keep the cost off the API startup path
        from pyannote.audio import Pipeline

        _pipeline = Pipeline.from_pretrained(_MODEL_NAME, use_auth_token=token)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            _pipeline.to(device)
        except Exception as e:  # newer pyannote returns self, older raises
            logger.debug("[Diarization] pipeline.to() not supported: %s", e)
        logger.info("[Diarization] Model ready on %s", device)
    return _pipeline


def diarize(audio_path: str, num_speakers: Optional[int] = None) -> list[dict]:
    """
    Run speaker diarization on a single audio file.

    Args:
        audio_path: Path to a WAV file (16kHz mono recommended).
        num_speakers: Hint how many speakers there are. If None, the pipeline
            figures it out automatically.

    Returns:
        List of {"start": float, "end": float, "speaker": str} dicts,
        sorted by start time. Speaker labels look like "SPEAKER_00",
        "SPEAKER_01", etc.

    Raises:
        RuntimeError: If HF_TOKEN is not set.
    """
    pipeline = _get_pipeline()

    logger.info("[Diarization] Running on: %s", audio_path)
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(audio_path, **kwargs)

    out: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        out.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    out.sort(key=lambda x: x["start"])
    logger.info(
        "[Diarization] Found %d turns across %d speakers",
        len(out),
        len({t["speaker"] for t in out}),
    )
    return out


def assign_speakers_to_segments(
    whisper_segments: list, diarization: list[dict]
) -> list[str]:
    """
    Align Whisper transcription segments with diarization turns.

    For each Whisper segment, find the diarization turn that has the
    most overlap and use its speaker label. If no turn overlaps (silence,
    music, etc.) fall back to "UNKNOWN".

    Args:
        whisper_segments: Whisper's "segments" list, each item must have
                          "start" and "end" floats (seconds).
        diarization: Output of diarize().

    Returns:
        List of speaker labels (one per whisper_segment), in the same order.
    """
    labels: list[str] = []
    for seg in whisper_segments:
        s_start = float(seg["start"])
        s_end = float(seg["end"])
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn in diarization:
            # Overlap length = positive part of (min(end) - max(start))
            overlap = max(0.0, min(s_end, turn["end"]) - max(s_start, turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
            # Diarization sorted by start, so once a turn starts after the
            # segment ends we can break early.
            if turn["start"] > s_end:
                break
        labels.append(best_speaker)
    return labels


def is_available() -> bool:
    """Quick check the caller can use to decide whether to expose diarization."""
    return bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"))
