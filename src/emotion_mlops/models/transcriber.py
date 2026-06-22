"""
Audio transcription module for the EmotionScope pipeline.

This module uses OpenAI's Whisper model to transcribe audio files into
timestamped text segments.

Example:
    >>> from emotion_mlops.models.transcriber import transcribe_audio
    >>> segments = transcribe_audio("audio.mp3", whisper_model_name="base", device="cpu")
    >>> for seg in segments:
    ...     print(seg.start, seg.end, seg.text)
"""

from dataclasses import dataclass
from typing import List

import whisper


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe_audio(
    audio_path: str, whisper_model_name: str = "base", device: str = "cpu"
) -> List[Segment]:
    """Transcribe audio using Whisper."""
    print(f"[Whisper] Loading model: {whisper_model_name} on {device}")
    model = whisper.load_model(whisper_model_name, device=device)
    print(f"[Whisper] Transcribing: {audio_path}")
    result = model.transcribe(audio_path, verbose=False, language="en")
    segments = [
        Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"].strip())
        for s in result["segments"]
    ]
    print(f"[Whisper] Done, {len(segments)} segments")
    return segments
