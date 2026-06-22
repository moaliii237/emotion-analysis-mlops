"""
Per-character emotion analysis (Card #2570 nice-to-have).

When a CSV transcript includes a Speaker column (e.g. MELD format), we can
break the emotion distribution down by character. This module computes:

    - For each speaker: emotion counts, dominant emotion, line count
    - For each (speaker, emotion): the top-N most emotionally intense lines
      ranked by classifier confidence.

The /predict/pipeline endpoint calls these helpers when the dataframe has
a 'Speaker' column.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def detect_speaker_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find a 'Speaker' / 'Character' / 'Name' column case-insensitively.

    Returns:
        The column's actual name in the dataframe, or None if not present.
    """
    candidates = {"speaker", "character", "name", "actor"}
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None


def per_character_summary(df: pd.DataFrame, speaker_col: str) -> dict:
    """
    Build per-character emotion stats.

    Args:
        df: DataFrame with at least Speaker (or equivalent), Emotion,
            and Confidence columns.
        speaker_col: Name of the column holding speaker / character labels.

    Returns:
        dict keyed by speaker name. Each value:
            {
                "line_count": int,
                "dominant_emotion": str,
                "emotion_counts": {emotion: count, ...},
                "avg_confidence": float,
            }
    """
    if speaker_col not in df.columns or "Emotion" not in df.columns:
        return {}

    out: dict[str, dict] = {}
    grouped = df.groupby(speaker_col, dropna=True)
    for speaker, sub in grouped:
        speaker_name = str(speaker).strip()
        if not speaker_name:
            continue
        emotion_counts = sub["Emotion"].value_counts().to_dict()
        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else ""
        avg_conf = (
            float(sub["Confidence"].mean())
            if "Confidence" in sub.columns and len(sub) > 0
            else 0.0
        )
        out[speaker_name] = {
            "line_count": int(len(sub)),
            "dominant_emotion": dominant,
            "emotion_counts": {k: int(v) for k, v in emotion_counts.items()},
            "avg_confidence": round(avg_conf, 4),
        }
    return out


def per_character_top_moments(df: pd.DataFrame, speaker_col: str, n: int = 2) -> dict:
    """
    For each (speaker, emotion), surface the top-N highest-confidence lines.

    Args:
        df: DataFrame with Speaker (or equivalent), Emotion, Confidence,
            Sentence, and optional Start/End columns.
        speaker_col: Name of the speaker column.
        n: How many top lines per (speaker, emotion) pair to keep.

    Returns:
        Nested dict: {speaker: {emotion: [ {Start, End, Sentence, Confidence}, ... ]}}
    """
    if speaker_col not in df.columns or "Emotion" not in df.columns:
        return {}

    cols = [c for c in ["Start", "End", "Sentence", "Confidence"] if c in df.columns]
    out: dict[str, dict] = {}
    for speaker, sub in df.groupby(speaker_col, dropna=True):
        speaker_name = str(speaker).strip()
        if not speaker_name:
            continue
        per_emotion: dict[str, list[dict]] = {}
        for emotion, esub in sub.groupby("Emotion"):
            if "Confidence" in esub.columns:
                top = esub.sort_values("Confidence", ascending=False).head(n)
            else:
                top = esub.head(n)
            per_emotion[str(emotion)] = top[cols].to_dict(orient="records")
        out[speaker_name] = per_emotion
    return out
