"""
Emotion Shift Detector.

Identifies turning points where the dominant emotion changes
significantly between consecutive segments. Designed to surface the
"narrative beats" in a transcript, the moments where the energy of
the content pivots (e.g., a calm interview suddenly turning hostile,
or a sad scene flipping into comic relief).

Used by the frontend timeline view to annotate inflection points and
by the content score to estimate pacing/drama.

Author: Ceren (Sprint 4, Day 6)
"""

from __future__ import annotations

from typing import Any, Dict, List


def _parse_time(time_str: Any) -> float:
    """Parse 'HH:MM:SS[,ms]' to seconds. Returns 0.0 on failure."""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    if not time_str:
        return 0.0
    try:
        s = str(time_str).strip().replace(",", ".")
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        return float(s)
    except Exception:
        return 0.0


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# How "different" are two emotions? Used to score shift magnitude.
# Higher = more dramatic pivot.
_EMOTION_DISTANCE = {
    # opposites get the highest score
    ("happiness", "sadness"): 1.0,
    ("happiness", "anger"): 0.9,
    ("happiness", "fear"): 0.85,
    ("happiness", "disgust"): 0.9,
    ("sadness", "anger"): 0.6,
    ("sadness", "fear"): 0.5,
    ("sadness", "happiness"): 1.0,
    ("anger", "happiness"): 0.9,
    ("anger", "sadness"): 0.6,
    ("anger", "fear"): 0.4,
    ("fear", "happiness"): 0.85,
    ("fear", "anger"): 0.4,
    ("fear", "surprise"): 0.3,
    ("surprise", "fear"): 0.3,
    ("disgust", "happiness"): 0.9,
    # everything-to-neutral and neutral-to-everything is medium
}


def _shift_magnitude(emo_a: str, emo_b: str) -> float:
    """
    Score how dramatic the shift from emo_a to emo_b is, 0..1.

    Same emotion = 0 (no shift). Opposites (happy vs sad) = ~1.0.
    Neutral transitions are treated as medium (0.4) so they're
    not over-counted.
    """
    if not emo_a or not emo_b:
        return 0.0
    if emo_a == emo_b:
        return 0.0

    # Direct lookup
    if (emo_a, emo_b) in _EMOTION_DISTANCE:
        return _EMOTION_DISTANCE[(emo_a, emo_b)]
    if (emo_b, emo_a) in _EMOTION_DISTANCE:
        return _EMOTION_DISTANCE[(emo_b, emo_a)]

    # Neutral involved -> medium-low
    if "neutral" in (emo_a, emo_b):
        return 0.4

    # Fallback for any other mismatched pair
    return 0.5


def detect_shifts(
    segments: List[Dict[str, Any]],
    min_magnitude: float = 0.5,
    min_confidence: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Find emotional turning points in the transcript.

    Args:
        segments: Pipeline segments with Emotion + Confidence + Start.
        min_magnitude: Only shifts with magnitude >= this are reported
                       (default 0.5 = at least a moderate pivot).
        min_confidence: Both segments must have confidence >= this so
                       we don't surface model-noise shifts (default 0.5).

    Returns:
        List of shift dicts (chronological):
        [
          {
            "shift_id": 1,
            "time":  8.0,
            "label": "0:08",
            "from_emotion": "anger",
            "to_emotion":   "happiness",
            "magnitude":    0.9,
            "from_sentence": "...",
            "to_sentence":   "...",
            "from_score":    0.88,
            "to_score":      0.85,
            "description":   "Anger -> Happiness"
          },
          ...
        ]
    """
    if not segments or len(segments) < 2:
        return []

    shifts = []
    shift_id = 1

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        prev_emo = (prev.get("Emotion") or prev.get("emotion") or "").lower().strip()
        curr_emo = (curr.get("Emotion") or curr.get("emotion") or "").lower().strip()
        if not prev_emo or not curr_emo or prev_emo == curr_emo:
            continue

        prev_conf = float(prev.get("Confidence") or prev.get("confidence") or 0.0)
        curr_conf = float(curr.get("Confidence") or curr.get("confidence") or 0.0)
        if prev_conf < min_confidence or curr_conf < min_confidence:
            continue

        magnitude = _shift_magnitude(prev_emo, curr_emo)
        if magnitude < min_magnitude:
            continue

        t = _parse_time(curr.get("Start") or curr.get("start") or 0)
        shifts.append(
            {
                "shift_id": shift_id,
                "time": round(t, 2),
                "label": _format_time(t),
                "from_emotion": prev_emo,
                "to_emotion": curr_emo,
                "magnitude": round(magnitude, 4),
                "from_sentence": (
                    prev.get("Sentence") or prev.get("sentence") or ""
                ).strip(),
                "to_sentence": (
                    curr.get("Sentence") or curr.get("sentence") or ""
                ).strip(),
                "from_score": round(prev_conf, 4),
                "to_score": round(curr_conf, 4),
                "description": f"{prev_emo.capitalize()} -> {curr_emo.capitalize()}",
            }
        )
        shift_id += 1

    return shifts
