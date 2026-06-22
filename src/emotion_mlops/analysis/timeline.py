"""
Emotion Timeline Analyzer.

Builds a time-series representation of emotion intensity from pipeline
segments. Output is shaped for Chart.js multi-line charts: each emotion
gets its own line, X-axis is time (seconds), Y-axis is intensity (0..1).

Also detects emotional peaks, moments where an emotion spikes above
its baseline, useful for sahne (scene) identification and trailer
highlight extraction (next sprint).

Author: Ceren (Sprint 4)
"""

from __future__ import annotations

from typing import Any, Dict, List

_EMOTION_LABELS = [
    "happiness",
    "neutral",
    "anger",
    "sadness",
    "surprise",
    "fear",
    "disgust",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _parse_time(time_str: Any) -> float:
    """
    Parse a time string like 'HH:MM:SS' or 'HH:MM:SS,ms' to seconds (float).

    Returns 0.0 on any parse failure so the timeline stays usable even
    with malformed segments.
    """
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


def _smooth(values: List[float], window: int = 3) -> List[float]:
    """
    Simple moving average. Window=1 = no smoothing.

    Pads edges by repeating values so output length matches input.
    """
    if window <= 1 or len(values) < window:
        return list(values)
    smoothed = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(round(sum(values[lo:hi]) / (hi - lo), 4))
    return smoothed


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def build_timeline(
    segments: List[Dict[str, Any]],
    smooth_window: int = 1,
    peak_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Build a Chart.js-friendly timeline from pipeline segments.

    Args:
        segments: List of segment dicts with Start, Sentence, Emotion, Confidence.
        smooth_window: Moving-average window size (1 = no smoothing).
        peak_threshold: Confidence above which to mark a moment as a "peak".

    Returns:
        {
            "times": [0, 4, 8, 12, ...],           # X-axis (seconds)
            "series": {
                "happiness": [0.1, 0.0, 0.2, ...],  # one row per emotion
                "anger": [0.9, 0.0, 0.0, ...],
                ...
            },
            "labels":   ["anger", "happiness", ...],  # dominant emotion per slot
            "sentences":["...", "...", ...],          # text per slot (for hover)
            "peaks": [
                {"time": 8, "emotion": "anger", "score": 0.94, "sentence": "..."}
            ],
            "duration": 23.0,
            "segment_count": 6,
        }
    """
    if not segments:
        return {
            "times": [],
            "series": {label: [] for label in _EMOTION_LABELS},
            "labels": [],
            "sentences": [],
            "peaks": [],
            "duration": 0.0,
            "segment_count": 0,
        }

    times: List[float] = []
    labels: List[str] = []
    confidences: List[float] = []
    sentences: List[str] = []
    raw_series: Dict[str, List[float]] = {label: [] for label in _EMOTION_LABELS}

    for seg in segments:
        t = _parse_time(seg.get("Start") or seg.get("start") or 0)
        label = seg.get("Emotion") or seg.get("emotion") or ""
        conf = float(seg.get("Confidence") or seg.get("confidence") or 0.0)
        sent = (seg.get("Sentence") or seg.get("sentence") or "").strip()

        times.append(round(t, 2))
        labels.append(label)
        confidences.append(round(conf, 4))
        sentences.append(sent)

        # One-hot: only the predicted emotion gets the confidence,
        # everything else is 0. This makes the multi-line chart show
        # clearly which emotion dominates each moment.
        for emo in _EMOTION_LABELS:
            raw_series[emo].append(conf if emo == label else 0.0)

    # Optional smoothing (only if requested)
    if smooth_window > 1:
        series = {emo: _smooth(vals, smooth_window) for emo, vals in raw_series.items()}
    else:
        series = raw_series

    # Peak detection: confidence above threshold counts as a peak
    peaks = []
    for i, (t, label, conf, sent) in enumerate(
        zip(times, labels, confidences, sentences)
    ):
        if conf >= peak_threshold and label:
            peaks.append(
                {
                    "time": t,
                    "emotion": label,
                    "score": conf,
                    "sentence": sent,
                    "index": i,
                }
            )
    # Highest-confidence peaks first
    peaks.sort(key=lambda p: p["score"], reverse=True)

    duration = times[-1] if times else 0.0

    return {
        "times": times,
        "series": series,
        "labels": labels,
        "sentences": sentences,
        "peaks": peaks,
        "duration": round(duration, 2),
        "segment_count": len(segments),
    }
