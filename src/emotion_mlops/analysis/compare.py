"""
Comparison Analyzer.

Compares two pipeline analysis results side-by-side. Computes emotion
deltas, similarity, winner badges, common actors, and statistical
summaries. Designed for the Compare frontend page where two pieces of
content (shows, episodes, podcasts, etc.) are analyzed in parallel.

Input shape (per side):
    {
        "segments": [{Start, End, Sentence, Emotion, Confidence}, ...],
        "emotions": [str, ...],          # flat list of labels
        "cast":     [{name, ...}, ...],  # cast_analyzer output
    }

Output:
    {
        "left":   <unchanged input>,
        "right":  <unchanged input>,
        "comparison": {
            "emotion_distribution":  {left: {...}, right: {...}},
            "emotion_deltas":        {anger: +0.15, ...},
            "winners":               {more_intense: "left", ...},
            "common_actors":         ["Trump", ...],
            "actor_overlap":         0.4,    # Jaccard
            "similarity":            0.82,   # cosine on emotion distribution
            "stats":                 {left_avg_conf: ..., right_avg_conf: ...},
        }
    }

Author: Ceren (Sprint 4)
"""

from __future__ import annotations

import math
from collections import Counter
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


def _emotion_distribution(segments: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute relative frequency of each emotion across segments.

    Returns a dict keyed by emotion label, values sum to 1.0.
    """
    if not segments:
        return {label: 0.0 for label in _EMOTION_LABELS}

    labels = [
        seg.get("Emotion") or seg.get("emotion")
        for seg in segments
        if (seg.get("Emotion") or seg.get("emotion"))
    ]
    if not labels:
        return {label: 0.0 for label in _EMOTION_LABELS}

    counts = Counter(labels)
    total = sum(counts.values())

    # Initialize all labels at 0 so output shape is stable
    distribution = {label: 0.0 for label in _EMOTION_LABELS}
    for label, count in counts.items():
        if label in distribution:
            distribution[label] = round(count / total, 4)
    return distribution


def _avg_confidence(segments: List[Dict[str, Any]]) -> float:
    """Mean classifier confidence across segments. 0 if empty."""
    if not segments:
        return 0.0
    confs = [
        float(seg.get("Confidence") or seg.get("confidence") or 0.0) for seg in segments
    ]
    confs = [c for c in confs if c > 0]
    if not confs:
        return 0.0
    return round(sum(confs) / len(confs), 4)


def _entropy(distribution: Dict[str, float]) -> float:
    """
    Shannon entropy of an emotion distribution. Higher = more diverse.

    Normalized to [0, 1] by dividing by log2(N) where N=len(labels).
    """
    values = [v for v in distribution.values() if v > 0]
    if not values:
        return 0.0
    H = -sum(p * math.log2(p) for p in values)
    H_max = math.log2(len(_EMOTION_LABELS))
    return round(H / H_max, 4) if H_max > 0 else 0.0


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Cosine similarity between two emotion distributions.

    Returns 0..1 (1 = identical, 0 = orthogonal).
    """
    keys = set(a.keys()) | set(b.keys())
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return round(dot / (norm_a * norm_b), 4)


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return round(intersection / union, 4) if union > 0 else 0.0


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def compare(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two pipeline analysis results.

    Args:
        left, right: Pipeline result dicts. Each should have:
                     segments, emotions, cast (cast may be empty).

    Returns:
        dict with 'left', 'right' (unchanged) and 'comparison' (new metrics).
    """
    left_segments = left.get("segments", [])
    right_segments = right.get("segments", [])
    left_cast = left.get("cast", []) or []
    right_cast = right.get("cast", []) or []

    # Emotion distributions
    left_dist = _emotion_distribution(left_segments)
    right_dist = _emotion_distribution(right_segments)

    # Per-emotion delta (left - right)
    emotion_deltas = {
        label: round(left_dist[label] - right_dist[label], 4)
        for label in _EMOTION_LABELS
    }

    # Confidence
    left_conf = _avg_confidence(left_segments)
    right_conf = _avg_confidence(right_segments)

    # Entropy (diversity)
    left_entropy = _entropy(left_dist)
    right_entropy = _entropy(right_dist)

    # Cast overlap
    left_actors = {c.get("name", "").strip() for c in left_cast if c.get("name")}
    right_actors = {c.get("name", "").strip() for c in right_cast if c.get("name")}
    common_actors = sorted(left_actors & right_actors)
    actor_overlap = _jaccard(left_actors, right_actors)

    # Cosine similarity (how alike emotionally)
    similarity = _cosine_similarity(left_dist, right_dist)

    # Winners
    def winner(a: float, b: float, threshold: float = 0.001) -> str:
        if abs(a - b) < threshold:
            return "tie"
        return "left" if a > b else "right"

    winners = {
        "more_happy": winner(left_dist["happiness"], right_dist["happiness"]),
        "more_sad": winner(left_dist["sadness"], right_dist["sadness"]),
        "more_angry": winner(left_dist["anger"], right_dist["anger"]),
        "more_fearful": winner(left_dist["fear"], right_dist["fear"]),
        "more_surprised": winner(left_dist["surprise"], right_dist["surprise"]),
        "more_disgusted": winner(left_dist["disgust"], right_dist["disgust"]),
        "more_intense": winner(left_conf, right_conf),
        "more_diverse": winner(left_entropy, right_entropy),
        "more_characters": winner(len(left_actors), len(right_actors)),
        "longer": winner(len(left_segments), len(right_segments)),
    }

    return {
        "left": left,
        "right": right,
        "comparison": {
            "emotion_distribution": {
                "left": left_dist,
                "right": right_dist,
            },
            "emotion_deltas": emotion_deltas,
            "winners": winners,
            "common_actors": common_actors,
            "actor_overlap": actor_overlap,
            "similarity": similarity,
            "stats": {
                "left_avg_confidence": left_conf,
                "right_avg_confidence": right_conf,
                "left_diversity": left_entropy,
                "right_diversity": right_entropy,
                "left_segment_count": len(left_segments),
                "right_segment_count": len(right_segments),
                "left_character_count": len(left_actors),
                "right_character_count": len(right_actors),
            },
        },
    }
