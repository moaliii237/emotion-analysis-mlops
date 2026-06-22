"""
Scene Detection Analyzer.

Groups consecutive segments into "scenes" based on emotional cohesion.
A scene is a run of segments that share a dominant emotion (or stay
emotionally similar). Designed to give producers an at-a-glance
breakdown: where's the climax? where's the comic relief? where would
a product placement fit?

Inputs come from pipeline segments (Start, End, Sentence, Emotion,
Confidence). Output is a list of scene dicts ready for the frontend.

Bonus: each scene can be given an AI-generated title via autotitle
(opt-in to keep the endpoint fast when titles aren't needed).

Author: Ceren (Sprint 4 - Day 3)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


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
    """Format seconds as 'M:SS' (e.g. 1:23)."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _classify_scene_mood(emotion_counts: Dict[str, int], total: int) -> str:
    """
    Give the scene a mood label based on its dominant emotion mix.

    Returns one of: Tension, Climax, Comic Relief, Sadness, Mystery,
    Disgust, Calm, Mixed.
    """
    if total == 0:
        return "Calm"

    pct = {k: v / total for k, v in emotion_counts.items()}
    anger = pct.get("anger", 0)
    fear = pct.get("fear", 0)
    sadness = pct.get("sadness", 0)
    happiness = pct.get("happiness", 0)
    surprise = pct.get("surprise", 0)
    disgust = pct.get("disgust", 0)
    neutral = pct.get("neutral", 0)

    # High-emotion mix -> climax
    intense_count = sum(1 for v in [anger, fear, sadness, happiness] if v > 0.25)
    if intense_count >= 3:
        return "Climax"

    if anger > 0.5 or (anger > 0.3 and (fear > 0.2 or disgust > 0.2)):
        return "Tension"
    if happiness > 0.5 and surprise > 0.1:
        return "Comic Relief"
    if happiness > 0.5:
        return "Joyful"
    if sadness > 0.5:
        return "Sadness"
    if fear > 0.4:
        return "Mystery"
    if disgust > 0.4:
        return "Disgust"
    if neutral > 0.6:
        return "Calm"
    return "Mixed"


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def detect_scenes(
    segments: List[Dict[str, Any]],
    min_segments_per_scene: int = 2,
    boundary_change_threshold: float = 0.4,
    generate_titles: bool = False,
    classifier=None,
) -> List[Dict[str, Any]]:
    """
    Group consecutive segments into emotional scenes.

    Algorithm:
        1. Walk through segments in order
        2. Start a new scene when the dominant emotion of recent segments
           shifts significantly (>= boundary_change_threshold)
        3. Merge short scenes (< min_segments_per_scene) into neighbours
        4. Compute stats per scene
        5. Optionally generate AI titles (Groq)

    Args:
        segments: Pipeline segment dicts (Start, End, Sentence, Emotion,
                  Confidence).
        min_segments_per_scene: Scenes shorter than this get merged.
        boundary_change_threshold: How much the rolling emotion average
                                   must shift to trigger a new scene (0-1).
        generate_titles: If True, call Groq autotitle for each scene.
        classifier: Required if you want per-scene cast (not used yet).

    Returns:
        List of scene dicts:
        [
          {
            "scene_id": 1,
            "start_time": 0.0,
            "end_time": 30.0,
            "start_label": "0:00",
            "end_label": "0:30",
            "duration": 30.0,
            "segment_count": 5,
            "dominant_emotion": "anger",
            "emotion_counts": {"anger": 4, "neutral": 1},
            "avg_confidence": 0.87,
            "mood": "Tension",
            "peak": {
                "time": 8.0, "emotion": "anger",
                "score": 0.96, "sentence": "..."
            },
            "title": "The Furious Outburst",   # if generate_titles=True
          },
          ...
        ]
    """
    if not segments:
        return []

    # Build per-segment normalized data
    parsed = []
    for seg in segments:
        parsed.append(
            {
                "start": _parse_time(seg.get("Start") or seg.get("start") or 0),
                "end": _parse_time(seg.get("End") or seg.get("end") or 0),
                "sentence": (seg.get("Sentence") or seg.get("sentence") or "").strip(),
                "emotion": (seg.get("Emotion") or seg.get("emotion") or "").strip(),
                "confidence": float(
                    seg.get("Confidence") or seg.get("confidence") or 0.0
                ),
            }
        )

    # Group into raw scenes by detecting emotion shifts
    raw_scenes: List[List[Dict[str, Any]]] = []
    current_scene: List[Dict[str, Any]] = []
    current_dominant = None

    for seg in parsed:
        if not current_scene:
            current_scene.append(seg)
            current_dominant = seg["emotion"]
            continue

        # Check if emotion has shifted
        # Use rolling window: last 2 segments + this one
        window = current_scene[-2:] + [seg]
        emotions = [s["emotion"] for s in window]
        most_common = Counter(emotions).most_common(1)[0][0]

        if most_common != current_dominant:
            # Boundary detected: save current scene, start new one
            raw_scenes.append(current_scene)
            current_scene = [seg]
            current_dominant = seg["emotion"]
        else:
            current_scene.append(seg)

    if current_scene:
        raw_scenes.append(current_scene)

    # Merge scenes shorter than min_segments_per_scene into neighbours
    merged: List[List[Dict[str, Any]]] = []
    for scene in raw_scenes:
        if (
            len(scene) < min_segments_per_scene
            and merged
            and len(merged[-1]) >= min_segments_per_scene
        ):
            # Append short scene to previous big one
            merged[-1].extend(scene)
        else:
            merged.append(scene)

    # If first scene ended up too short, merge it forward
    if merged and len(merged[0]) < min_segments_per_scene and len(merged) > 1:
        first = merged.pop(0)
        merged[0] = first + merged[0]

    # Build scene metadata
    out = []
    for i, scene_segs in enumerate(merged, start=1):
        if not scene_segs:
            continue

        emotions = [s["emotion"] for s in scene_segs if s["emotion"]]
        emotion_counts = dict(Counter(emotions))
        dominant = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
        confs = [s["confidence"] for s in scene_segs if s["confidence"] > 0]
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
        start_time = scene_segs[0]["start"]
        end_time = max(s["end"] or s["start"] for s in scene_segs)
        mood = _classify_scene_mood(emotion_counts, len(emotions))

        # Find this scene's peak (highest confidence segment)
        peak_seg = max(scene_segs, key=lambda s: s["confidence"])
        peak = {
            "time": peak_seg["start"],
            "emotion": peak_seg["emotion"],
            "score": round(peak_seg["confidence"], 4),
            "sentence": peak_seg["sentence"],
        }

        scene_dict = {
            "scene_id": i,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "start_label": _format_time(start_time),
            "end_label": _format_time(end_time),
            "duration": round(end_time - start_time, 2),
            "segment_count": len(scene_segs),
            "dominant_emotion": dominant,
            "emotion_counts": emotion_counts,
            "avg_confidence": avg_conf,
            "mood": mood,
            "peak": peak,
            "sentences": [s["sentence"] for s in scene_segs],
        }

        # Optional: AI title via Groq
        if generate_titles:  # pragma: no cover
            try:
                from emotion_mlops.analysis.autotitle import generate_title

                # Build a mini "segments" list for title generation
                mini_segs = [
                    {"Sentence": s["sentence"]} for s in scene_segs if s["sentence"]
                ]
                scene_dict["title"] = generate_title(
                    segments=mini_segs,
                    cast=None,
                    dominant_emotion=dominant,
                    fallback=f"Scene {i}",
                )
            except Exception:
                scene_dict["title"] = f"Scene {i}"

        out.append(scene_dict)

    return out
