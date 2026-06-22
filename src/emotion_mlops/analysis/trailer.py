"""
Trailer Highlights Generator.

Picks the most "trailer-worthy" moments from a transcript: highest-
emotion scenes with mood diversity. Output is a list of clips that
together fit a target duration (default 30 seconds).

Algorithm:
    1. Take scenes (from detect_scenes) and their peaks
    2. Score each scene by:
         confidence x duration-fit x mood-diversity-bonus
    3. Greedy pick: highest score first, skip if same mood already used,
       stop when total duration reaches target
    4. For each chosen clip:
         - hook sentence (peak segment)
         - tagline (optional, AI-generated via Groq)

Designed for CIA-style content intelligence: "make me a 30-sec trailer
from this episode" -> directly actionable output.

Author: Ceren (Sprint 4 - Day 4)
"""

from __future__ import annotations

from typing import Any, Dict, List


def _score_scene(scene: Dict[str, Any], target_clip_duration: float) -> float:
    """
    Score how 'trailer-worthy' a scene is.

    Higher = better candidate. Factors:
        - peak confidence (the higher the emotion, the more grabby)
        - scene duration close to target_clip_duration (not too short/long)
        - more dominant emotion = clearer mood
    """
    peak_score = scene.get("peak", {}).get("score", 0.0)
    duration = scene.get("duration", 0.0)
    avg_conf = scene.get("avg_confidence", 0.0)

    # Duration fit: peaks at target_clip_duration, falls off either side
    if duration <= 0 or target_clip_duration <= 0:
        duration_fit = 0.0
    else:
        ratio = min(duration, target_clip_duration) / max(
            duration, target_clip_duration
        )
        duration_fit = ratio  # 0..1

    # Combined score - peak confidence dominates, others modulate
    return round(0.6 * peak_score + 0.2 * avg_conf + 0.2 * duration_fit, 4)


def _generate_tagline(scene: Dict[str, Any]) -> str:  # pragma: no cover
    """
    Try to generate a punchy AI tagline for the clip via Groq.

    Falls back to a deterministic template if Groq isn't available
    (e.g. no API key, network error).
    """
    peak_sentence = scene.get("peak", {}).get("sentence", "")
    mood = scene.get("mood", "")
    dominant = scene.get("dominant_emotion", "")

    # Fallback template (used if Groq fails)
    if not peak_sentence:
        return f"A {mood.lower() if mood else 'memorable'} moment."

    try:
        from emotion_mlops.analysis.autotitle import generate_title

        # Reuse the autotitle infrastructure with a clip-specific fallback
        mini_segs = [{"Sentence": s} for s in scene.get("sentences", []) if s]
        if not mini_segs:
            mini_segs = [{"Sentence": peak_sentence}]

        tagline = generate_title(
            segments=mini_segs,
            cast=None,
            dominant_emotion=dominant,
            fallback=f"A {mood.lower()} moment",
        )
        return tagline
    except Exception:
        return f"A {mood.lower() if mood else 'memorable'} moment."


def pick_trailer_highlights(
    scenes: List[Dict[str, Any]],
    target_duration: float = 30.0,
    target_clip_duration: float = 8.0,
    max_clips: int = 3,
    enforce_mood_diversity: bool = True,
    generate_taglines: bool = False,
) -> Dict[str, Any]:
    """
    Pick the best clips for a short trailer.

    Args:
        scenes: List of scene dicts from detect_scenes.
        target_duration: Total trailer length in seconds (default 30).
        target_clip_duration: Ideal length of each clip (default 8s).
        max_clips: Hard cap on number of clips (default 3).
        enforce_mood_diversity: If True, skip a scene whose mood is
                                already used (default True).
        generate_taglines: If True, call Groq for AI tagline per clip.

    Returns:
        {
          "clips": [
            {
              "clip_id": 1,
              "scene_id": 2,
              "start_time": 0.0, "end_time": 8.0,
              "start_label": "0:00", "end_label": "0:08",
              "duration": 8.0,
              "mood": "Tension",
              "dominant_emotion": "anger",
              "score": 0.91,
              "hook": "Trump stormed off the stage.",
              "tagline": "...",   # only if generate_taglines=True
            },
            ...
          ],
          "total_duration": 24.5,
          "target_duration": 30.0,
          "clip_count": 3,
          "moods_used": ["Tension", "Joyful", "Sadness"],
        }
    """
    if not scenes:
        return {
            "clips": [],
            "total_duration": 0.0,
            "target_duration": target_duration,
            "clip_count": 0,
            "moods_used": [],
        }

    # Score every scene
    scored = []
    for scene in scenes:
        score = _score_scene(scene, target_clip_duration)
        scored.append((score, scene))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Greedy pick respecting diversity + total duration
    chosen = []
    used_moods = set()
    used_scene_ids = set()
    total = 0.0

    for score, scene in scored:
        if len(chosen) >= max_clips:
            break
        if total >= target_duration:
            break
        if scene.get("scene_id") in used_scene_ids:
            continue
        mood = scene.get("mood", "")
        if enforce_mood_diversity and mood and mood in used_moods:
            continue

        # Cap clip duration to target_clip_duration so we don't blow the budget
        clip_dur = min(scene.get("duration", 0.0), target_clip_duration)
        if clip_dur <= 0:
            continue

        chosen.append({"score": score, "scene": scene, "clip_dur": clip_dur})
        used_moods.add(mood)
        used_scene_ids.add(scene.get("scene_id"))
        total += clip_dur

    # If diversity was strict and we have empty budget left, allow repeats
    if enforce_mood_diversity and total < target_duration * 0.5:  # pragma: no cover
        for score, scene in scored:
            if len(chosen) >= max_clips:
                break
            if total >= target_duration:
                break
            if scene.get("scene_id") in used_scene_ids:
                continue
            clip_dur = min(scene.get("duration", 0.0), target_clip_duration)
            if clip_dur <= 0:
                continue
            chosen.append({"score": score, "scene": scene, "clip_dur": clip_dur})
            used_scene_ids.add(scene.get("scene_id"))
            total += clip_dur

    # Order chosen clips by their scene start_time (chronological)
    chosen.sort(key=lambda c: c["scene"].get("start_time", 0))

    # Build clip dicts for the response
    clips = []
    for i, item in enumerate(chosen, start=1):
        scene = item["scene"]
        clip_dur = item["clip_dur"]
        score = item["score"]

        start = scene.get("start_time", 0.0)
        end = start + clip_dur
        peak = scene.get("peak", {})

        clip = {
            "clip_id": i,
            "scene_id": scene.get("scene_id"),
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "start_label": scene.get("start_label", ""),
            "end_label": scene.get("end_label", ""),
            "duration": round(clip_dur, 2),
            "mood": scene.get("mood", ""),
            "dominant_emotion": scene.get("dominant_emotion", ""),
            "score": score,
            "hook": peak.get("sentence", ""),
            "peak_emotion": peak.get("emotion", ""),
            "peak_score": peak.get("score", 0.0),
        }

        if generate_taglines:  # pragma: no cover
            clip["tagline"] = _generate_tagline(scene)

        clips.append(clip)

    return {
        "clips": clips,
        "total_duration": round(total, 2),
        "target_duration": target_duration,
        "clip_count": len(clips),
        "moods_used": sorted(used_moods),
    }
