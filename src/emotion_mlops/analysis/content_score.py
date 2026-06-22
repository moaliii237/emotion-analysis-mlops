"""
Content Score Generator.

Takes a full pipeline output (segments + scenes + timeline) and produces
a producer-facing content scorecard. Each axis gets a 0..1 score, plus
a final letter grade and an AI-written classification + audience
recommendation.

Designed for content-intelligence clients (CIA-style): given a TV
episode, podcast, or YouTube video, answer "What kind of content is
this and who's it for?".

Author: Ceren (Sprint 4 - Day 5)
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
# Per-axis scorers
# ----------------------------------------------------------------------


def _engagement_score(dist: Dict[str, float], avg_conf: float) -> float:
    """
    How emotionally engaging is the content?

    High when a lot of segments carry strong emotion (low neutral, high
    avg confidence). Low when most content is flat/neutral.
    """
    neutral = dist.get("neutral", 0.0)
    non_neutral = 1.0 - neutral  # share of clearly emotional content
    return round(0.7 * non_neutral + 0.3 * avg_conf, 4)


def _drama_score(dist: Dict[str, float]) -> float:
    """
    How dramatic is it? Sum of "heavy" emotion shares.

    Anger, sadness, fear, disgust all contribute. Capped at 1.0.
    """
    drama_emotions = ["anger", "sadness", "fear", "disgust"]
    raw = sum(dist.get(e, 0.0) for e in drama_emotions)
    return round(min(raw, 1.0), 4)


def _comedy_score(dist: Dict[str, float]) -> float:
    """How comedic is it? Happiness + surprise are the comedy axes."""
    raw = dist.get("happiness", 0.0) * 0.8 + dist.get("surprise", 0.0) * 0.2
    return round(min(raw, 1.0), 4)


def _intensity_score(avg_conf: float, dist: Dict[str, float]) -> float:
    """
    Average emotional intensity. High avg confidence + low neutral.
    """
    neutral_penalty = 1.0 - dist.get("neutral", 0.0)
    return round(0.5 * avg_conf + 0.5 * neutral_penalty, 4)


def _emotional_range(dist: Dict[str, float]) -> float:
    """
    How varied are the emotions? Normalized Shannon entropy.

    1.0 = perfectly balanced across all emotions, 0.0 = one emotion only.
    """
    values = [v for v in dist.values() if v > 0]
    if not values:
        return 0.0
    H = -sum(p * math.log2(p) for p in values)
    H_max = math.log2(len(_EMOTION_LABELS))
    return round(H / H_max, 4) if H_max > 0 else 0.0


def _pacing_score(labels: List[str]) -> float:
    """
    How often does the dominant emotion change?

    Counts label transitions / max possible transitions. High = dynamic
    content (lots of mood swings), low = monotone.
    """
    if len(labels) < 2:
        return 0.0
    transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    return round(transitions / (len(labels) - 1), 4)


def _grade(scores: Dict[str, float]) -> str:
    """
    Combine axes into a letter grade A+/A/A-/B+/B/B-/C+/C.

    Weighting:
        engagement     0.30
        intensity      0.20
        emotional_range 0.20
        pacing         0.15
        drama_or_comedy_peak 0.15  (whichever is higher)
    """
    drama_or_comedy = max(scores["drama"], scores["comedy"])
    composite = (
        0.30 * scores["engagement"]
        + 0.20 * scores["intensity"]
        + 0.20 * scores["emotional_range"]
        + 0.15 * scores["pacing"]
        + 0.15 * drama_or_comedy
    )

    if composite >= 0.85:
        return "A+"
    if composite >= 0.75:
        return "A"
    if composite >= 0.65:
        return "A-"
    if composite >= 0.55:
        return "B+"
    if composite >= 0.45:
        return "B"
    if composite >= 0.35:
        return "B-"
    if composite >= 0.25:
        return "C+"
    return "C"


def _classify_content(scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Heuristic classification fallback. Used if AI classification fails.

    Looks at the dominant axis and returns a content-type label +
    audience recommendation.
    """
    drama = scores["drama"]
    comedy = scores["comedy"]
    intensity = scores["intensity"]
    erange = scores["emotional_range"]

    if drama > 0.5 and intensity > 0.6:
        cls = "Intense drama"
        audience = "Drama enthusiasts, 25-54"
        suitable = ["Prime-time TV drama", "Streaming series"]
    elif comedy > 0.5 and intensity > 0.5:
        cls = "Light-hearted comedy"
        audience = "Family audience, 18-49"
        suitable = ["Comedy show", "Variety / talk show"]
    elif erange > 0.7:
        cls = "Emotional rollercoaster"
        audience = "Reality TV fans, 18-45"
        suitable = ["Reality competition", "Big Brother-style format"]
    elif intensity < 0.3:
        cls = "Calm and conversational"
        audience = "Podcast listeners, 25-65"
        suitable = ["Podcast", "Educational content", "Documentary"]
    elif drama > 0.3 and comedy > 0.3:
        cls = "Dramedy"
        audience = "Broad demographic, 18-54"
        suitable = ["Streaming series", "Sitcom"]
    else:
        cls = "Mixed-tone content"
        audience = "General audience"
        suitable = ["General programming"]

    return {
        "classification": cls,
        "target_audience": audience,
        "suitable_for": suitable,
    }


def _ai_classification(  # pragma: no cover
    scores: Dict[str, float],
    dominant_emotion: str,
    scene_count: int,
    cast_names: List[str],
) -> Dict[str, str] | None:
    """
    Try to get an AI-written classification + audience pitch via Groq.

    Returns None on failure so the caller can fall back to the
    heuristic classifier.
    """
    try:
        from emotion_mlops.analysis.autotitle import _GROQ_MODEL, _get_client

        client = _get_client()
        prompt = f"""You are a TV content analyst. Given these signals
from automated transcript analysis, give a one-line classification of
the content type and the most likely target audience.

Signals:
- Engagement:      {scores['engagement']:.2f}
- Drama level:     {scores['drama']:.2f}
- Comedy level:    {scores['comedy']:.2f}
- Intensity:       {scores['intensity']:.2f}
- Emotional range: {scores['emotional_range']:.2f}
- Pacing:          {scores['pacing']:.2f}
- Dominant emotion: {dominant_emotion}
- Scenes detected:  {scene_count}
- Main characters:  {', '.join(cast_names[:5]) if cast_names else '(none)'}

Reply in this exact format (two lines only, no quotes, no extra text):
Classification: <one short phrase>
Audience: <one short phrase>"""

        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.5,
        )
        text = response.choices[0].message.content.strip()

        cls = None
        aud = None
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("classification:"):
                cls = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line.lower().startswith("audience:"):
                aud = line.split(":", 1)[1].strip().strip('"').strip("'")

        if cls and aud:
            return {"classification": cls, "target_audience": aud}
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def score_content(
    segments: List[Dict[str, Any]],
    cast: List[Dict[str, Any]] | None = None,
    scenes: List[Dict[str, Any]] | None = None,
    use_ai_classification: bool = True,
) -> Dict[str, Any]:
    """
    Produce a full content scorecard.

    Args:
        segments: Pipeline segments (with Emotion and Confidence).
        cast: Optional cast list (from cast_analyzer).
        scenes: Optional scenes list (from detect_scenes).
        use_ai_classification: If True, ask Groq for classification +
                               audience. Falls back to heuristic on
                               failure.

    Returns:
        {
          "scores": {
              "engagement": ..., "drama": ..., "comedy": ...,
              "intensity": ..., "emotional_range": ..., "pacing": ...
          },
          "grade": "A-",
          "classification": "...",
          "target_audience": "...",
          "suitable_for": [...],
          "stats": {
              "segment_count": ..., "scene_count": ...,
              "character_count": ..., "dominant_emotion": ...,
              "ai_classification_used": True/False
          }
        }
    """
    if not segments:
        return {
            "scores": {
                k: 0.0
                for k in [
                    "engagement",
                    "drama",
                    "comedy",
                    "intensity",
                    "emotional_range",
                    "pacing",
                ]
            },
            "grade": "C",
            "classification": "No content",
            "target_audience": "-",
            "suitable_for": [],
            "stats": {
                "segment_count": 0,
                "scene_count": 0,
                "character_count": 0,
                "dominant_emotion": None,
                "ai_classification_used": False,
            },
        }

    # Emotion distribution
    labels = [
        s.get("Emotion") or s.get("emotion")
        for s in segments
        if (s.get("Emotion") or s.get("emotion"))
    ]
    counts = Counter(labels)
    total = sum(counts.values())
    dist = {label: 0.0 for label in _EMOTION_LABELS}
    for label, c in counts.items():
        if label in dist:
            dist[label] = c / total

    # Avg confidence
    confs = [float(s.get("Confidence") or s.get("confidence") or 0.0) for s in segments]
    confs = [c for c in confs if c > 0]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    # Compute axes
    scores = {
        "engagement": _engagement_score(dist, avg_conf),
        "drama": _drama_score(dist),
        "comedy": _comedy_score(dist),
        "intensity": _intensity_score(avg_conf, dist),
        "emotional_range": _emotional_range(dist),
        "pacing": _pacing_score(labels),
    }

    grade = _grade(scores)
    dominant = counts.most_common(1)[0][0] if counts else None
    cast_names = [c.get("name", "") for c in (cast or []) if c.get("name")]

    # AI classification (with fallback)
    ai_result = None
    if use_ai_classification:  # pragma: no cover
        ai_result = _ai_classification(
            scores=scores,
            dominant_emotion=dominant or "neutral",
            scene_count=len(scenes or []),
            cast_names=cast_names,
        )

    if ai_result:
        classification_data = {
            "classification": ai_result["classification"],
            "target_audience": ai_result["target_audience"],
            "suitable_for": _classify_content(scores)["suitable_for"],
        }
        ai_used = True
    else:
        classification_data = _classify_content(scores)
        ai_used = False

    return {
        "scores": scores,
        "grade": grade,
        "classification": classification_data["classification"],
        "target_audience": classification_data["target_audience"],
        "suitable_for": classification_data["suitable_for"],
        "stats": {
            "segment_count": len(segments),
            "scene_count": len(scenes or []),
            "character_count": len(cast_names),
            "dominant_emotion": dominant,
            "ai_classification_used": ai_used,
        },
    }
