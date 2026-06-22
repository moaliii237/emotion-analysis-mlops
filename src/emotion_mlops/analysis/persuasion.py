"""
Persuasion Score Analyzer.

Scores a speaker's persuasive power based on transcript signals:
how confident the model is (clarity), how much emotional appeal they
use, how consistent their rhetoric is, and how much they vary tone.

Designed for analyzing debates, sales pitches, podcasts, and
political speeches. The Groq LLM produces a human-readable assessment
at the end so the scorecard tells a story.

Author: Ceren (Sprint 4 - Day 6)
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

# Strong-emotion labels that contribute to emotional appeal.
# Neutral and mild emotions don't move audiences.
_STRONG_EMOTIONS = {"anger", "fear", "happiness", "sadness", "disgust"}


# ----------------------------------------------------------------------
# Per-axis scorers
# ----------------------------------------------------------------------


def _rhetorical_strength(confidences: List[float]) -> float:
    """
    How clear / well-articulated is the speech?

    Proxy: average classifier confidence. When the model is sure of
    each sentence's emotion, the speech is well-articulated. When
    confidences are low, the speaker is muddled or off-message.
    """
    if not confidences:
        return 0.0
    return round(sum(confidences) / len(confidences), 4)


def _emotional_appeal(labels: List[str]) -> float:
    """
    What fraction of segments carry a strong (move-the-audience) emotion?

    Speakers who stay neutral don't move people. Persuasive ones lean
    on strong emotions (fear, hope/happiness, outrage/anger, sadness).
    """
    if not labels:
        return 0.0
    strong = sum(1 for label in labels if label in _STRONG_EMOTIONS)
    return round(strong / len(labels), 4)


def _consistency(labels: List[str]) -> float:
    """
    Stays-on-message score: share occupied by the dominant emotion.

    Highly persuasive speakers hammer a single emotional note (anger
    against the elite, hope for the future, fear of the other side).
    They don't constantly switch register.
    """
    if not labels:
        return 0.0
    counts = Counter(labels)
    most_common = counts.most_common(1)[0][1]
    return round(most_common / len(labels), 4)


def _variety_bonus(labels: List[str]) -> float:
    """
    Some variety is good - relentless monotone loses people.

    Returns Shannon entropy normalized to 0..1. Used as a soft modulator
    so the final score doesn't reward pure monotony.
    """
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    H_max = math.log2(len(_EMOTION_LABELS))
    return round(H / H_max, 4) if H_max > 0 else 0.0


def _confidence_label(persuasion: float) -> str:
    """Bucket the 0..1 score into a human-readable label."""
    if persuasion >= 0.8:
        return "Very High"
    if persuasion >= 0.65:
        return "High"
    if persuasion >= 0.5:
        return "Moderate"
    if persuasion >= 0.35:
        return "Low"
    return "Very Low"


def _ai_assessment(  # pragma: no cover
    scores: Dict[str, float],
    dominant_emotion: str,
    sample_sentences: List[str],
) -> str | None:
    """
    Ask Groq for a one-paragraph qualitative assessment.

    Returns None on any failure so the caller can fall back to a
    deterministic summary.
    """
    try:
        from emotion_mlops.analysis.autotitle import _GROQ_MODEL, _get_client

        sample = " ".join(sample_sentences[:5])[:400]
        client = _get_client()
        prompt = f"""You are a media analyst evaluating how persuasive a
speaker is, based on automated transcript analysis.

Scores (0..1, higher = stronger):
- Rhetorical strength (clarity / articulation): {scores['rhetorical_strength']:.2f}
- Emotional appeal (use of strong emotions):    {scores['emotional_appeal']:.2f}
- Consistency (stays on-message):               {scores['consistency']:.2f}
- Variety (avoids monotone):                    {scores['variety']:.2f}
- Overall persuasion score:                     {scores['persuasion_score']:.2f}
- Dominant emotional register: {dominant_emotion}

Excerpt: "{sample}"

Write ONE concise paragraph (2-3 sentences max) assessing this speaker's
persuasive power and the kind of audience they'd most likely move. Be
specific and observational - not flattering, not dismissive."""

        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.55,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def _fallback_assessment(
    scores: Dict[str, float],
    dominant_emotion: str,
) -> str:
    """Deterministic summary used when the LLM call fails."""
    score = scores["persuasion_score"]
    label = _confidence_label(score)
    appeal = scores["emotional_appeal"]
    consistency = scores["consistency"]

    if score >= 0.7 and appeal >= 0.7:
        style = f"high-impact emotional rhetoric anchored in {dominant_emotion}"
    elif consistency >= 0.7:
        style = f"consistent {dominant_emotion}-driven messaging"
    elif appeal >= 0.5:
        style = "moderate emotional appeal with mixed register"
    else:
        style = "low-impact, mostly neutral delivery"

    return (
        f"{label} persuasive potential - {style}. "
        f"Likely to resonate with audiences receptive to {dominant_emotion}-"
        f"based framing."
    )


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def analyze_persuasion(
    segments: List[Dict[str, Any]],
    use_ai_assessment: bool = True,
) -> Dict[str, Any]:
    """
    Compute a persuasion scorecard for the speaker(s) in the transcript.

    Args:
        segments: Pipeline segments with Sentence, Emotion, Confidence.
        use_ai_assessment: If True, ask Groq for a paragraph summary.
                           Falls back to a deterministic summary on error.

    Returns:
        {
          "persuasion_score": 0.82,
          "confidence_level": "Very High" | "High" | ...,
          "scores": {
              "rhetorical_strength": ...,
              "emotional_appeal":    ...,
              "consistency":         ...,
              "variety":             ...
          },
          "dominant_emotion": "anger",
          "assessment": "...",
          "stats": {
              "segment_count": ...,
              "strong_emotion_count": ...,
              "ai_used": True/False
          }
        }
    """
    if not segments:
        return {
            "persuasion_score": 0.0,
            "confidence_level": "Very Low",
            "scores": {
                "rhetorical_strength": 0.0,
                "emotional_appeal": 0.0,
                "consistency": 0.0,
                "variety": 0.0,
            },
            "dominant_emotion": None,
            "assessment": "No content to analyze.",
            "stats": {
                "segment_count": 0,
                "strong_emotion_count": 0,
                "ai_used": False,
            },
        }

    labels = [
        (s.get("Emotion") or s.get("emotion") or "").lower().strip() for s in segments
    ]
    labels = [label for label in labels if label]
    confidences = [
        float(s.get("Confidence") or s.get("confidence") or 0.0) for s in segments
    ]
    confidences = [c for c in confidences if c > 0]
    sentences = [
        (s.get("Sentence") or s.get("sentence") or "").strip() for s in segments
    ]
    sentences = [s for s in sentences if s]

    # Axes
    rhetorical = _rhetorical_strength(confidences)
    appeal = _emotional_appeal(labels)
    consistency = _consistency(labels)
    variety = _variety_bonus(labels)

    # Composite persuasion score
    # Weighting: emotional appeal + rhetorical strength dominate;
    # consistency + variety modulate. Variety is small bonus, monotony
    # is mildly penalized via the (variety * 0.5 + 0.5) factor.
    persuasion = (
        0.4 * appeal
        + 0.3 * rhetorical
        + 0.2 * consistency
        + 0.1 * (variety * 0.5 + 0.5)
    )
    persuasion = round(min(max(persuasion, 0.0), 1.0), 4)

    confidence_label = _confidence_label(persuasion)

    # Dominant emotion
    counts = Counter(labels)
    dominant = counts.most_common(1)[0][0] if counts else None
    strong_count = sum(1 for lab in labels if lab in _STRONG_EMOTIONS)

    scores_dict = {
        "rhetorical_strength": rhetorical,
        "emotional_appeal": appeal,
        "consistency": consistency,
        "variety": variety,
        "persuasion_score": persuasion,  # included for LLM prompt context
    }

    # AI assessment (with fallback)
    ai_result = None
    if use_ai_assessment:  # pragma: no cover
        ai_result = _ai_assessment(
            scores=scores_dict,
            dominant_emotion=dominant or "neutral",
            sample_sentences=sentences,
        )

    assessment = ai_result or _fallback_assessment(
        scores=scores_dict,
        dominant_emotion=dominant or "neutral",
    )

    return {
        "persuasion_score": persuasion,
        "confidence_level": confidence_label,
        "scores": {
            "rhetorical_strength": rhetorical,
            "emotional_appeal": appeal,
            "consistency": consistency,
            "variety": variety,
        },
        "dominant_emotion": dominant,
        "assessment": assessment,
        "stats": {
            "segment_count": len(segments),
            "strong_emotion_count": strong_count,
            "ai_used": ai_result is not None,
        },
    }
