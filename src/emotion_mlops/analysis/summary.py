"""
AI summary module, generates the short "AI Verdict" shown on the
Analyse results page using a Groq-hosted LLM.

How it works:
    1. Build a transcript block from the analysed segments (not just a
       tiny sample, the model needs the actual content to say anything
       specific about it).
    2. Add the aggregate signals (dominant emotion, cast, scenes, score)
       as structured context.
    3. Ask Groq for a 2-3 sentence verdict.

If Groq is unavailable (no GROQ_API_KEY, network error, model error) we
return a clearly-labelled deterministic fallback with ai_generated=False,
and log the real reason, so a bad summary is never silently mistaken for
an AI one.

Configuration:
    GROQ_API_KEY  required for AI summaries (console.groq.com/keys)
    GROQ_MODEL    override the model, defaults to llama-3.3-70b-versatile
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# llama-3.3-70b-versatile gives noticeably richer summaries than the small
# instant model and is still fast on Groq. Swap via GROQ_MODEL if needed.
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# How much transcript the model gets to read. Groq's Llama models have a
# 128k-token window, so 9000 characters is comfortably safe while being
# enough to cover a typical analysed clip in full.
_MAX_TRANSCRIPT_CHARS = 9000


def _build_transcript_block(segments: List[Dict[str, Any]]) -> str:
    """
    Render the segments as numbered transcript lines for the prompt.

    Includes the per-line emotion so the model can reference the emotional
    flow. Stops at _MAX_TRANSCRIPT_CHARS and says how much was cut, so the
    model knows it saw a partial transcript.
    """
    lines: list[str] = []
    total = 0
    for i, seg in enumerate(segments, 1):
        sentence = (seg.get("Sentence") or seg.get("sentence") or "").strip()
        if not sentence:
            continue
        emotion = seg.get("Emotion") or seg.get("emotion") or ""
        line = f"[{i:03d}] {sentence}" + (f"  ({emotion})" if emotion else "")
        if total + len(line) > _MAX_TRANSCRIPT_CHARS:
            lines.append(
                f"... (transcript truncated, {len(segments) - i + 1} more lines)"
            )
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def generate_summary(
    segments: List[Dict[str, Any]],
    cast: List[Dict[str, Any]] | None = None,
    scenes: List[Dict[str, Any]] | None = None,
    score: Dict[str, Any] | None = None,
    persuasion: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Generate a 2-3 sentence AI verdict of the full analysis."""
    cast = cast or []
    scenes = scenes or []

    # Aggregate signals from the analysis, used as structured context.
    emotion_counts: Dict[str, int] = {}
    for s in segments:
        e = s.get("Emotion") or s.get("emotion")
        if e:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

    dominant = (
        max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
    )
    top_chars = (
        ", ".join(c.get("name", "?") for c in cast[:3])
        if cast
        else "no named characters"
    )
    moods = [s.get("mood", "Mixed") for s in scenes][:5]

    classification = (score or {}).get("classification", "Untitled")
    grade = (score or {}).get("grade", "-")
    pers_level = (persuasion or {}).get("confidence_level", "-")

    transcript_block = _build_transcript_block(segments)

    prompt = (
        "You are a sharp content analyst. Below is a transcript that was "
        "analysed sentence-by-sentence for emotion, plus aggregate signals.\n\n"
        f"TRANSCRIPT ({len(segments)} lines, per-line emotion in parentheses):\n"
        f"{transcript_block}\n\n"
        "AGGREGATE ANALYSIS:\n"
        f"- Classification: {classification} (grade {grade})\n"
        f"- Dominant emotion: {dominant} "
        f"({emotion_counts.get(dominant, 0)} of {len(segments)} segments)\n"
        f"- Emotion mix: {dict(emotion_counts)}\n"
        f"- Characters: {top_chars}\n"
        f"- Scene moods: {', '.join(moods) if moods else 'n/a'}\n"
        f"- Persuasion level: {pers_level}\n\n"
        "Write a 2-3 sentence verdict that captures (1) what this content "
        "actually is and what happens in it, (2) its emotional dynamic, and "
        "(3) who would enjoy it. Reference specific people, topics, or "
        "moments from the transcript. Never use filler like 'focused "
        "narrative arc' or 'consistent emotional tone'. No bullet points, "
        "no quotes around the response, conversational tone."
    )

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.warning("[Summary] GROQ_API_KEY not set, returning fallback verdict")
        return {
            "summary": _fallback_summary(dominant, classification, len(segments)),
            "ai_generated": False,
            "error": "GROQ_API_KEY not configured",
        }

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # Low-ish temperature: vivid wording is welcome, invented
            # facts about the content are not.
            temperature=0.4,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = text.strip('"').strip("'").strip()
        if not text:
            raise ValueError("Groq returned an empty response")
        return {"summary": text, "ai_generated": True, "model": _GROQ_MODEL}
    except Exception as e:
        # Log loudly: a silent fallback here is exactly how "the summary
        # is bad" goes unnoticed while everything looks green.
        logger.error(
            "[Summary] Groq call failed (%s), using fallback: %s", _GROQ_MODEL, e
        )
        return {
            "summary": _fallback_summary(dominant, classification, len(segments)),
            "ai_generated": False,
            "error": str(e),
        }


def _fallback_summary(dominant: str, classification: str, n_segments: int) -> str:
    """Deterministic one-liner used when the LLM is unavailable."""
    return (
        f"This {classification.lower()} piece spans {n_segments} segments "
        f"with {dominant} as the dominant emotion. (AI verdict unavailable, "
        f"this is an automatic fallback.)"
    )
