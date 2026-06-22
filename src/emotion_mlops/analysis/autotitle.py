"""
AI Auto-Title Generator.

Uses Groq LLM (llama-3.1-8b-instant, free tier) to generate punchy,
descriptive titles for analyzed content. Given transcript segments,
dominant emotion, and cast, produces a 3-7 word title that captures
what the content is about.

Used by the Compare endpoint to label left/right sides with meaningful
names instead of "Show A" / "Show B" when the user doesn't provide one.

Lazy + thread-safe singleton client. Falls back gracefully on errors.

Author: Ceren (Sprint 4)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_client = None
_GROQ_MODEL = "llama-3.1-8b-instant"


def _get_client():
    """Lazy-load Groq client. Reads GROQ_API_KEY from env."""
    global _client
    if _client is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass  # dotenv optional, env may already be set

        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to .env or environment.")
        _client = Groq(api_key=api_key)
    return _client


def _build_prompt(
    segments: List[Dict[str, Any]],
    cast: List[Dict[str, Any]],
    dominant_emotion: str | None = None,
) -> str:
    """
    Build a prompt for the title-generation LLM.

    We give the model:
      - up to 6 representative sentences from the transcript,
      - the dominant emotion (if computed already),
      - the top 3 character names (if cast present).
    Then ask for a single 3-7 word title.
    """
    # Sample 6 sentences (first + spread + last) for context
    sentences = [
        (seg.get("Sentence") or seg.get("sentence") or "").strip() for seg in segments
    ]
    sentences = [s for s in sentences if s]
    if len(sentences) > 6:
        # Take first, last, and 4 evenly spread from the middle
        step = max(1, len(sentences) // 6)
        sentences = sentences[::step][:6]

    sentence_block = "\n".join(f"- {s}" for s in sentences) or "(no sentences)"

    # Top 3 characters by mentions
    cast_block = ""
    if cast:
        sorted_cast = sorted(cast, key=lambda c: c.get("mentions", 0), reverse=True)
        names = [c.get("name", "") for c in sorted_cast[:3] if c.get("name")]
        if names:
            cast_block = f"\nMain characters: {', '.join(names)}"

    emotion_block = ""
    if dominant_emotion:
        emotion_block = f"\nDominant emotion: {dominant_emotion}"

    prompt = f"""You are a content analyst for a TV/film research agency.

Given this transcript excerpt, generate a punchy 3-7 word title that
captures what this content is about. Be specific and evocative.
Examples of good titles:
  - "Trump's Furious Press Walkout"
  - "Bake-Off Final Showdown"
  - "Late-Night Comedy Monologue"

Transcript excerpt:
{sentence_block}
{cast_block}{emotion_block}

Reply with ONLY the title, no quotes, no explanation, no period at the end.
Title:"""
    return prompt


def generate_title(
    segments: List[Dict[str, Any]],
    cast: List[Dict[str, Any]] | None = None,
    dominant_emotion: str | None = None,
    fallback: str = "Untitled Show",
) -> str:
    """
    Generate an AI title for content.

    Args:
        segments: List of segment dicts with a Sentence key.
        cast: Optional cast list (cast_analyzer output).
        dominant_emotion: Optional pre-computed dominant emotion.
        fallback: String to return if the LLM call fails.

    Returns:
        A short title string (3-7 words), or the fallback on failure.
    """
    if not segments:
        return fallback

    try:
        client = _get_client()
        prompt = _build_prompt(segments, cast or [], dominant_emotion)

        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.6,
        )
        title = response.choices[0].message.content.strip()

        # Clean common LLM artifacts
        title = title.strip().strip('"').strip("'").rstrip(".")
        # Strip "Title:" prefix if model echoed it
        for prefix in ("Title:", "title:", "TITLE:"):
            if title.startswith(prefix):
                title = title[len(prefix) :].strip()

        # Cap absurdly long answers
        if len(title) > 80:
            title = title[:80].rsplit(" ", 1)[0]

        return title or fallback

    except Exception as e:
        logger.warning("[AutoTitle] LLM call failed: %s", e)
        return fallback
