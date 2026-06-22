"""
LLM Q&A over transcripts using Groq.

Why Groq:
    - Free tier: 30 req/min, plenty for demos and assessment.
    - Sub-second latency on Llama-3.1-70B-Versatile.
    - Zero RAM/GPU cost on OUR server, the LLM runs in Groq's cloud.

Configuration:
    Set GROQ_API_KEY in the environment (or in a .env file at the project root).
    Get a free key at https://console.groq.com/keys

Override the model with the GROQ_MODEL env var if you want to swap to
llama-3.3-70b-versatile, mixtral-8x7b-32768, or any other Groq model.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# llama-3.3-70b-versatile answers transcript questions far better than the
# small instant model, and Groq still returns in about a second.
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_DEFAULT_MAX_CONTEXT_CHARS = 12000  # keep context under most models' window


_SYSTEM_PROMPT = (
    "You are an analytical assistant helping a film studies student explore "
    "TV / movie transcripts that have been annotated with per-line emotions. "
    "Answer questions strictly from the transcript provided. If the answer "
    "isn't there, say so clearly instead of guessing. Be concise (under 200 "
    "words unless asked for detail) and cite line numbers or speaker names "
    "when relevant."
)


def _get_client():
    """Return an authenticated Groq client or raise a clear error."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            'then add to your environment: setx GROQ_API_KEY "gsk_..." (Windows) '
            "or export GROQ_API_KEY=gsk_... (Linux/Mac)."
        )
    from groq import Groq

    return Groq(api_key=api_key)


def _build_context(
    segments: list, max_chars: int = _DEFAULT_MAX_CONTEXT_CHARS
) -> tuple[str, int]:
    """
    Render the transcript segments as a numbered context block for the LLM.

    Cuts off after max_chars to stay within the model's context window, later we can replace this with RAG / sliding window for long transcripts.

    Returns:
        (context_text, segments_included) so callers can report honestly
        how much of the transcript the model actually saw.
    """
    lines = []
    total = 0
    included = 0
    for i, seg in enumerate(segments, 1):
        start = seg.get("Start", "")
        sentence = (seg.get("Sentence") or seg.get("sentence") or "").strip()
        emotion = seg.get("Emotion") or seg.get("emotion") or ""
        speaker = seg.get("Speaker") or seg.get("speaker") or ""
        prefix = f"[{i:03d}]"
        if start:
            prefix += f" {start}"
        if speaker:
            prefix += f" {speaker}:"
        line = f'{prefix} "{sentence}"  ({emotion})'
        if total + len(line) > max_chars:
            lines.append(f"... (truncated, {len(segments) - i + 1} more segments) ...")
            break
        lines.append(line)
        total += len(line)
        included += 1
    return "\n".join(lines), included


def ask_question(
    question: str,
    segments: list,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> dict:
    """
    Answer a question about a transcript using a Groq-hosted LLM.

    Args:
        question: The user's question (free-form text).
        segments: List of transcript segments (same shape as pipeline output).
        model: Override the Groq model. Defaults to GROQ_MODEL env or llama-3.1-8b-instant.
        max_tokens: Upper bound on the LLM's answer length.
        temperature: Sampling temperature (0.2 = mostly deterministic).

    Returns:
        dict with:
            - answer (str): the LLM's answer
            - model (str): which model answered
            - context_segments (int): how many segments were included
            - usage (dict): token counts from the Groq response
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not segments:
        raise ValueError("segments must be a non-empty list")

    client = _get_client()
    used_model = model or _GROQ_MODEL
    context, included = _build_context(segments)

    user_prompt = (
        f"TRANSCRIPT:\n{context}\n\n" f"QUESTION: {question.strip()}\n\n" "ANSWER:"
    )

    logger.info(
        "[QA] Asking Groq (%s), %d segments, %d chars context",
        used_model,
        len(segments),
        len(context),
    )

    completion = client.chat.completions.create(
        model=used_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    answer = completion.choices[0].message.content or ""
    usage = {
        "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
        "completion_tokens": getattr(completion.usage, "completion_tokens", None),
        "total_tokens": getattr(completion.usage, "total_tokens", None),
    }

    return {
        "answer": answer.strip(),
        "model": used_model,
        # How many segments actually made it into the prompt, which can be
        # fewer than len(segments) when the transcript is long.
        "context_segments": included,
        "usage": usage,
    }
