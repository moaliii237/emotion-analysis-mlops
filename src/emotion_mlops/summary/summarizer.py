"""
Transcript summarizer for EmotionScope (Card #2570 nice-to-have).

Uses a distilled BART (distilbart-cnn-6-6, ~300 MB) for fast abstractive
summarization. The model is loaded lazily on first call and cached.

Implementation note:
    transformers 5.x dropped the `summarization` pipeline task. We load the
    seq2seq model directly via AutoModelForSeq2SeqLM and call .generate(),
    which is what the pipeline did under the hood anyway.

Override the model with the SUMMARIZER_MODEL env var if needed.
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_MODEL_NAME = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-6-6")
_MAX_INPUT_TOKENS = 1024  # BART's hard cap
_DEFAULT_MIN_LEN = 30
_DEFAULT_MAX_LEN = 150

# Lazy singletons, populated on first use, kept for the process lifetime.
_tokenizer = None
_model = None
_device: Optional[str] = None
_lock = Lock()


def _ensure_loaded():
    """Load the seq2seq model + tokenizer once and cache them (thread-safe)."""
    global _tokenizer, _model, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        logger.info("[Summary] Loading summarizer model: %s", _MODEL_NAME)
        # Local import so the rest of the API can start without paying this cost
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME).to(_device)
        _model.eval()
        logger.info("[Summary] Model ready on %s", _device)


@torch.no_grad()
def summarize(
    text: str,
    min_length: int = _DEFAULT_MIN_LEN,
    max_length: int = _DEFAULT_MAX_LEN,
    num_beams: int = 4,
) -> dict:
    """
    Summarize a chunk of text using BART (seq2seq generation).

    Args:
        text: The text to summarize (transcript, article, etc.).
        min_length: Minimum length of the summary in tokens.
        max_length: Maximum length of the summary in tokens.
        num_beams: Beam search width. 4 is the BART paper default.

    Returns:
        dict with:
            - summary (str): the generated summary
            - model (str): model name used
            - input_words (int): word count of the input (before truncation)
            - input_truncated (bool): True if the input was truncated to fit
    """
    if not text or not text.strip():
        return {
            "summary": "",
            "model": _MODEL_NAME,
            "input_words": 0,
            "input_truncated": False,
        }

    _ensure_loaded()
    original_words = len(text.split())

    # Tokenize. truncation=True caps the input at the model's max length
    # (BART = 1024), so we don't need to pre-truncate by words.
    inputs = _tokenizer(
        text,
        max_length=_MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    ).to(_device)

    was_truncated = inputs["input_ids"].shape[1] >= _MAX_INPUT_TOKENS

    summary_ids = _model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        min_length=min_length,
        max_length=max_length,
        num_beams=num_beams,
        length_penalty=2.0,  # BART CNN default, discourages super-short summaries
        early_stopping=True,
        no_repeat_ngram_size=3,  # avoid repeating the same trigram
    )
    summary_text = _tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    return {
        "summary": summary_text,
        "model": _MODEL_NAME,
        "input_words": original_words,
        "input_truncated": was_truncated,
    }


def summarize_segments(segments: list, key: str = "Sentence") -> Optional[dict]:
    """
    Convenience wrapper, concatenate per-segment Sentence values and summarize.

    Args:
        segments: List of dicts (e.g. from /predict/pipeline 'segments').
        key: Which dict key holds the text. Defaults to 'Sentence'.

    Returns:
        Same dict shape as summarize(), or None if there's nothing to summarize.
    """
    if not segments:
        return None
    sentences = [str(s.get(key, "")).strip() for s in segments if s.get(key)]
    if not sentences:
        return None
    return summarize(" ".join(sentences))
