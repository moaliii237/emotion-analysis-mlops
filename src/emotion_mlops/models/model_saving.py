"""Utilities for saving and loading trained models."""

import logging
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def save_model(model, tokenizer, output_dir: str) -> None:
    """
    Save model and tokenizer to disk.

    Args:
        model: Trained HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        output_dir: Directory to save to.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to: %s", output_dir)


def load_model(model_dir: str):
    """
    Load model and tokenizer from disk.

    Args:
        model_dir: Directory containing saved model.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    logger.info("Model loaded from: %s", model_dir)
    return model, tokenizer
