"""
Emotion classification module for the EmotionScope pipeline.

This module provides two usage styles:

1. ``predict_emotions()`` - Batch prediction for CSV/pipeline workflows.
2. ``EmotionClassifier`` - Single sentence prediction with confidence score, used by the API.

Both use the same fine-tuned DeBERTa model but serve different use cases.

Example:
    Batch usage::

        from emotion_mlops.models.classifier import predict_emotions
        results = predict_emotions(["I am happy", "I am sad"], "models/best_model_v3", "cpu")

    API usage::

        from emotion_mlops.models.classifier import EmotionClassifier
        clf = EmotionClassifier("models/best_model_v3")
        clf.load()
        emotion, confidence = clf.predict("I am so excited!")
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Batch prediction (for pipeline.py / CSV usage, unchanged)
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_emotions(
    texts: List[str],
    model_dir: str,
    device: str,
    max_length: int = 192,
    bs: int = 16,
) -> List[str]:
    """Predict emotions for a list of sentences (batch)."""
    print(f"[Emotion] Loading model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    id2label = {i: str(l).lower() for i, l in model.config.id2label.items()}

    preds = []
    for i in tqdm(range(0, len(texts), bs), desc="[Emotion] Predicting"):
        batch = texts[i : i + bs]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        logits = model(**enc).logits
        top = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy().tolist()
        preds.extend([id2label[t] for t in top])
    return preds


# ---------------------------------------------------------------------------
# Class-based predictor (for the API, new)
# ---------------------------------------------------------------------------


class EmotionClassifier:
    """
    Wraps the fine-tuned DeBERTa emotion classification model.

    Differences from predict_emotions() above:
      - Model is loaded once (at API startup), then stays in memory.
      - Works on a single sentence (typical API usage).
      - Returns confidence score too (not just the label).

    Usage:
        clf = EmotionClassifier("models/best_model_v3")
        clf.load()                                  # once at startup
        emotion, conf = clf.predict("Listen to me!")
    """

    def __init__(
        self,
        model_path: str | Path,
        model_version: str = "best_model_v3",
        max_length: int = 192,
    ):
        self.model_path = Path(model_path)
        self.model_version = model_version
        self.max_length = max_length

        # None until load() is called
        self.model = None
        self.tokenizer = None
        self.id2label: dict[int, str] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        """Load the model and tokenizer from disk. Called once at API startup."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model folder not found: {self.model_path}")

        print(f"[Classifier] Loading model: {self.model_path} ({self.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            int(k): str(v).lower() for k, v in self.model.config.id2label.items()
        }

        print(f"[Classifier] Model ready. Labels: {list(self.id2label.values())}")

    @property
    def is_loaded(self) -> bool:
        """Is the model loaded? Used by /health endpoint."""
        return self.model is not None and self.tokenizer is not None

    @torch.no_grad()
    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict the emotion for a single sentence.

        Returns:
            (emotion_label, confidence), confidence is a float in [0, 1].
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]

        confidence, idx = torch.max(probs, dim=-1)

        emotion = self.id2label[int(idx.item())]
        return emotion, float(confidence.item())

    @torch.no_grad()
    def predict_all(self, text: str) -> dict[str, float]:
        """
        Predict probabilities for ALL emotion labels (not just the top one).

        Used by the /predict/text/full endpoint and the `emotionscope predict-full`
        CLI command. Returns a dict mapping emotion name to probability in [0, 1].

        The probabilities sum to ~1.0 (softmax output).

        Returns
        -------
        dict[str, float]
            Example: {"happiness": 0.85, "neutral": 0.05, "anger": 0.03, ...}
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}
