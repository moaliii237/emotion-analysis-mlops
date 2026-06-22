"""
Model drift detection for emotion classification.

Monitors prediction distribution over time to detect
when model performance may be degrading.
"""

import logging
from collections import Counter

import mlflow

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]

# Expected distribution from training data (approximate)
BASELINE_DISTRIBUTION = {
    "happiness": 0.45,
    "neutral": 0.28,
    "anger": 0.16,
    "sadness": 0.05,
    "surprise": 0.04,
    "fear": 0.01,
    "disgust": 0.01,
}


def compute_distribution(predictions: list[str]) -> dict[str, float]:
    """
    Compute emotion distribution from a list of predictions.

    Args:
        predictions: List of predicted emotion labels.

    Returns:
        Dictionary mapping emotion labels to proportions.
    """
    counts = Counter(predictions)
    total = len(predictions)
    return {e: counts.get(e, 0) / total for e in EMOTION_LABELS}


def detect_drift(
    predictions: list[str],
    threshold: float = 0.15,
) -> dict[str, float]:
    """
    Compare current prediction distribution to baseline.

    Flags emotions where the distribution has shifted by more
    than the threshold compared to training data baseline.

    Args:
        predictions: List of recent predicted emotion labels.
        threshold: Maximum allowed deviation from baseline.

    Returns:
        Dictionary of drifted emotions and their deviation scores.
    """
    current = compute_distribution(predictions)
    drifted = {}

    for emotion in EMOTION_LABELS:
        deviation = abs(current.get(emotion, 0) - BASELINE_DISTRIBUTION[emotion])
        if deviation > threshold:
            drifted[emotion] = round(deviation, 4)
            logger.warning(
                "Drift detected for '%s': deviation=%.4f (threshold=%.2f)",
                emotion,
                deviation,
                threshold,
            )

    if not drifted:
        logger.info("No drift detected.")

    return drifted


def log_drift_metrics(predictions: list[str]) -> None:
    """
    Log drift metrics to MLflow.

    Args:
        predictions: List of recent predicted emotion labels.
    """
    current = compute_distribution(predictions)
    drifted = detect_drift(predictions)

    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metrics({f"dist_{e}": v for e, v in current.items()})
        mlflow.log_metrics({f"drift_{e}": v for e, v in drifted.items()})
        mlflow.log_metric("drift_detected", int(len(drifted) > 0))
