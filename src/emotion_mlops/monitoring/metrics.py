"""
Operational monitoring for the emotion classification pipeline.

Tracks technical metrics: latency, request counts, error rates.
Logged to MLflow and visible in the Streamlit dashboard.
"""

import logging
from dataclasses import dataclass

import mlflow

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """
    Container for pipeline run metrics.

    Attributes:
        t_whisper: Whisper transcription time in seconds.
        t_translate: MarianMT translation time in seconds.
        t_classify: DeBERTa classification time in seconds.
        t_total: Total pipeline time in seconds.
        segment_count: Number of segments processed.
        error: Error message if pipeline failed.
    """

    t_whisper: float = 0.0
    t_translate: float = 0.0
    t_classify: float = 0.0
    t_total: float = 0.0
    segment_count: int = 0
    error: str = ""


def log_pipeline_metrics(metrics: PipelineMetrics, run_id: str = "") -> None:
    """
    Log pipeline operational metrics to MLflow.

    Args:
        metrics: PipelineMetrics instance with timing and count data.
        run_id: Optional identifier for the pipeline run.
    """
    with mlflow.start_run(run_name=f"pipeline_{run_id}"):
        mlflow.log_metrics(
            {
                "t_whisper": metrics.t_whisper,
                "t_translate": metrics.t_translate,
                "t_classify": metrics.t_classify,
                "t_total": metrics.t_total,
                "segment_count": metrics.segment_count,
            }
        )
        logger.info(
            "Logged pipeline metrics: total=%.2fs segments=%d",
            metrics.t_total,
            metrics.segment_count,
        )
