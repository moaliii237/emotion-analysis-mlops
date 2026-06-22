"""
Azure ML pipeline schedule for emotion classification.

Schedules the training pipeline to run weekly.

Usage:
    python src/emotion_mlops/azure/schedule.py
"""

import logging
from pathlib import Path

from azure.ai.ml.entities import JobSchedule, RecurrenceTrigger

from emotion_mlops.azure.connect import get_ml_client
from emotion_mlops.azure.pipeline import build_pipeline_job

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_FILE = _REPO_ROOT / "logs" / "azure.log"
_LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def create_schedule() -> None:
    """
    Create a weekly schedule for the emotion classification training pipeline.

    Reuses the pipeline definition from pipeline.py and attaches a
    recurrence trigger that fires every week at 08:00 UTC.
    """
    logger.info("Connecting to Azure ML workspace...")
    client = get_ml_client()

    logger.info("Building pipeline job...")
    pipeline_job = build_pipeline_job()

    schedule = JobSchedule(
        name="emotion-weekly-retrain",
        trigger=RecurrenceTrigger(
            frequency="week",
            interval=1,
            start_time="2026-05-26T08:00:00",
            time_zone="UTC",
        ),
        create_job=pipeline_job,
    )

    logger.info("Creating schedule: emotion-weekly-retrain")
    client.schedules.begin_create_or_update(schedule).result()
    logger.info("Schedule created successfully")
    print("Schedule created: emotion-weekly-retrain, runs every week at 08:00 UTC")


if __name__ == "__main__":
    create_schedule()
