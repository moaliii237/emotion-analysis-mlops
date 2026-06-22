"""
Data preprocessing script for emotion classification.

Orchestrates ingestion, validation, splitting, and saving.

Usage:
    python -m emotion_mlops.data.data_preprocessing \
        --input data/raw/goemotions_collapsed_majority.csv \
        --output data/processed/
"""

import argparse
import logging
from pathlib import Path

from emotion_mlops.data.data_ingestion import load_raw
from emotion_mlops.data.data_splitting import split_data
from emotion_mlops.data.data_validation import validate_and_clean

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_FILE = _REPO_ROOT / "logs" / "pipeline.log"
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


def preprocess(input_path: str, output_dir: str) -> None:
    """
    Full preprocessing pipeline: ingest, validate, split, save.

    Args:
        input_path: Path to raw CSV file.
        output_dir: Directory to save train/val/test CSVs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_raw(input_path)
    df = validate_and_clean(df)
    train, val, test = split_data(df)

    train.to_csv(output_path / "train.csv", index=False)
    val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)

    logger.info("Saved splits to: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)
