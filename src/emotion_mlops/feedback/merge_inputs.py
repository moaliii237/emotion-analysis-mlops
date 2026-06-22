"""
In-job merge of base training data with user feedback corrections.

Runs INSIDE the Azure ML retraining job. Azure ML mounts both the base
data asset (emotion-raw) and the feedback corrections asset as local files,
so this step only has to read the two CSVs, keep the Sentence/Emotion
columns, concatenate them (base first, corrections appended last) and write
one merged CSV for the normal preprocessing step.

Appending corrections last and de-duplicating on the sentence keeps the
human-corrected label when the same sentence also exists in the base data.

Usage:
    python -m emotion_mlops.feedback.merge_inputs \
        --base <base.csv> \
        --corrections <corrections.csv> \
        --output data/raw/merged.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = ["Sentence", "Emotion"]


def _read_slim(csv_path: str, source: str) -> pd.DataFrame:
    """
    Read a CSV and return just the Sentence/Emotion columns.

    Accepts the same column aliases as data_ingestion.load_raw so either a
    raw dataset export or a corrections export works unchanged.

    Args:
        csv_path: Path to the CSV file.
        source: Human-readable label used in error messages ("base"/"corrections").

    Returns:
        DataFrame with exactly the Sentence and Emotion columns.

    Raises:
        ValueError: If no sentence or emotion column can be found.
    """
    df = pd.read_csv(csv_path)
    col_map = {c.lower(): c for c in df.columns}
    sentence_col = (
        col_map.get("sentence") or col_map.get("text") or col_map.get("utterance")
    )
    emotion_col = col_map.get("emotion") or col_map.get("label")
    if not sentence_col or not emotion_col:
        raise ValueError(
            f"{source} CSV missing Sentence/Emotion columns. Got: {list(df.columns)}"
        )
    slim = df.rename(columns={sentence_col: "Sentence", emotion_col: "Emotion"})
    return slim[_COLUMNS].copy()


def merge(base_path: str, corrections_path: str, output_path: str) -> int:
    """
    Concatenate base training data with feedback corrections.

    Corrections are appended last and exact duplicate sentences are de-duped
    keeping the last occurrence, so a human correction overrides the original
    label for the same sentence.

    Args:
        base_path: Path to the base training CSV (e.g. emotion-raw).
        corrections_path: Path to the feedback corrections CSV.
        output_path: Where to write the merged CSV.

    Returns:
        Number of rows in the merged CSV.
    """
    base = _read_slim(base_path, "base")
    corrections = _read_slim(corrections_path, "corrections")

    merged = pd.concat([base, corrections], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset="Sentence", keep="last").reset_index(
        drop=True
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    logger.info(
        "Merged base=%d + corrections=%d -> %d rows "
        "(%d duplicate sentence(s) dropped) -> %s",
        len(base),
        len(corrections),
        len(merged),
        before - len(merged),
        output_path,
    )
    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge base training data with feedback corrections."
    )
    parser.add_argument(
        "--base", required=True, help="Base training CSV (emotion-raw)."
    )
    parser.add_argument(
        "--corrections", required=True, help="Feedback corrections CSV."
    )
    parser.add_argument(
        "--output", required=True, help="Output path for the merged CSV."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    merge(args.base, args.corrections, args.output)


if __name__ == "__main__":
    main()
