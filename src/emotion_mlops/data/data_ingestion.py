"""
Data ingestion for emotion classification.

Loads raw CSV files and normalises column names.
"""

import logging

import pandas as pd

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


def load_raw(input_path: str) -> pd.DataFrame:
    """
    Load raw CSV and normalise column names to Sentence and Emotion.

    Args:
        input_path: Path to raw CSV file.

    Returns:
        DataFrame with Sentence and Emotion columns.

    Raises:
        ValueError: If required columns are missing.
    """
    logger.info("Loading data from: %s", input_path)
    df = pd.read_csv(input_path)

    col_map = {c.lower(): c for c in df.columns}
    sentence_col = (
        col_map.get("sentence") or col_map.get("text") or col_map.get("utterance")
    )
    emotion_col = col_map.get("emotion") or col_map.get("label")

    if not sentence_col:
        raise ValueError(f"No Sentence/Text/Utterance column. Got: {list(df.columns)}")
    if not emotion_col:
        raise ValueError(f"No Emotion/Label column. Got: {list(df.columns)}")

    df = df.rename(columns={sentence_col: "Sentence", emotion_col: "Emotion"})
    return df[["Sentence", "Emotion"]].copy()
