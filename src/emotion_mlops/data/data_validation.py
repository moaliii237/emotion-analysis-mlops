"""
Data validation for emotion classification.

Cleans and validates emotion labels.
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


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate a DataFrame of emotion-labelled sentences.

    Strips whitespace, lowercases labels, drops nulls, and removes
    rows with unrecognised emotion labels.

    Args:
        df: DataFrame with Sentence and Emotion columns.

    Returns:
        Cleaned and validated DataFrame.
    """
    df = df.dropna(subset=["Sentence", "Emotion"])
    df["Sentence"] = df["Sentence"].str.strip()
    df["Emotion"] = df["Emotion"].str.lower().str.strip()

    before = len(df)
    df = df[df["Emotion"].isin(EMOTION_LABELS)]
    after = len(df)
    logger.info("Kept %d/%d rows with valid emotion labels", after, before)

    return df
