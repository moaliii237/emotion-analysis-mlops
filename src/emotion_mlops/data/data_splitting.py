"""
Data splitting for emotion classification.

Splits a cleaned DataFrame into train, validation, and test sets.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, validation, and test sets.

    Uses stratified splitting when all classes have enough samples,
    falls back to random splitting for small datasets.

    Args:
        df: Cleaned DataFrame with Sentence and Emotion columns.
        train_size: Fraction for training set (default 70%).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    min_class_count = df["Emotion"].value_counts().min()
    use_stratify = min_class_count >= 2

    if not use_stratify:
        logger.warning(
            "Some emotion classes have < 2 samples, using random split instead of stratified."
        )

    train, temp = train_test_split(
        df,
        test_size=1 - train_size,
        random_state=random_state,
        stratify=df["Emotion"] if use_stratify else None,
    )

    min_temp_count = temp["Emotion"].value_counts().min()
    use_stratify_temp = min_temp_count >= 2

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=random_state,
        stratify=temp["Emotion"] if use_stratify_temp else None,
    )
    logger.info(
        "Split sizes, train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test
