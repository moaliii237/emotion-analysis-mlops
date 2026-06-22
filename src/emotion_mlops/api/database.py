"""
Database module for logging predictions to PostgreSQL.

Usage:
    from emotion_mlops.api.database import log_prediction, init_db

    # Call once at startup
    init_db()

    # Call after each prediction
    log_prediction(text="I am happy", emotion="happiness", confidence=0.98, source="azure-ml-endpoint")
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

_conn = None


@dataclass
class PipelineMetrics:
    t_whisper: float
    t_translate: float
    t_classify: float
    t_total: float
    segment_count: int


def log_pipeline_metrics(metrics: PipelineMetrics, run_id: str = "") -> None:
    """Log pipeline timing metrics to the database."""
    conn = get_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id          SERIAL PRIMARY KEY,
                timestamp   TIMESTAMP NOT NULL,
                run_id      VARCHAR(100),
                t_whisper   FLOAT,
                t_translate FLOAT,
                t_classify  FLOAT,
                t_total     FLOAT,
                segment_count INT
            )
        """
        )
        cur.execute(
            """
            INSERT INTO pipeline_runs
            (timestamp, run_id, t_whisper, t_translate, t_classify, t_total, segment_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                datetime.utcnow(),
                run_id,
                metrics.t_whisper,
                metrics.t_translate,
                metrics.t_classify,
                metrics.t_total,
                metrics.segment_count,
            ),
        )
        cur.close()
        logger.debug("[DB] Logged pipeline run: %s", run_id)
    except Exception as e:
        logger.warning("[DB] Could not log pipeline metrics: %s", e)


def get_connection():  # pragma: no cover
    global _conn
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    try:
        import psycopg2

        if _conn is None or _conn.closed:
            _conn = psycopg2.connect(db_url, connect_timeout=5)
            _conn.autocommit = True
        return _conn
    except Exception as e:
        logger.warning("[DB] Could not connect to database: %s", e)
        return None


def init_db() -> None:
    """Create predictions table if it doesn't exist."""
    conn = get_connection()
    if not conn:
        logger.warning("[DB] DATABASE_URL not set - prediction logging disabled")
        return
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id          SERIAL PRIMARY KEY,
                timestamp   TIMESTAMP NOT NULL,
                text        TEXT NOT NULL,
                emotion     VARCHAR(50) NOT NULL,
                confidence  FLOAT NOT NULL,
                source      VARCHAR(50) NOT NULL
            )
        """
        )
        cur.close()
        logger.info("[DB] Database initialized - predictions table ready")
    except Exception as e:
        logger.warning("[DB] Could not initialize database: %s", e)


def log_prediction(
    text: str,
    emotion: str,
    confidence: float,
    source: str = "local",
) -> None:
    """
    Log a prediction to the database.

    Args:
        text: Input sentence.
        emotion: Predicted emotion label.
        confidence: Confidence score (0-1).
        source: Prediction source ('local' or 'azure-ml-endpoint').
    """
    conn = get_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (timestamp, text, emotion, confidence, source)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (datetime.utcnow(), text, emotion, confidence, source),
        )
        cur.close()
        logger.debug(
            "[DB] Logged prediction: %s -> %s (%.2f)", text[:50], emotion, confidence
        )
    except Exception as e:
        logger.warning("[DB] Could not log prediction: %s", e)


def get_recent_predictions(limit: int = 100) -> list[dict]:
    """
    Retrieve recent predictions from the database.

    Args:
        limit: Maximum number of predictions to return.

    Returns:
        List of prediction dicts with timestamp, text, emotion, confidence, source.
    """
    conn = get_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timestamp, text, emotion, confidence, source
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "timestamp": row[0].isoformat(),
                "text": row[1],
                "emotion": row[2],
                "confidence": row[3],
                "source": row[4],
            }
            for row in rows
        ]
    except Exception as e:
        logger.warning("[DB] Could not fetch predictions: %s", e)
        return []
