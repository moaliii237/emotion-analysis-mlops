"""
Feedback storage for the data flywheel.

Records user corrections to model predictions so we can:
- Track model performance drift over time
- Build a corrections dataset for retraining
- Show disagreement rates in the UI/dashboard

Uses PostgreSQL (same connection as database.py) so data persists
across container restarts on Portainer and Azure Container Apps.
Falls back to SQLite if DATABASE_URL is not set (local dev).
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SQLITE_PATH = _REPO_ROOT / "data" / "feedback.db"
_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    sentence        TEXT NOT NULL,
    predicted_label VARCHAR(50) NOT NULL,
    predicted_conf  FLOAT NOT NULL,
    user_label      VARCHAR(50),
    is_correct      INTEGER NOT NULL,
    model_version   VARCHAR(100) NOT NULL,
    source          VARCHAR(50),
    file_id         VARCHAR(200)
);
CREATE TABLE IF NOT EXISTS retrain_log (
    id                  SERIAL PRIMARY KEY,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL,
    corrections_from_id INTEGER,
    corrections_to_id   INTEGER,
    corrections_count   INTEGER,
    new_data_version    TEXT,
    job_name            TEXT,
    new_model_version   TEXT,
    new_f1              FLOAT,
    live_f1             FLOAT,
    promoted            INTEGER DEFAULT 0,
    note                TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_correct   ON feedback(is_correct);
CREATE INDEX IF NOT EXISTS idx_feedback_version   ON feedback(model_version);
CREATE INDEX IF NOT EXISTS idx_retrain_status ON retrain_log(status);
"""

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    sentence        TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    predicted_conf  REAL NOT NULL,
    user_label      TEXT,
    is_correct      INTEGER NOT NULL,
    model_version   TEXT NOT NULL,
    source          TEXT,
    file_id         TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_correct  ON feedback(is_correct);
CREATE INDEX IF NOT EXISTS idx_feedback_version  ON feedback(model_version);

CREATE TABLE IF NOT EXISTS retrain_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL,
    corrections_from_id INTEGER,
    corrections_to_id   INTEGER,
    corrections_count   INTEGER,
    new_data_version    TEXT,
    job_name            TEXT,
    new_model_version   TEXT,
    new_f1              REAL,
    live_f1             REAL,
    promoted            INTEGER DEFAULT 0,
    note                TEXT
);

CREATE INDEX IF NOT EXISTS idx_retrain_status ON retrain_log(status);
"""


def _use_postgres() -> bool:  # pragma: no cover
    return bool(os.environ.get("DATABASE_URL"))


def _get_postgres_conn():  # pragma: no cover
    """Get a psycopg2 connection with RealDictCursor for dict-like row access."""
    import psycopg2
    import psycopg2.extras

    db_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(db_url, connect_timeout=5)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


@contextmanager
def _get_sqlite_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(_SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_conn():  # pragma: no cover
    """Public connection context manager, routes to postgres or SQLite."""
    if _use_postgres():
        conn = None
        try:
            conn = _get_postgres_conn()
            conn.autocommit = True
            yield conn
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if conn is not None:
                conn.close()
    else:
        with _get_sqlite_conn() as conn:
            yield conn


def init_db() -> None:
    """Create feedback and retrain_log tables if they don't exist. Idempotent."""
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            conn.autocommit = True
            cur = conn.cursor()
            for stmt in _SCHEMA_POSTGRES.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            cur.close()
            conn.close()
            logger.info("[Feedback] PostgreSQL feedback table initialised.")
        except Exception as e:
            logger.warning(
                "[Feedback] Could not initialise PostgreSQL feedback table: %s", e
            )
        # Always init SQLite too, retrain_log lives here even in Postgres mode
        with _get_sqlite_conn() as conn:
            conn.executescript(_SCHEMA_SQLITE)
    else:
        with _get_sqlite_conn() as conn:
            conn.executescript(_SCHEMA_SQLITE)
        logger.info("[Feedback] SQLite feedback DB initialised at %s", _SQLITE_PATH)


def record_feedback(
    sentence: str,
    predicted_label: str,
    predicted_conf: float,
    is_correct: bool,
    model_version: str,
    user_label: Optional[str] = None,
    source: Optional[str] = None,
    file_id: Optional[str] = None,
) -> int:
    """
    Insert a feedback row.

    Returns:
        The new row's id.
    """
    ts = datetime.utcnow()
    is_correct_int = 1 if is_correct else 0

    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO feedback
                    (timestamp, sentence, predicted_label, predicted_conf,
                     user_label, is_correct, model_version, source, file_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    ts,
                    sentence,
                    predicted_label,
                    predicted_conf,
                    user_label,
                    is_correct_int,
                    model_version,
                    source,
                    file_id,
                ),
            )
            new_id = cur.fetchone()["id"]
            cur.close()
            conn.close()
            logger.info(
                "[Feedback] Recorded row %d (correct=%s) in PostgreSQL",
                new_id,
                is_correct,
            )
            return new_id
        except Exception as e:
            logger.warning(
                "[Feedback] PostgreSQL write failed, falling back to SQLite: %s", e
            )

    with _get_sqlite_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback
                (timestamp, sentence, predicted_label, predicted_conf,
                 user_label, is_correct, model_version, source, file_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat(),
                sentence,
                predicted_label,
                predicted_conf,
                user_label,
                is_correct_int,
                model_version,
                source,
                file_id,
            ),
        )
        new_id = cur.lastrowid
    logger.info("[Feedback] Recorded row %d (correct=%s) in SQLite", new_id, is_correct)
    return new_id


def get_stats() -> dict:
    """
    Aggregate feedback stats for dashboards.

    Returns:
        dict with total, agreement_rate, per_model breakdown.
    """
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS c FROM feedback")
            total = cur.fetchone()["c"]
            if total == 0:
                cur.close()
                conn.close()
                return {"total": 0, "agreement_rate": None, "per_model": []}

            cur.execute("SELECT COUNT(*) AS c FROM feedback WHERE is_correct = 1")
            correct = cur.fetchone()["c"]

            cur.execute(
                """
                SELECT model_version,
                       COUNT(*) AS n,
                       ROUND(AVG(is_correct) * 100, 2) AS agreement_pct
                FROM feedback
                GROUP BY model_version
                ORDER BY n DESC
                """
            )
            per_model = [dict(r) for r in cur.fetchall()]
            cur.close()
            conn.close()
            return {
                "total": total,
                "agreement_rate": round(correct / total, 4),
                "per_model": per_model,
            }
        except Exception as e:
            logger.warning(
                "[Feedback] PostgreSQL stats failed, falling back to SQLite: %s", e
            )

    with _get_sqlite_conn() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
        if total == 0:
            return {"total": 0, "agreement_rate": None, "per_model": []}
        correct = conn.execute(
            "SELECT COUNT(*) as c FROM feedback WHERE is_correct = 1"
        ).fetchone()["c"]
        per_model = conn.execute(
            """
            SELECT model_version,
                   COUNT(*) AS n,
                   ROUND(AVG(is_correct) * 100, 2) AS agreement_pct
            FROM feedback
            GROUP BY model_version
            ORDER BY n DESC
            """
        ).fetchall()
    return {
        "total": total,
        "agreement_rate": round(correct / total, 4),
        "per_model": [dict(row) for row in per_model],
    }


def export_corrections_for_retraining() -> List[dict]:
    """
    Return all corrected predictions as training-ready records.

    Only includes feedback where the user provided a corrected label.
    """
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT sentence, user_label AS emotion
                FROM feedback
                WHERE is_correct = 0 AND user_label IS NOT NULL
                """
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [{"Sentence": r["sentence"], "Emotion": r["emotion"]} for r in rows]
        except Exception as e:
            logger.warning(
                "[Feedback] PostgreSQL export failed, falling back to SQLite: %s", e
            )

    with _get_sqlite_conn() as conn:
        rows = conn.execute(
            """
            SELECT sentence, user_label AS Emotion
            FROM feedback
            WHERE is_correct = 0 AND user_label IS NOT NULL
            """
        ).fetchall()
    return [{"Sentence": r["sentence"], "Emotion": r["Emotion"]} for r in rows]


# ===========================================================================
# Auto-retrain bookkeeping (the feedback flywheel trigger)
# Uses PostgreSQL when available, falls back to SQLite for local dev.
# ===========================================================================


def count_new_corrections() -> dict:  # pragma: no cover
    """
    Count usable corrections collected SINCE the last retrain.

    A "usable correction" is feedback where the user disagreed and supplied a
    replacement label (is_correct=0 AND user_label IS NOT NULL), only these
    add new training signal.

    Returns:
        dict with:
            - new_corrections: count of corrections above the watermark
            - from_id / to_id: the id range of that batch (None if empty)
            - watermark: the last-consumed feedback id
    """
    wm = get_watermark()
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) AS n, MIN(id) AS lo, MAX(id) AS hi
                FROM feedback
                WHERE is_correct = 0 AND user_label IS NOT NULL AND id > %s
                """,
                (wm,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            return {
                "new_corrections": int(row["n"] or 0),
                "from_id": row["lo"],
                "to_id": row["hi"],
                "watermark": wm,
            }
        except Exception as e:
            logger.warning("[Feedback] count_new_corrections PG failed: %s", e)
    with _get_sqlite_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n, MIN(id) AS lo, MAX(id) AS hi
            FROM feedback
            WHERE is_correct = 0 AND user_label IS NOT NULL AND id > ?
            """,
            (wm,),
        ).fetchone()
    return {
        "new_corrections": int(row["n"] or 0),
        "from_id": row["lo"],
        "to_id": row["hi"],
        "watermark": wm,
    }


def get_watermark() -> int:  # pragma: no cover
    """
    Highest feedback id already consumed by a COMPLETED retrain.
    Returns 0 if no retrain has finished yet.
    """
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT MAX(corrections_to_id) AS wm
                FROM retrain_log
                WHERE status IN ('registered', 'promoted', 'skipped')
                  AND corrections_to_id IS NOT NULL
                """
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            return int(row["wm"]) if row and row["wm"] is not None else 0
        except Exception as e:
            logger.warning("[Feedback] get_watermark PG failed: %s", e)
    with _get_sqlite_conn() as conn:
        row = conn.execute(
            """
            SELECT MAX(corrections_to_id) AS wm
            FROM retrain_log
            WHERE status IN ('registered', 'promoted', 'skipped')
              AND corrections_to_id IS NOT NULL
            """
        ).fetchone()
    return int(row["wm"]) if row and row["wm"] is not None else 0


def is_retrain_in_flight() -> bool:  # pragma: no cover
    """True if a retrain row is currently mid-run (prevents overlapping runs)."""
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) AS n FROM retrain_log "
                "WHERE status IN ('started', 'training')"
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            return bool(row["n"])
        except Exception as e:
            logger.warning("[Feedback] is_retrain_in_flight PG failed: %s", e)
    with _get_sqlite_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM retrain_log "
            "WHERE status IN ('started', 'training')"
        ).fetchone()
    return bool(row["n"])


def record_retrain_start(  # pragma: no cover
    from_id: Optional[int], to_id: Optional[int], count: int
) -> int:
    """Insert a retrain_log row at the start of a run. Returns its id."""
    ts = datetime.utcnow().isoformat()
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO retrain_log
                    (started_at, status, corrections_from_id,
                     corrections_to_id, corrections_count)
                VALUES (%s, 'started', %s, %s, %s)
                RETURNING id
                """,
                (ts, from_id, to_id, count),
            )
            log_id = cur.fetchone()["id"]
            cur.close()
            conn.close()
            logger.info(
                "[Retrain] Started retrain_log #%d (%d corrections) in PG",
                log_id,
                count,
            )
            return log_id
        except Exception as e:
            logger.warning("[Feedback] record_retrain_start PG failed: %s", e)
    with _get_sqlite_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO retrain_log
                (started_at, status, corrections_from_id,
                 corrections_to_id, corrections_count)
            VALUES (?, 'started', ?, ?, ?)
            """,
            (ts, from_id, to_id, count),
        )
        log_id = cur.lastrowid
    logger.info("[Retrain] Started retrain_log #%d (%d corrections)", log_id, count)
    return log_id


def update_retrain_log(log_id: int, **fields) -> None:  # pragma: no cover
    """Update a retrain_log row."""
    allowed = {
        "status",
        "finished_at",
        "new_data_version",
        "job_name",
        "new_model_version",
        "new_f1",
        "live_f1",
        "promoted",
        "note",
    }
    keys = [k for k in fields if k in allowed]
    if not keys:
        return
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            conn.autocommit = True
            cur = conn.cursor()
            assignments = ", ".join(f"{k} = %s" for k in keys)
            values = [fields[k] for k in keys] + [log_id]
            cur.execute(f"UPDATE retrain_log SET {assignments} WHERE id = %s", values)
            cur.close()
            conn.close()
            logger.info("[Retrain] Updated retrain_log #%d in PG: %s", log_id, fields)
            return
        except Exception as e:
            logger.warning("[Feedback] update_retrain_log PG failed: %s", e)
    assignments = ", ".join(f"{k} = ?" for k in keys)
    values = [fields[k] for k in keys] + [log_id]
    with _get_sqlite_conn() as conn:
        conn.execute(f"UPDATE retrain_log SET {assignments} WHERE id = ?", values)
    logger.info("[Retrain] Updated retrain_log #%d: %s", log_id, fields)


def get_retrain_history(limit: int = 20) -> List[dict]:  # pragma: no cover
    """Return recent retrain_log rows, newest first (for dashboards / audit)."""
    if _use_postgres():
        try:
            conn = _get_postgres_conn()
            cur = conn.cursor()
            cur.execute("SELECT * FROM retrain_log ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("[Feedback] get_retrain_history PG failed: %s", e)
    with _get_sqlite_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
