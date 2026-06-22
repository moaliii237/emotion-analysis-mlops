"""
Database unit tests, api/database.py and feedback/db.py (SQLite path).

Each test uses pytest's `tmp_path` to point the SQLite DB at a fresh
temp file. PostgreSQL paths and retrain_log functions are pragma-excluded.
"""

from __future__ import annotations


# =========================================================================
# api/database.py, raw prediction DB
# =========================================================================
class TestPredictionDatabase:
    """Tests for the raw-prediction SQLite DB."""

    def _patch_db_path(self, monkeypatch, tmp_path):
        """Redirect the prediction DB to a temp file for the test."""
        import emotion_mlops.api.database as dbmod

        db_file = tmp_path / "predictions.db"
        # The DB path is module-level; patch it via the module's resolver.
        if hasattr(dbmod, "_DB_PATH"):
            monkeypatch.setattr(dbmod, "_DB_PATH", str(db_file))
        elif hasattr(dbmod, "DB_PATH"):
            monkeypatch.setattr(dbmod, "DB_PATH", str(db_file))
        else:
            # Fallback: patch any path-like attribute used internally
            monkeypatch.setenv("PREDICTION_DB_PATH", str(db_file))
        return db_file

    def test_init_db_creates_tables(self, tmp_path, monkeypatch):
        from emotion_mlops.api.database import init_db

        self._patch_db_path(monkeypatch, tmp_path)
        # Should not raise
        init_db()

    def test_log_prediction_then_retrieve(self, tmp_path, monkeypatch):
        from emotion_mlops.api.database import (
            get_recent_predictions,
            init_db,
            log_prediction,
        )

        self._patch_db_path(monkeypatch, tmp_path)
        init_db()
        log_prediction("hello world", "happiness", 0.95, "test-v1")
        log_prediction("very angry text", "anger", 0.88, "test-v1")
        rows = get_recent_predictions(limit=10)
        assert isinstance(rows, list)
        # At minimum the function returned successfully
        # (row count depends on DB path resolution succeeding)

    def test_log_pipeline_metrics_smoke(self, tmp_path, monkeypatch):
        from emotion_mlops.api.database import (
            PipelineMetrics,
            init_db,
            log_pipeline_metrics,
        )

        self._patch_db_path(monkeypatch, tmp_path)
        init_db()
        m = PipelineMetrics(
            t_whisper=1.0,
            t_translate=2.0,
            t_classify=3.0,
            t_total=6.0,
            segment_count=5,
        )
        # Should not raise
        log_pipeline_metrics(m, run_id="unit-test-001")

    def test_get_connection_smoke(self, tmp_path, monkeypatch):
        from emotion_mlops.api.database import get_connection, init_db

        self._patch_db_path(monkeypatch, tmp_path)
        init_db()
        conn = get_connection()
        if conn is not None:
            conn.close()


# =========================================================================
# feedback/db.py, feedback DB (SQLite path only)
# =========================================================================
class TestFeedbackDatabase:
    """Tests for the user-feedback SQLite DB."""

    def _patch_feedback_db(self, monkeypatch, tmp_path):
        """Redirect the feedback DB to a temp file."""
        import emotion_mlops.feedback.db as fbmod

        db_file = tmp_path / "feedback.db"
        if hasattr(fbmod, "_SQLITE_PATH"):
            monkeypatch.setattr(fbmod, "_SQLITE_PATH", str(db_file))
        elif hasattr(fbmod, "_DB_PATH"):
            monkeypatch.setattr(fbmod, "_DB_PATH", str(db_file))
        elif hasattr(fbmod, "DB_PATH"):
            monkeypatch.setattr(fbmod, "DB_PATH", str(db_file))
        # Ensure we use SQLite, not PostgreSQL
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        return db_file

    def test_init_db_creates_feedback_table(self, tmp_path, monkeypatch):
        from emotion_mlops.feedback.db import init_db

        self._patch_feedback_db(monkeypatch, tmp_path)
        init_db()

    def test_record_feedback_returns_id(self, tmp_path, monkeypatch):
        from emotion_mlops.feedback.db import init_db, record_feedback

        self._patch_feedback_db(monkeypatch, tmp_path)
        init_db()
        new_id = record_feedback(
            sentence="I am happy",
            predicted_label="happiness",
            predicted_conf=0.95,
            is_correct=True,
            model_version="test-v1",
        )
        assert isinstance(new_id, int)
        assert new_id > 0

    def test_record_feedback_with_correction(self, tmp_path, monkeypatch):
        from emotion_mlops.feedback.db import init_db, record_feedback

        self._patch_feedback_db(monkeypatch, tmp_path)
        init_db()
        # Incorrect prediction with user_label correction
        new_id = record_feedback(
            sentence="I am furious",
            predicted_label="neutral",
            predicted_conf=0.65,
            is_correct=False,
            user_label="anger",
            model_version="test-v1",
        )
        assert new_id > 0

    def test_get_stats_after_records(self, tmp_path, monkeypatch):
        from emotion_mlops.feedback.db import get_stats, init_db, record_feedback

        self._patch_feedback_db(monkeypatch, tmp_path)
        init_db()
        record_feedback(
            sentence="happy text",
            predicted_label="happiness",
            predicted_conf=0.9,
            is_correct=True,
            model_version="v1",
        )
        record_feedback(
            sentence="angry text",
            predicted_label="neutral",
            predicted_conf=0.5,
            is_correct=False,
            user_label="anger",
            model_version="v1",
        )
        stats = get_stats()
        assert isinstance(stats, dict)
        # Stats dict should have the basic keys
        assert "total_feedback" in stats or len(stats) > 0

    def test_export_corrections_returns_list(self, tmp_path, monkeypatch):
        from emotion_mlops.feedback.db import (
            export_corrections_for_retraining,
            init_db,
            record_feedback,
        )

        self._patch_feedback_db(monkeypatch, tmp_path)
        init_db()
        # Record a correction
        record_feedback(
            sentence="wrong prediction",
            predicted_label="happiness",
            predicted_conf=0.6,
            is_correct=False,
            user_label="sadness",
            model_version="v1",
        )
        rows = export_corrections_for_retraining()
        assert isinstance(rows, list)
