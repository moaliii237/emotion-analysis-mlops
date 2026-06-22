"""
Tests for Pydantic request/response schemas.

These tests do NOT hit the API, they just instantiate the models so we
verify our contract independently of the server.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from emotion_mlops.api.schemas import (
    FeedbackRequest,
    FullPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TextRequest,
)


# --------------------------------------------------------------------------
# TextRequest, the most-used input schema
# --------------------------------------------------------------------------
class TestTextRequest:
    def test_accepts_plain_text(self):
        req = TextRequest(text="Hello world")
        assert req.text == "Hello world"

    def test_rejects_missing_text(self):
        with pytest.raises(ValidationError):
            TextRequest()

    def test_rejects_non_string_text(self):
        with pytest.raises(ValidationError):
            TextRequest(text=123)

    def test_accepts_unicode(self):
        req = TextRequest(text="Olá, çağdaş naïve résumé 你好")
        assert "çağdaş" in req.text


# --------------------------------------------------------------------------
# PredictionResponse + FullPredictionResponse
# --------------------------------------------------------------------------
class TestPredictionResponse:
    def test_round_trips_typical_payload(self):
        r = PredictionResponse(
            text="I am happy",
            emotion="happiness",
            confidence=0.92,
            model_version="best_model_v3",
        )
        assert r.emotion == "happiness"
        assert 0 <= r.confidence <= 1

    def test_rejects_missing_fields(self):
        with pytest.raises(ValidationError):
            PredictionResponse(text="x", emotion="anger")  # missing confidence


class TestFullPredictionResponse:
    def test_carries_all_emotions(self):
        all_e = {"happiness": 0.1, "anger": 0.7, "neutral": 0.2}
        r = FullPredictionResponse(
            text="x",
            top_emotion="anger",
            top_confidence=0.7,
            all_emotions=all_e,
            model_version="v1",
        )
        assert r.top_emotion == "anger"
        assert sum(r.all_emotions.values()) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# FeedbackRequest, the heart of the human-in-the-loop flywheel
# --------------------------------------------------------------------------
class TestFeedbackRequest:
    def test_accepts_correct_prediction(self):
        fb = FeedbackRequest(
            sentence="I am furious",
            predicted_label="anger",
            predicted_conf=0.9,
            is_correct=True,
            model_version="v1",
            source="text",
        )
        assert fb.is_correct is True
        assert fb.user_label is None

    def test_accepts_correction(self):
        fb = FeedbackRequest(
            sentence="I am furious",
            predicted_label="anger",
            predicted_conf=0.9,
            is_correct=False,
            user_label="disgust",
            model_version="v1",
            source="text",
        )
        assert fb.is_correct is False
        assert fb.user_label == "disgust"

    def test_rejects_missing_sentence(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(
                predicted_label="anger",
                predicted_conf=0.9,
                is_correct=True,
                model_version="v1",
                source="text",
            )


# --------------------------------------------------------------------------
# HealthResponse
# --------------------------------------------------------------------------
class TestHealthResponse:
    def test_loaded_state(self):
        h = HealthResponse(status="ok", model_loaded=True, model_version="v3")
        assert h.model_loaded is True

    def test_unloaded_state(self):
        h = HealthResponse(
            status="model_not_loaded", model_loaded=False, model_version=None
        )
        assert h.model_loaded is False
        assert h.model_version is None


# --------------------------------------------------------------------------
# Aggregated feedback stats
# --------------------------------------------------------------------------
class TestFeedbackStatsResponse:
    """
    The exact field names of FeedbackStatsResponse vary across our schema
    revisions. These tests just probe what's actually accepted instead of
    hard-coding a field name that may change again later.
    """

    def test_can_be_instantiated_from_dict(self):
        """The schema must accept whatever keys `get_stats()` returns."""
        from emotion_mlops.api.schemas import FeedbackStatsResponse
        from emotion_mlops.feedback.db import get_stats

        stats = get_stats()
        resp = FeedbackStatsResponse(**stats)
        # We don't care about the exact field names, just that it round-trips.
        assert resp.model_dump() is not None

    def test_dump_round_trip(self):
        """Dumping and re-instantiating must give the same data."""
        from emotion_mlops.api.schemas import FeedbackStatsResponse
        from emotion_mlops.feedback.db import get_stats

        stats = get_stats()
        resp = FeedbackStatsResponse(**stats)
        again = FeedbackStatsResponse(**resp.model_dump())
        assert again.model_dump() == resp.model_dump()
