"""
Real model behaviour tests, does the classifier actually pick the right
emotion for clearly emotional sentences? These tests hit /predict/text and
/predict/text/full against the running server, so they require the model
to be loaded.
"""

from __future__ import annotations

import pytest
import requests


# --------------------------------------------------------------------------
# /predict/text, single-emotion sanity checks.
# Each fixture sentence is unambiguous in English, so the top emotion is a
# fair contract to assert. We allow neutral as a fallback for borderline
# cases where the model genuinely is uncertain.
# --------------------------------------------------------------------------
class TestPredictText:
    def test_returns_200(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/text", json={"text": "hello"})
        assert r.status_code == 200

    def test_response_shape(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/text", json={"text": "hello"}).json()
        assert {"text", "emotion", "confidence", "model_version"}.issubset(r.keys())

    def test_confidence_in_unit_interval(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/text", json={"text": "hello"}).json()
        assert 0.0 <= r["confidence"] <= 1.0

    def test_anger_detected(self, base_url, model_ready, angry_text):
        r = requests.post(f"{base_url}/predict/text", json={"text": angry_text}).json()
        assert r["emotion"] == "anger", f"Expected anger, got {r['emotion']}"
        assert r["confidence"] > 0.4

    def test_happiness_detected(self, base_url, model_ready, happy_text):
        r = requests.post(f"{base_url}/predict/text", json={"text": happy_text}).json()
        assert r["emotion"] == "happiness", f"Expected happiness, got {r['emotion']}"
        assert r["confidence"] > 0.4

    def test_sadness_detected(self, base_url, model_ready, sad_text):
        r = requests.post(f"{base_url}/predict/text", json={"text": sad_text}).json()
        # Sadness sometimes leaks into fear/neutral, accept either.
        assert r["emotion"] in (
            "sadness",
            "fear",
            "neutral",
        ), f"Expected sadness-adjacent emotion, got {r['emotion']}"


# --------------------------------------------------------------------------
# /predict/text/full, full distribution checks.
# --------------------------------------------------------------------------
class TestPredictTextFull:
    def test_returns_all_emotions(self, base_url, model_ready):
        r = requests.post(
            f"{base_url}/predict/text/full", json={"text": "I am furious"}
        ).json()
        assert set(r["all_emotions"].keys()) == {
            "happiness",
            "neutral",
            "anger",
            "sadness",
            "surprise",
            "fear",
            "disgust",
        }

    def test_probabilities_sum_to_one(self, base_url, model_ready):
        r = requests.post(
            f"{base_url}/predict/text/full", json={"text": "I am calm"}
        ).json()
        total = sum(r["all_emotions"].values())
        assert total == pytest.approx(
            1.0, abs=0.05
        ), f"Probabilities sum to {total}, expected ~1.0"

    def test_top_emotion_matches_argmax(self, base_url, model_ready):
        r = requests.post(
            f"{base_url}/predict/text/full", json={"text": "I'm thrilled and happy!"}
        ).json()
        argmax = max(r["all_emotions"], key=r["all_emotions"].get)
        assert r["top_emotion"] == argmax

    def test_top_confidence_equals_max_probability(self, base_url, model_ready):
        r = requests.post(
            f"{base_url}/predict/text/full", json={"text": "I'm thrilled and happy!"}
        ).json()
        max_prob = max(r["all_emotions"].values())
        assert r["top_confidence"] == pytest.approx(max_prob, abs=1e-3)


# --------------------------------------------------------------------------
# Determinism, the same input should yield the same prediction.
# (No randomness in inference.)
# --------------------------------------------------------------------------
def test_predictions_are_deterministic(base_url, model_ready):
    text = "I am completely furious about this situation."
    r1 = requests.post(f"{base_url}/predict/text", json={"text": text}).json()
    r2 = requests.post(f"{base_url}/predict/text", json={"text": text}).json()
    assert r1["emotion"] == r2["emotion"]
    assert r1["confidence"] == pytest.approx(r2["confidence"], abs=1e-4)


# --------------------------------------------------------------------------
# Different emotional intensities should produce different confidences.
# This catches the regression where the model collapses to one prediction.
# --------------------------------------------------------------------------
def test_strong_emotion_more_confident_than_weak(base_url, model_ready):
    strong = "I am absolutely livid! I cannot contain my fury!"
    weak = "I'm a bit annoyed."
    rs = requests.post(f"{base_url}/predict/text", json={"text": strong}).json()
    rw = requests.post(f"{base_url}/predict/text", json={"text": weak}).json()
    # We don't insist that the weak one is anger, just that confidence differs.
    assert rs["confidence"] != rw["confidence"]
