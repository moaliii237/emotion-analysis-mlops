"""Text prediction unit tests with FakeClassifier."""

import pytest


class TestPredictText:
    def test_returns_200(self, client):
        r = client.post("/predict/text", json={"text": "hello"})
        assert r.status_code == 200

    def test_response_shape(self, client):
        r = client.post("/predict/text", json={"text": "hello"}).json()
        assert {"text", "emotion", "confidence", "model_version"}.issubset(r.keys())

    def test_confidence_in_unit_interval(self, client):
        r = client.post("/predict/text", json={"text": "anything"}).json()
        assert 0.0 <= r["confidence"] <= 1.0

    def test_anger_keyword_yields_anger(self, client, angry_text):
        r = client.post("/predict/text", json={"text": angry_text}).json()
        assert r["emotion"] == "anger"

    def test_happiness_keyword_yields_happiness(self, client, happy_text):
        r = client.post("/predict/text", json={"text": happy_text}).json()
        assert r["emotion"] == "happiness"

    def test_neutral_fallback(self, client):
        r = client.post("/predict/text", json={"text": "The cat sat on the mat"}).json()
        assert r["emotion"] == "neutral"


class TestPredictTextFull:
    def test_returns_all_emotions(self, client):
        r = client.post("/predict/text/full", json={"text": "I am furious"}).json()
        assert set(r["all_emotions"].keys()) == {
            "happiness",
            "neutral",
            "anger",
            "sadness",
            "surprise",
            "fear",
            "disgust",
        }

    def test_probabilities_sum_to_one(self, client):
        r = client.post("/predict/text/full", json={"text": "I am calm"}).json()
        assert sum(r["all_emotions"].values()) == pytest.approx(1.0, abs=0.05)

    def test_top_emotion_matches_argmax(self, client):
        r = client.post("/predict/text/full", json={"text": "I am furious"}).json()
        argmax = max(r["all_emotions"], key=r["all_emotions"].get)
        assert r["top_emotion"] == argmax


def test_predictions_are_deterministic(client):
    r1 = client.post("/predict/text", json={"text": "I am furious"}).json()
    r2 = client.post("/predict/text", json={"text": "I am furious"}).json()
    assert r1["emotion"] == r2["emotion"]
    assert r1["confidence"] == pytest.approx(r2["confidence"], abs=1e-4)


def test_different_inputs_produce_different_emotions(client):
    angry = client.post("/predict/text", json={"text": "I am furious"}).json()
    happy = client.post("/predict/text", json={"text": "I am happy"}).json()
    assert angry["emotion"] != happy["emotion"]
