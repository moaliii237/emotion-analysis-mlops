"""
Unit-test fixtures, FastAPI TestClient with all heavy components mocked.

No live server, no real DeBERTa, no Groq API calls, no Azure ML.
Runs in ~10 seconds and is the suite executed by GitHub Actions on every push / PR.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Make `src/` importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


EMOTIONS = ["happiness", "neutral", "anger", "sadness", "surprise", "fear", "disgust"]


# ---- Fake classifier -----------------------------------------------------
class FakeClassifier:
    is_loaded = True
    model_version = "fake-v1"
    model_path = "/fake/path"

    KEYWORDS = {
        "anger": ["furious", "angry", "outrage", "livid", "fury", "pay for"],
        "happiness": [
            "happy",
            "joy",
            "wonderful",
            "amazing",
            "thrilled",
            "love",
            "smile",
        ],
        "sadness": ["sad", "down", "hopeless", "miss", "lonely", "cry", "terrible"],
        "fear": ["scared", "afraid", "terror", "fear", "panic"],
        "surprise": ["surprised", "shocked", "unexpected", "wow", "amazed"],
        "disgust": ["disgust", "gross", "horrible", "revolting"],
    }

    def predict(self, text: str) -> tuple[str, float]:
        text_low = (text or "").lower()
        for emotion, words in self.KEYWORDS.items():
            if any(w in text_low for w in words):
                return (emotion, 0.90)
        return ("neutral", 0.70)

    def predict_all(self, text: str) -> dict:
        emotion, conf = self.predict(text)
        d = {e: 0.04 for e in EMOTIONS}
        d[emotion] = conf
        total = sum(d.values())
        return {e: v / total for e, v in d.items()}

    def load(self):
        return


def _fake_generate_summary(
    segments, cast=None, scenes=None, score=None, persuasion=None
):
    return {
        "summary": "This content blends multiple emotional tones and resonates across the spectrum.",
        "ai_generated": False,
    }


def _fake_generate_title(segments, cast=None, dominant_emotion=None, fallback="Show"):
    return fallback


# ---- Fake Azure ML endpoint ---------------------------------------------
# /predict/text in production hits an Azure ML Online Endpoint via HTTP.
# In tests, we intercept the HTTP call and return a response that mirrors
# the FakeClassifier logic, so tests stay deterministic and offline.
def _fake_azure_post(url, json=None, headers=None, timeout=None, **kwargs):
    """Simulate a successful response from the Azure ML scoring endpoint."""
    text = (json or {}).get("sentence", "")
    fake = FakeClassifier()
    emotion, conf = fake.predict(text)
    all_emotions = fake.predict_all(text)

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "emotion": emotion,
                    "confidence": float(conf),
                    "all_emotions": all_emotions,
                }
            ]

    return FakeResp()


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient with classifier + Groq + Azure ML mocked."""
    from emotion_mlops.api import app as app_module

    app_module.classifier = FakeClassifier()
    app_module.whisper_state["model"] = object()
    app_module.whisper_state["device"] = "cpu"
    app_module.mt_state["model"] = object()
    app_module.mt_state["tokenizer"] = object()

    patches = [
        # Mock Groq-backed helpers so we don't hit any external LLM
        patch(
            "emotion_mlops.analysis.summary.generate_summary",
            side_effect=_fake_generate_summary,
        ),
        patch(
            "emotion_mlops.analysis.autotitle.generate_title",
            side_effect=_fake_generate_title,
        ),
        # Mock the Azure ML Online Endpoint:
        #   - _get_endpoint_url_and_key returns fake creds (so no RuntimeError)
        #   - requests.post intercepts the actual HTTP call
        patch(
            "emotion_mlops.api.app._get_endpoint_url_and_key",
            return_value=("http://fake-azure.local/score", "fake-key"),
        ),
        patch("requests.post", side_effect=_fake_azure_post),
    ]
    for p in patches:
        p.start()
    try:
        with TestClient(app_module.app) as c:
            yield c
    finally:
        for p in patches:
            p.stop()


# ---- Shared fixtures ----------------------------------------------------
@pytest.fixture
def angry_text() -> str:
    return "I am absolutely furious. This is outrageous."


@pytest.fixture
def happy_text() -> str:
    return "I'm so happy today! Everything is wonderful."


@pytest.fixture
def sad_text() -> str:
    return "I feel so sad. Everything is hopeless."


@pytest.fixture
def angry_segments():
    return [
        {
            "Sentence": "I am furious.",
            "Emotion": "anger",
            "Confidence": 0.92,
            "Start": "00:00:00",
            "End": "00:00:03",
        },
        {
            "Sentence": "You will pay.",
            "Emotion": "anger",
            "Confidence": 0.88,
            "Start": "00:00:04",
            "End": "00:00:07",
        },
        {
            "Sentence": "This is outrageous.",
            "Emotion": "anger",
            "Confidence": 0.85,
            "Start": "00:00:08",
            "End": "00:00:11",
        },
    ]


@pytest.fixture
def happy_segments():
    return [
        {
            "Sentence": "What a wonderful day.",
            "Emotion": "happiness",
            "Confidence": 0.91,
            "Start": "00:00:00",
            "End": "00:00:03",
        },
        {
            "Sentence": "I love my friends.",
            "Emotion": "happiness",
            "Confidence": 0.87,
            "Start": "00:00:04",
            "End": "00:00:07",
        },
        {
            "Sentence": "Everything is great.",
            "Emotion": "happiness",
            "Confidence": 0.85,
            "Start": "00:00:08",
            "End": "00:00:11",
        },
    ]


@pytest.fixture
def mixed_segments():
    return [
        {
            "Sentence": "I was happy.",
            "Emotion": "happiness",
            "Confidence": 0.88,
            "Start": "00:00:00",
            "End": "00:00:03",
        },
        {
            "Sentence": "Then terrible news.",
            "Emotion": "sadness",
            "Confidence": 0.82,
            "Start": "00:00:04",
            "End": "00:00:07",
        },
        {
            "Sentence": "I am furious!",
            "Emotion": "anger",
            "Confidence": 0.90,
            "Start": "00:00:08",
            "End": "00:00:11",
        },
        {
            "Sentence": "My friend smiled.",
            "Emotion": "happiness",
            "Confidence": 0.79,
            "Start": "00:00:12",
            "End": "00:00:15",
        },
        {
            "Sentence": "Unexpected turn.",
            "Emotion": "surprise",
            "Confidence": 0.84,
            "Start": "00:00:16",
            "End": "00:00:19",
        },
    ]
