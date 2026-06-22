"""
Pytest configuration and shared fixtures for the EmotionScope test suite.

These tests target a RUNNING server (default http://localhost:8000) so that
every endpoint is exercised against the real models. Set TEST_BASE_URL to
point at a different host if needed.
"""

from __future__ import annotations

import os

import pytest
import requests

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DEFAULT_BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")

EMOTIONS = ["happiness", "neutral", "anger", "sadness", "surprise", "fear", "disgust"]


# --------------------------------------------------------------------------
# Server availability check, skips ALL tests if the API is not reachable
# instead of producing 100+ noisy ConnectionError failures.
# --------------------------------------------------------------------------
def _server_is_up(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_server():
    """Auto-skip the whole suite if the API is not running."""
    if not _server_is_up(DEFAULT_BASE_URL):
        pytest.skip(
            f"API not running at {DEFAULT_BASE_URL}. "
            "Start the server with `uvicorn emotion_mlops.api.app:app --port 8000` first.",
            allow_module_level=True,
        )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL of the running API."""
    return DEFAULT_BASE_URL


@pytest.fixture(scope="session")
def model_ready(base_url):
    """Wait until /health reports the model is loaded. Fails fast otherwise."""
    r = requests.get(f"{base_url}/health", timeout=5).json()
    if not r.get("model_loaded"):
        pytest.fail("Model is not loaded, start the server and wait for warmup.")
    return r


# --- Demo text samples ----------------------------------------------------
@pytest.fixture
def angry_text() -> str:
    return "I am absolutely furious. This is outrageous. You will pay for this."


@pytest.fixture
def happy_text() -> str:
    return "I'm so happy today! Everything is wonderful. This is the best day ever!"


@pytest.fixture
def sad_text() -> str:
    return "I feel so down. Everything seems hopeless. I miss them deeply."


@pytest.fixture
def neutral_text() -> str:
    return "The meeting is at three. Please bring the documents. Room 4B is available."


# --- Pre-classified segments (no model call needed) -----------------------
@pytest.fixture
def angry_segments():
    """A short list of segments already labelled with anger."""
    return [
        {
            "Sentence": "I am furious at this disaster.",
            "Emotion": "anger",
            "Confidence": 0.92,
            "Start": "00:00:00",
            "End": "00:00:03",
        },
        {
            "Sentence": "You will pay for this insult.",
            "Emotion": "anger",
            "Confidence": 0.88,
            "Start": "00:00:04",
            "End": "00:00:07",
        },
        {
            "Sentence": "I cannot believe what you've done.",
            "Emotion": "anger",
            "Confidence": 0.81,
            "Start": "00:00:08",
            "End": "00:00:11",
        },
    ]


@pytest.fixture
def happy_segments():
    return [
        {
            "Sentence": "What a wonderful day this is.",
            "Emotion": "happiness",
            "Confidence": 0.91,
            "Start": "00:00:00",
            "End": "00:00:03",
        },
        {
            "Sentence": "I love being with my friends.",
            "Emotion": "happiness",
            "Confidence": 0.87,
            "Start": "00:00:04",
            "End": "00:00:07",
        },
        {
            "Sentence": "Everything is going so well.",
            "Emotion": "happiness",
            "Confidence": 0.85,
            "Start": "00:00:08",
            "End": "00:00:11",
        },
    ]


@pytest.fixture
def mixed_segments():
    """Segments with several different emotions, useful for shift/scene tests."""
    return [
        {
            "Sentence": "I was so happy this morning.",
            "Emotion": "happiness",
            "Confidence": 0.88,
            "Start": "00:00:00",
            "End": "00:00:03",
        },
        {
            "Sentence": "Then I heard the terrible news.",
            "Emotion": "sadness",
            "Confidence": 0.82,
            "Start": "00:00:04",
            "End": "00:00:07",
        },
        {
            "Sentence": "I am furious at the injustice.",
            "Emotion": "anger",
            "Confidence": 0.90,
            "Start": "00:00:08",
            "End": "00:00:11",
        },
        {
            "Sentence": "But my friend made me smile.",
            "Emotion": "happiness",
            "Confidence": 0.79,
            "Start": "00:00:12",
            "End": "00:00:15",
        },
        {
            "Sentence": "What an unexpected turn of events.",
            "Emotion": "surprise",
            "Confidence": 0.84,
            "Start": "00:00:16",
            "End": "00:00:19",
        },
    ]


# --------------------------------------------------------------------------
# Server-check hook, applied ONLY to tests that ask for `base_url`.
# Schema tests don't use base_url, so they run without a server.
# --------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    if _server_is_up(DEFAULT_BASE_URL):
        return
    skip_marker = pytest.mark.skip(
        reason=f"API not running at {DEFAULT_BASE_URL}, skipping HTTP-based tests."
    )
    for item in items:
        if "base_url" in getattr(item, "fixturenames", ()):
            item.add_marker(skip_marker)
