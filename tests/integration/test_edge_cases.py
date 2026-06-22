"""
Edge case tests, empty inputs, malformed payloads, very long inputs,
oversized requests, and other tricky scenarios that often break APIs
in production.

These cover what real users will inevitably throw at the server:
copy-paste mistakes, network glitches, accidental empty fields, etc.
"""

from __future__ import annotations

import requests

# ==========================================================================
# Empty / missing input
# ==========================================================================


class TestEmptyInputs:
    def test_predict_text_empty_string_does_not_500(self, base_url, model_ready):
        """An empty string is a legitimate user mistake, never a 500."""
        r = requests.post(f"{base_url}/predict/text", json={"text": ""})
        # The API may return 200 (with neutral) or 422, but NOT 500.
        assert r.status_code in (
            200,
            422,
        ), f"Expected 200 or 422 for empty text, got {r.status_code}"

    def test_predict_text_missing_field_returns_422(self, base_url, model_ready):
        """Pydantic should reject missing `text` field with 422."""
        r = requests.post(f"{base_url}/predict/text", json={})
        assert r.status_code == 422

    def test_predict_text_null_text_returns_422(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/text", json={"text": None})
        assert r.status_code == 422

    def test_cast_empty_text_returns_422(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/cast", json={"text": "   ", "top_n": 5})
        assert r.status_code == 422


# ==========================================================================
# Malformed JSON / wrong types
# ==========================================================================
class TestMalformedRequests:
    def test_invalid_json_returns_422(self, base_url, model_ready):
        """Sending a string where a dict is expected must not crash."""
        r = requests.post(
            f"{base_url}/predict/text",
            data="not a json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422)

    def test_wrong_type_for_text_returns_422(self, base_url, model_ready):
        """A number where a string is expected must fail validation."""
        r = requests.post(f"{base_url}/predict/text", json={"text": 42})
        assert r.status_code == 422

    def test_segments_as_string_returns_422(self, base_url, model_ready):
        """`segments` must be a list, passing a string should 422."""
        r = requests.post(
            f"{base_url}/predict/timeline",
            json={"segments": "not a list"},
        )
        assert r.status_code == 422

    def test_compare_missing_side_returns_422(
        self, base_url, model_ready, angry_segments
    ):
        """Compare requires both `left` and `right`."""
        r = requests.post(
            f"{base_url}/compare",
            json={"left": {"type": "text", "result": {"segments": angry_segments}}},
        )
        assert r.status_code == 422


# ==========================================================================
# Boundary inputs, very short, very long, unusual characters
# ==========================================================================
class TestBoundaryInputs:
    def test_single_character_text(self, base_url, model_ready):
        """Edge case: just one letter. Should not crash."""
        r = requests.post(f"{base_url}/predict/text", json={"text": "a"})
        assert r.status_code in (200, 422)

    def test_very_long_text_is_handled(self, base_url, model_ready):
        """3000-char paragraph; the API must handle it gracefully."""
        long_text = "I am furious. " * 200  # ~2800 chars
        r = requests.post(f"{base_url}/predict/text", json={"text": long_text})
        # Either truncates and returns 200, or politely rejects with 422.
        assert r.status_code in (200, 422)
        if r.status_code == 200:
            assert 0 <= r.json()["confidence"] <= 1

    def test_unicode_and_emojis_dont_crash(self, base_url, model_ready):
        """Real users paste emoji-heavy text from social media."""
        text = "I'm so happy  Today is amazing! Çağdaş naïve résumé 你好"
        r = requests.post(f"{base_url}/predict/text", json={"text": text})
        assert r.status_code == 200
        assert r.json()["emotion"] in (
            "happiness",
            "neutral",
            "anger",
            "sadness",
            "surprise",
            "fear",
            "disgust",
        )

    def test_whitespace_only_text(self, base_url, model_ready):
        """Whitespace-only input is effectively empty, should not 500."""
        r = requests.post(f"{base_url}/predict/text", json={"text": "     \n\t  "})
        assert r.status_code in (200, 422)
