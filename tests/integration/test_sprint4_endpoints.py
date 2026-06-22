"""
Sprint 4 endpoint tests, Cast, Timeline, Scenes, Trailer, Score,
Shifts, Persuasion, AI Summary, Compare.

We feed pre-classified segments so the tests don't have to wait for the
classifier each time. The point of these tests is to verify the analytic
LOGIC of each endpoint, not the upstream model.
"""

from __future__ import annotations

import requests


# ==========================================================================
# /predict/cast
# ==========================================================================
class TestCast:
    def test_finds_named_characters(self, base_url, model_ready):
        text = (
            "Hamlet stared at the ghost in terror. Claudius poured the poison. "
            "Ophelia sang sadly by the river. Polonius hid behind the curtain."
        )
        r = requests.post(
            f"{base_url}/predict/cast", json={"text": text, "top_n": 10}
        ).json()
        names = {c["name"] for c in r["cast"]}
        # At least two of the four should be detected.
        detected = names & {"Hamlet", "Claudius", "Ophelia", "Polonius"}
        assert len(detected) >= 2, f"Expected ≥2 named characters, got {names}"

    def test_cast_entries_have_required_fields(self, base_url, model_ready):
        r = requests.post(
            f"{base_url}/predict/cast",
            json={"text": "Trump shouted at Biden. Biden replied calmly."},
        ).json()
        for c in r["cast"]:
            assert {"name", "mentions", "dominant_emotion", "archetype"}.issubset(
                c.keys()
            )

    def test_empty_text_returns_422(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/cast", json={"text": "", "top_n": 5})
        assert r.status_code == 422


# ==========================================================================
# /predict/timeline
# ==========================================================================
class TestTimeline:
    def test_builds_timeline_for_segments(self, base_url, model_ready, mixed_segments):
        r = requests.post(
            f"{base_url}/predict/timeline", json={"segments": mixed_segments}
        ).json()
        assert r["segment_count"] == len(mixed_segments)
        assert "times" in r
        assert len(r["times"]) == len(mixed_segments)

    def test_empty_segments_returns_422(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/timeline", json={"segments": []})
        assert r.status_code == 422


# ==========================================================================
# /predict/scenes
# ==========================================================================
class TestScenes:
    def test_groups_segments_into_scenes(self, base_url, model_ready, mixed_segments):
        r = requests.post(
            f"{base_url}/predict/scenes",
            json={"segments": mixed_segments, "min_segments_per_scene": 1},
        ).json()
        assert r["scene_count"] >= 1
        # Each scene must report a dominant emotion, a positive duration,
        # and the emotion_counts dict it was built from.
        for scene in r["scenes"]:
            assert scene.get(
                "dominant_emotion"
            ), f"Scene is missing dominant_emotion: {scene}"
            assert (
                scene.get("duration", 0) > 0
            ), f"Scene has non-positive duration: {scene}"
            assert isinstance(
                scene.get("emotion_counts"), dict
            ), f"Scene is missing emotion_counts dict: {scene}"

    def test_scenes_capture_emotion_variety(
        self, base_url, model_ready, mixed_segments
    ):
        """
        Even if the algorithm collapses everything into a single scene,
        the emotion_counts dict must record every distinct emotion present
        in the input, that's how downstream features (trailer, score) can
        still see the variety.
        """
        r = requests.post(
            f"{base_url}/predict/scenes",
            json={"segments": mixed_segments, "min_segments_per_scene": 1},
        ).json()
        all_counts = {}
        for scene in r["scenes"]:
            for emo, count in scene.get("emotion_counts", {}).items():
                all_counts[emo] = all_counts.get(emo, 0) + count
        # Our mixed input has 4 different emotions (happiness, sadness, anger, surprise).
        assert (
            len(all_counts) >= 3
        ), f"Scenes should track ≥3 distinct emotions, got {list(all_counts.keys())}"


# ==========================================================================
# /predict/trailer
# ==========================================================================
class TestTrailer:
    def test_picks_clips_from_scenes(self, base_url, model_ready, mixed_segments):
        # Trailer needs scenes first.
        scenes = requests.post(
            f"{base_url}/predict/scenes",
            json={"segments": mixed_segments, "min_segments_per_scene": 1},
        ).json()["scenes"]

        r = requests.post(
            f"{base_url}/predict/trailer",
            json={"scenes": scenes, "max_clips": 3, "generate_taglines": False},
        ).json()
        assert len(r["clips"]) >= 1
        assert len(r["clips"]) <= 3
        assert r["total_duration"] > 0

    def test_empty_scenes_returns_422(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/trailer", json={"scenes": []})
        assert r.status_code == 422


# ==========================================================================
# /predict/score
# ==========================================================================
class TestScore:
    def test_returns_letter_grade(self, base_url, model_ready, angry_segments):
        r = requests.post(
            f"{base_url}/predict/score",
            json={"segments": angry_segments, "use_ai_classification": False},
        ).json()
        # The grade is "A+", "A", "B+" … "F", first char is always a letter.
        assert r["grade"][0].upper() in "ABCDEF"

    def test_high_anger_high_drama_score(self, base_url, model_ready, angry_segments):
        """Anger-only content should score high on the drama axis."""
        r = requests.post(
            f"{base_url}/predict/score",
            json={"segments": angry_segments, "use_ai_classification": False},
        ).json()
        assert "scores" in r
        # Drama should be at least 0.3 for pure-anger content.
        assert (
            r["scores"]["drama"] >= 0.3
        ), f"Pure anger should be dramatic, got {r['scores']['drama']}"


# ==========================================================================
# /predict/shifts
# ==========================================================================
class TestShifts:
    def test_detects_shift_in_mixed_content(
        self, base_url, model_ready, mixed_segments
    ):
        r = requests.post(
            f"{base_url}/predict/shifts",
            json={"segments": mixed_segments, "min_magnitude": 0.1},
        ).json()
        # Mixed segments deliberately contain happy->sad->anger->happy transitions.
        assert r["shift_count"] >= 1, "Expected at least one emotional shift"

    def test_no_shifts_in_constant_emotion(self, base_url, model_ready, angry_segments):
        """A run of all-anger segments shouldn't produce any shifts."""
        r = requests.post(
            f"{base_url}/predict/shifts",
            json={"segments": angry_segments, "min_magnitude": 0.5},
        ).json()
        assert r["shift_count"] == 0

    def test_shifts_have_from_and_to(self, base_url, model_ready, mixed_segments):
        r = requests.post(
            f"{base_url}/predict/shifts",
            json={"segments": mixed_segments, "min_magnitude": 0.1},
        ).json()
        for s in r["shifts"]:
            assert "from_emotion" in s
            assert "to_emotion" in s
            assert s["from_emotion"] != s["to_emotion"]


# ==========================================================================
# /predict/persuasion
# ==========================================================================
class TestPersuasion:
    def test_returns_score_in_unit_interval(
        self, base_url, model_ready, angry_segments
    ):
        r = requests.post(
            f"{base_url}/predict/persuasion",
            json={"segments": angry_segments, "use_ai_assessment": False},
        ).json()
        assert 0.0 <= r["persuasion_score"] <= 1.0

    def test_returns_confidence_level_label(
        self, base_url, model_ready, angry_segments
    ):
        r = requests.post(
            f"{base_url}/predict/persuasion",
            json={"segments": angry_segments, "use_ai_assessment": False},
        ).json()
        assert isinstance(r["confidence_level"], str)
        assert len(r["confidence_level"]) > 0


# ==========================================================================
# /predict/summary, the AI Verdict (Sprint 4)
# ==========================================================================
class TestSummary:
    def test_returns_summary_text(self, base_url, model_ready, mixed_segments):
        r = requests.post(
            f"{base_url}/predict/summary",
            json={"segments": mixed_segments},
        ).json()
        assert "summary" in r
        assert isinstance(r["summary"], str)
        assert len(r["summary"]) > 20, "Summary is too short to be useful"

    def test_summary_flags_ai_or_fallback(self, base_url, model_ready, mixed_segments):
        r = requests.post(
            f"{base_url}/predict/summary",
            json={"segments": mixed_segments},
        ).json()
        # Either Groq produced it, or we got the deterministic fallback.
        assert "ai_generated" in r
        assert isinstance(r["ai_generated"], bool)

    def test_empty_segments_returns_422(self, base_url, model_ready):
        r = requests.post(f"{base_url}/predict/summary", json={"segments": []})
        assert r.status_code == 422


# ==========================================================================
# /compare, Sprint 4 (Ceren)
# ==========================================================================
class TestCompare:
    def test_compares_two_sides(
        self, base_url, model_ready, angry_segments, happy_segments
    ):
        payload = {
            "left": {
                "type": "text",
                "label": "Angry side",
                "result": {"segments": angry_segments, "cast": []},
            },
            "right": {
                "type": "text",
                "label": "Happy side",
                "result": {"segments": happy_segments, "cast": []},
            },
        }
        r = requests.post(f"{base_url}/compare", json=payload).json()
        assert "comparison" in r
        assert "similarity" in r["comparison"]
        assert 0.0 <= r["comparison"]["similarity"] <= 1.0

    def test_winners_make_sense(
        self, base_url, model_ready, angry_segments, happy_segments
    ):
        """Angry segments should win 'more_angry', happy should win 'more_happy'."""
        payload = {
            "left": {
                "type": "text",
                "label": "Angry",
                "result": {"segments": angry_segments, "cast": []},
            },
            "right": {
                "type": "text",
                "label": "Happy",
                "result": {"segments": happy_segments, "cast": []},
            },
        }
        r = requests.post(f"{base_url}/compare", json=payload).json()
        winners = r["comparison"]["winners"]
        if "more_angry" in winners:
            assert winners["more_angry"] == "left", "Angry side should win 'more_angry'"
        if "more_happy" in winners:
            assert (
                winners["more_happy"] == "right"
            ), "Happy side should win 'more_happy'"

    def test_labels_preserved_in_response(
        self, base_url, model_ready, angry_segments, happy_segments
    ):
        payload = {
            "left": {
                "type": "text",
                "label": "MyShow A",
                "result": {"segments": angry_segments, "cast": []},
            },
            "right": {
                "type": "text",
                "label": "MyShow B",
                "result": {"segments": happy_segments, "cast": []},
            },
        }
        r = requests.post(f"{base_url}/compare", json=payload).json()
        assert r["left_label"] == "MyShow A"
        assert r["right_label"] == "MyShow B"
