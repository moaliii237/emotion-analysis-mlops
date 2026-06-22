"""Sprint 4 endpoint unit tests, all Groq calls mocked away."""


class TestTimeline:
    def test_builds_timeline(self, client, mixed_segments):
        r = client.post("/predict/timeline", json={"segments": mixed_segments}).json()
        assert r["segment_count"] == len(mixed_segments)

    def test_empty_segments_422(self, client):
        r = client.post("/predict/timeline", json={"segments": []})
        assert r.status_code == 422


class TestScenes:
    def test_groups_segments_into_scenes(self, client, mixed_segments):
        r = client.post(
            "/predict/scenes",
            json={"segments": mixed_segments, "min_segments_per_scene": 1},
        ).json()
        assert r["scene_count"] >= 1
        for scene in r["scenes"]:
            assert scene.get("dominant_emotion")
            assert scene.get("duration", 0) > 0
            assert isinstance(scene.get("emotion_counts"), dict)

    def test_scenes_capture_emotion_variety(self, client, mixed_segments):
        r = client.post(
            "/predict/scenes",
            json={"segments": mixed_segments, "min_segments_per_scene": 1},
        ).json()
        all_counts = {}
        for scene in r["scenes"]:
            for emo, count in scene.get("emotion_counts", {}).items():
                all_counts[emo] = all_counts.get(emo, 0) + count
        assert len(all_counts) >= 3


class TestTrailer:
    def test_picks_clips_from_scenes(self, client, mixed_segments):
        scenes = client.post(
            "/predict/scenes",
            json={"segments": mixed_segments, "min_segments_per_scene": 1},
        ).json()["scenes"]
        r = client.post(
            "/predict/trailer",
            json={"scenes": scenes, "max_clips": 3, "generate_taglines": False},
        ).json()
        assert 1 <= len(r["clips"]) <= 3
        assert r["total_duration"] > 0


class TestScore:
    def test_returns_letter_grade(self, client, angry_segments):
        r = client.post(
            "/predict/score",
            json={"segments": angry_segments, "use_ai_classification": False},
        ).json()
        assert r["grade"][0].upper() in "ABCDEF"


class TestShifts:
    def test_detects_shift_in_mixed_content(self, client, mixed_segments):
        r = client.post(
            "/predict/shifts",
            json={"segments": mixed_segments, "min_magnitude": 0.1},
        ).json()
        assert r["shift_count"] >= 1

    def test_no_shifts_in_constant_emotion(self, client, angry_segments):
        r = client.post(
            "/predict/shifts",
            json={"segments": angry_segments, "min_magnitude": 0.5},
        ).json()
        assert r["shift_count"] == 0


class TestPersuasion:
    def test_returns_score_in_unit_interval(self, client, angry_segments):
        r = client.post(
            "/predict/persuasion",
            json={"segments": angry_segments, "use_ai_assessment": False},
        ).json()
        assert 0.0 <= r["persuasion_score"] <= 1.0


class TestSummary:
    def test_returns_summary_text(self, client, mixed_segments):
        r = client.post("/predict/summary", json={"segments": mixed_segments}).json()
        assert isinstance(r["summary"], str)
        assert len(r["summary"]) > 20

    def test_ai_generated_flag_false_when_mocked(self, client, mixed_segments):
        r = client.post("/predict/summary", json={"segments": mixed_segments}).json()
        assert r["ai_generated"] is False


class TestCompare:
    def test_compares_two_sides(self, client, angry_segments, happy_segments):
        payload = {
            "left": {
                "type": "text",
                "label": "A",
                "result": {"segments": angry_segments, "cast": []},
            },
            "right": {
                "type": "text",
                "label": "B",
                "result": {"segments": happy_segments, "cast": []},
            },
        }
        r = client.post("/compare", json=payload).json()
        assert "comparison" in r
        assert 0.0 <= r["comparison"]["similarity"] <= 1.0

    def test_labels_preserved(self, client, angry_segments, happy_segments):
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
        r = client.post("/compare", json=payload).json()
        assert r["left_label"] == "MyShow A"
        assert r["right_label"] == "MyShow B"
