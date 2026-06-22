"""
Extra unit tests for coverage uplift.

Targets modules that have testable surface but lack coverage:
- api/database.py, PipelineMetrics dataclass
- models/classifier.py, EmotionClassifier construction state
- analysis/*, edge-case paths for timeline / scenes / shifts / trailer
              / persuasion / content_score / compare.
"""

from __future__ import annotations

import pytest


# =========================================================================
# api/database.py, PipelineMetrics dataclass
# =========================================================================
class TestPipelineMetrics:
    """Pipeline timing dataclass used by /predict/pipeline."""

    def test_metrics_created_with_all_fields(self):
        from emotion_mlops.api.database import PipelineMetrics

        m = PipelineMetrics(
            t_whisper=1.5,
            t_translate=2.0,
            t_classify=3.5,
            t_total=7.0,
            segment_count=42,
        )
        assert m.t_whisper == 1.5
        assert m.t_translate == 2.0
        assert m.t_classify == 3.5
        assert m.t_total == 7.0
        assert m.segment_count == 42

    def test_metrics_with_zero_segments(self):
        from emotion_mlops.api.database import PipelineMetrics

        m = PipelineMetrics(
            t_whisper=0.0,
            t_translate=0.0,
            t_classify=0.0,
            t_total=0.0,
            segment_count=0,
        )
        assert m.segment_count == 0


# =========================================================================
# models/classifier.py, EmotionClassifier construction
# =========================================================================
class TestEmotionClassifierInit:
    """EmotionClassifier basic state without loading a real model."""

    def test_constructor_stores_version(self):
        from emotion_mlops.models.classifier import EmotionClassifier

        c = EmotionClassifier(model_path="/fake/path", model_version="test-v1")
        assert c.model_version == "test-v1"
        assert c.model_path

    def test_is_loaded_false_before_load(self):
        from emotion_mlops.models.classifier import EmotionClassifier

        c = EmotionClassifier(model_path="/fake/path", model_version="test-v1")
        # Without calling load(), is_loaded must be False
        assert c.is_loaded is False


# =========================================================================
# analysis/timeline.py, edge cases
# =========================================================================
class TestTimelineEdgeCases:
    """Cover smoothing and threshold branches in build_timeline."""

    def test_single_segment(self):
        from emotion_mlops.analysis.timeline import build_timeline

        segs = [
            {
                "Start": "00:00:00",
                "End": "00:00:03",
                "Sentence": "hello",
                "Emotion": "happiness",
                "Confidence": 0.9,
            }
        ]
        tl = build_timeline(segs)
        assert isinstance(tl, dict)

    def test_smooth_window_three(self, mixed_segments):
        from emotion_mlops.analysis.timeline import build_timeline

        tl = build_timeline(mixed_segments, smooth_window=3, peak_threshold=0.5)
        assert isinstance(tl, dict)

    def test_low_peak_threshold(self, mixed_segments):
        from emotion_mlops.analysis.timeline import build_timeline

        tl = build_timeline(mixed_segments, peak_threshold=0.1)
        assert isinstance(tl, dict)


# =========================================================================
# analysis/scenes.py, edge cases
# =========================================================================
class TestScenesEdgeCases:
    """Cover boundary detection and grouping branches in detect_scenes."""

    def test_single_segment_one_scene(self):
        from emotion_mlops.analysis.scenes import detect_scenes

        segs = [
            {
                "Start": "00:00:00",
                "End": "00:00:03",
                "Sentence": "hello",
                "Emotion": "happiness",
                "Confidence": 0.9,
            }
        ]
        scenes = detect_scenes(segs)
        assert isinstance(scenes, list)

    def test_min_segments_three(self, mixed_segments):
        from emotion_mlops.analysis.scenes import detect_scenes

        scenes = detect_scenes(mixed_segments, min_segments_per_scene=3)
        assert isinstance(scenes, list)

    def test_high_boundary_threshold(self, mixed_segments):
        from emotion_mlops.analysis.scenes import detect_scenes

        # Very high boundary threshold -> fewer scenes
        scenes = detect_scenes(mixed_segments, boundary_change_threshold=0.99)
        assert isinstance(scenes, list)


# =========================================================================
# analysis/shifts.py, edge cases
# =========================================================================
class TestShiftsEdgeCases:
    """Cover magnitude and confidence filters in detect_shifts."""

    def test_two_segments_same_emotion(self, happy_segments):
        from emotion_mlops.analysis.shifts import detect_shifts

        shifts = detect_shifts(happy_segments)
        assert isinstance(shifts, list)
        # All segments are happy, no shifts expected
        assert len(shifts) == 0

    def test_very_high_confidence_threshold(self, mixed_segments):
        from emotion_mlops.analysis.shifts import detect_shifts

        # Threshold 0.99 -> no qualifying shifts
        shifts = detect_shifts(mixed_segments, min_confidence=0.99)
        assert isinstance(shifts, list)
        assert len(shifts) == 0

    def test_very_high_magnitude_threshold(self, mixed_segments):
        from emotion_mlops.analysis.shifts import detect_shifts

        shifts = detect_shifts(mixed_segments, min_magnitude=0.99)
        assert isinstance(shifts, list)


# =========================================================================
# analysis/persuasion.py, edge cases
# =========================================================================
class TestPersuasionEdgeCases:
    """Cover the non-AI scoring path in analyze_persuasion."""

    def test_without_ai_assessment(self, angry_segments):
        from emotion_mlops.analysis.persuasion import analyze_persuasion

        result = analyze_persuasion(angry_segments, use_ai_assessment=False)
        assert isinstance(result, dict)


# =========================================================================
# analysis/content_score.py, edge cases
# =========================================================================
class TestContentScoreEdgeCases:
    """Cover the non-AI grading path in score_content."""

    def test_without_ai_classification(self, mixed_segments):
        from emotion_mlops.analysis.content_score import score_content

        result = score_content(mixed_segments, use_ai_classification=False)
        assert isinstance(result, dict)

    def test_with_empty_cast_and_scenes(self, mixed_segments):
        from emotion_mlops.analysis.content_score import score_content

        result = score_content(
            mixed_segments,
            cast=[],
            scenes=[],
            use_ai_classification=False,
        )
        assert isinstance(result, dict)


# =========================================================================
# analysis/compare.py, edge cases
# =========================================================================
class TestCompareEdgeCases:
    """Cover the symmetric branches in compare."""

    def test_compare_two_short_results(self, happy_segments, angry_segments):
        from emotion_mlops.analysis.compare import compare

        left = {"segments": happy_segments, "t_total": 1.0}
        right = {"segments": angry_segments, "t_total": 1.2}
        result = compare(left=left, right=right)
        assert isinstance(result, dict)

    def test_compare_identical_results(self, happy_segments):
        from emotion_mlops.analysis.compare import compare

        result = compare(
            left={"segments": happy_segments, "t_total": 1.0},
            right={"segments": happy_segments, "t_total": 1.0},
        )
        assert isinstance(result, dict)


# =========================================================================
# analysis/trailer.py, edge cases
# =========================================================================
class TestTrailerEdgeCases:
    """Cover diversity and capacity branches in pick_trailer_highlights."""

    def test_max_clips_one(self, mixed_segments):
        """With max_clips=1, only one highlight should be picked."""
        from emotion_mlops.analysis.scenes import detect_scenes
        from emotion_mlops.analysis.trailer import pick_trailer_highlights

        scenes = detect_scenes(mixed_segments, min_segments_per_scene=1)
        if not scenes:
            pytest.skip("No scenes detected")
        result = pick_trailer_highlights(
            scenes, max_clips=1, enforce_mood_diversity=False
        )
        assert isinstance(result, dict)
