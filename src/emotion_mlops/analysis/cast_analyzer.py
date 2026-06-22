"""
Text-based Cast Analyzer.

Extracts named entities (people) from text using spaCy NER,
runs emotion classification per sentence, and aggregates per-person
emotion profiles. Output includes dominant emotion, emotion
distribution, mention count, and a character archetype label.

Designed for script/dialogue analysis where speakers are referred
to by name rather than diarization labels. Complements per_character.py
which works on CSV transcripts with explicit Speaker columns.

Author: Ceren (Block D Sprint 4)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import spacy

# Singleton spaCy model (loaded once at module import)
_nlp = None

# spaCy en_core_web_sm sometimes mislabels classical or non-Anglo names as
# NORP (e.g., "Claudius") or GPE (e.g., "Ophelia"). Widen the filter so we
# don't miss any character candidates. Frequency + sentence context filters
# out false positives downstream.
_NAME_LABELS = {"PERSON", "NORP", "GPE", "ORG"}


def _get_nlp():
    """Lazy-load spaCy model. en_core_web_sm includes PERSON NER."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _classify_archetype(avg_emotions: Dict[str, float]) -> str:
    """
    Assign a character archetype based on dominant emotions.
    """
    anger = avg_emotions.get("anger", 0)
    fear = avg_emotions.get("fear", 0)
    sadness = avg_emotions.get("sadness", 0)
    happiness = avg_emotions.get("happiness", 0)
    surprise = avg_emotions.get("surprise", 0)
    disgust = avg_emotions.get("disgust", 0)

    if anger > 0.4 and (disgust > 0.3 or fear > 0.3):
        return "Antagonist"
    if sadness > 0.5:
        return "Tragic Hero"
    if happiness > 0.4 and surprise > 0.2:
        return "Comic Relief"
    if happiness > 0.4 and anger < 0.2:
        return "Hero"
    if fear > 0.4:
        return "Mysterious"
    return "Supporting"


def analyze_cast(
    text: str,
    classifier,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extract characters from text and build emotion profiles.

    Args:
        text: Input text (script, dialogue, story).
        classifier: An EmotionClassifier with a `predict_full(text)` method
                    returning (top_label, top_score, all_emotions_dict).
        top_n: Maximum number of characters to return.

    Returns:
        List of dicts sorted by mention count (descending), capped at top_n.
        Each dict has: name, mentions, dominant_emotion, score, archetype,
        emotions (full dict), top_lines (top 3 most confident sentences).
    """
    nlp = _get_nlp()
    doc = nlp(text)

    person_data: Dict[str, List[tuple]] = defaultdict(list)

    for sent in doc.sents:
        people = [ent.text.strip() for ent in sent.ents if ent.label_ in _NAME_LABELS]
        people = [p for p in people if p]

        if not people:
            continue

        try:
            _, _, probs = classifier.predict_full(sent.text.strip())
        except Exception as e:
            print(f"[Cast] predict_full failed on: {sent.text[:60]!r} ({e})")
            continue

        for person in people:
            person_data[person].append((sent.text.strip(), probs))

    results = []
    for name, entries in person_data.items():
        if not entries:
            continue

        emotion_keys = entries[0][1].keys()
        avg_emotions = {
            emo: sum(probs[emo] for _, probs in entries) / len(entries)
            for emo in emotion_keys
        }

        dominant_emo, dominant_score = max(avg_emotions.items(), key=lambda x: x[1])

        scored_lines = []
        for sent_text, probs in entries:
            top_emo, top_score = max(probs.items(), key=lambda x: x[1])
            scored_lines.append(
                {
                    "sentence": sent_text,
                    "emotion": top_emo,
                    "score": round(top_score, 4),
                }
            )
        scored_lines.sort(key=lambda x: x["score"], reverse=True)
        top_lines = scored_lines[:3]

        archetype = _classify_archetype(avg_emotions)

        results.append(
            {
                "name": name,
                "mentions": len(entries),
                "dominant_emotion": dominant_emo,
                "score": round(dominant_score, 4),
                "archetype": archetype,
                "emotions": {k: round(v, 4) for k, v in avg_emotions.items()},
                "top_lines": top_lines,
            }
        )

    results.sort(key=lambda x: x["mentions"], reverse=True)
    return results[:top_n]
