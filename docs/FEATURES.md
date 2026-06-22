# EmotionScope, Feature Guide

EmotionScope takes audio, video, a transcript CSV, raw text, or a YouTube URL
and gives back sentence-level emotion. On top of that there are ten analysis
features. This file is a tour of what the app can do and which endpoint powers
each part.

## The core pipeline

```text
 audio / video / transcript / text / YouTube URL
        |
        v
 Whisper      ->  speech-to-text (optional, for audio/video)
        |
        v
 MarianMT     ->  EN to NL translation (optional)
        |
        v
 DeBERTa-v3-base  ->  per-sentence emotion + confidence
        |
        v
 PostgreSQL + 10 analysis features + feedback loop
```

Every run produces a table of `(Start, Sentence, Emotion, Confidence)` rows, and
all the downstream features build on that table.

## The emotion model

| Property | Value |
|---|---|
| Architecture | DeBERTa-v3-base (fine-tuned) |
| Labels | 7 (anger, disgust, fear, happiness, sadness, surprise, neutral) |
| Training data | GoEmotions (~58k rows, collapsed to 7 labels) |
| Serving | In-process locally, or the Azure ML online endpoint (blue/green) |

## Feature catalogue

| Feature | Endpoint | What it does |
|---|---|---|
| Single prediction | `POST /predict/text` | Emotion + confidence for one sentence |
| Full probabilities | `POST /predict/text/full` | Probability for all 7 labels |
| Cloud prediction | `POST /predict/cloud` | Same, via the Azure ML endpoint |
| Full pipeline | `POST /predict/pipeline` | Audio / CSV to per-sentence emotions |
| YouTube | `POST /predict/youtube` | Same pipeline straight from a URL |
| Timeline | `POST /predict/timeline` | Emotion intensity as a time-series |
| Scenes | `POST /predict/scenes` | Splits the transcript into emotional scenes |
| Shifts | `POST /predict/shifts` | Finds emotional turning points |
| Trailer | `POST /predict/trailer` | Auto-picks the most emotional clips |
| Content score | `POST /predict/score` | Producer-style scorecard |
| Persuasion | `POST /predict/persuasion` | Scores a speaker's rhetorical strength |
| Cast | `POST /predict/cast` | Per-character emotion profiles via spaCy NER |
| Summary | `POST /summarize`, `POST /predict/summary` | BART summary plus a short verdict |
| Q&A | `POST /ask` | Ask questions about a transcript (Groq LLM) |
| Compare | `POST /compare` | Side-by-side comparison of two analyses |
| Feedback | `POST /feedback` ... | Rate predictions, drives retraining |

## Feature groups

Narrative analysis (Timeline, Scenes, Shifts). Shows how emotion moves through an
episode: the overall arc, where scenes break, and the exact turning points.

Production tools (Trailer, Content Score, Persuasion). Made for producers: pull
the best clips, get a content scorecard, and measure how persuasive a speaker is.

Character analysis (Cast). Uses spaCy named-entity recognition to find characters
by name and build a per-character emotion profile plus an archetype (Hero,
Antagonist, Tragic Hero, and so on).

Language features (Summary, Q&A, Compare). BART summarization, Groq-LLM question
answering over the transcript, and a two-result diff view.

Feedback and retraining. Every prediction can be rated up or down, and the
corrections feed the continuous-retraining flywheel. See
[`FEEDBACK_LOOP.md`](FEEDBACK_LOOP.md).

## Emotion labels

| Label | Meaning |
|---|---|
| `anger` | frustration / rage |
| `disgust` | revulsion |
| `fear` | anxiety / dread |
| `happiness` | joy / satisfaction |
| `sadness` | sorrow / grief |
| `surprise` | shock / astonishment |
| `neutral` | no strong emotion |

## Tech behind the features

FastAPI, DeBERTa-v3-base, Whisper, MarianMT, BART (summary), Groq LLM (Q&A),
spaCy `en_core_web_sm` (cast NER), PostgreSQL, Azure ML.
