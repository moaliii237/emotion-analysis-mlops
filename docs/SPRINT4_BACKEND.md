# Sprint 4, Backend Feature Expansion

**Author**: Ceren (Block D, Group NLP-1)
**Branch**: `feature/cere-sprint4-backend`
**Goal**: Turn the EmotionScope MLOps platform into a content intelligence backend that a real client (Content Intelligence Agency, TV / film / Big Brother format analytics) could use.

---

## What was added

This sprint added **8 new analysis modules** + **9 new REST endpoints**, plus **5 places where the Groq LLM (llama-3.1-8b-instant)** produces human-readable AI output. All features work across every input type the existing pipeline already supports (raw text, audio/video, transcript CSV, YouTube URL).

| # | Feature | Module | Endpoint | AI? |
|---|---------|--------|----------|-----|
| 1 | Cast Analyzer (NER-based character profiling) | `analysis/cast_analyzer.py` | `POST /predict/cast` + auto in pipeline |, |
| 2 | Side-by-side Compare | `analysis/compare.py` | `POST /compare` | Auto-title |
| 3 | Emotion Timeline (Chart.js-ready + peaks) | `analysis/timeline.py` | `POST /predict/timeline` + auto in pipeline |, |
| 4 | Scene Detection | `analysis/scenes.py` | `POST /predict/scenes` | Scene title |
| 5 | Trailer Highlights (best 30-sec clips) | `analysis/trailer.py` | `POST /predict/trailer` | Tagline |
| 6 | Content Scorecard (engagement / drama / comedy / pacing) | `analysis/content_score.py` | `POST /predict/score` | Classification |
| 7 | Emotion Shifts (turning points) | `analysis/shifts.py` | `POST /predict/shifts` |, |
| 8 | Persuasion Analyzer | `analysis/persuasion.py` | `POST /predict/persuasion` | Assessment |
|, | AI Auto-Title Helper (shared) | `analysis/autotitle.py` | (used internally) |, |

---

## Why these features

The existing pipeline produced **per-segment emotion labels**. That's a building block, not a product. A real client doesn't ask "what's the emotion of sentence #14?", they ask:

- *"Who are the main characters and how do they come across?"* -> **Cast**
- *"How do these two episodes compare?"* -> **Compare**
- *"Show me how the energy moves over time."* -> **Timeline**
- *"Where are the natural scenes / acts?"* -> **Scenes**
- *"Give me the best 30 seconds for a trailer."* -> **Trailer**
- *"What kind of show is this and who's it for?"* -> **Score**
- *"Where do things take a sharp turn?"* -> **Shifts**
- *"How persuasive is this speaker?"* -> **Persuasion**

Each module turns raw per-segment data into a producer-facing answer.

---

## Endpoint reference (one-page)

All endpoints accept `application/json` and return `application/json`. They re-use the segment shape produced by `/predict/pipeline` and `/predict/youtube`, so chaining requests on the frontend is straightforward.

### `POST /predict/cast`
```json
{ "text": "Hamlet stood alone...", "top_n": 5 }
```
Returns up to `top_n` characters with mention count, dominant emotion, archetype (Antagonist / Hero / Tragic Hero / Comic Relief / Mysterious / Supporting), full emotion distribution, and top 3 most confident lines per character.

### `POST /compare`
Send two pipeline-shaped results (from text/audio/YouTube/CSV) side-by-side. Optional `label` per side, if omitted, the AI generates one (e.g. *"Trump's Furious Press Walkout"* vs *"Baker's Joyful Moment in the Spotlight"*). Returns emotion deltas, winners (more_happy / more_intense / more_diverse / …), common actors, Jaccard actor overlap, cosine similarity, and full stats.

### `POST /predict/timeline`
Pipeline segments -> Chart.js-ready dict: `times`, per-emotion `series`, `labels`, `sentences`, `peaks` (sorted by score), `duration`, `segment_count`. Optional `smooth_window` for moving average; `peak_threshold` controls peak sensitivity.

### `POST /predict/scenes`
Groups consecutive segments into emotional scenes (Tension / Climax / Comic Relief / Joyful / Sadness / Mystery / Calm / Mixed). Each scene gets duration, dominant emotion, peak moment, and (optionally, `generate_titles=true`) an AI-written title.

### `POST /predict/trailer`
Takes the `scenes` array and returns the top N clips for a target trailer duration. Enforces mood diversity (won't pick two tense clips), caps each clip at `target_clip_duration`, and (optionally, `generate_taglines=true`) AI-writes a tagline per clip.

### `POST /predict/score`
Producer-facing scorecard. Six 0..1 axes (engagement, drama, comedy, intensity, emotional_range, pacing), a letter grade (A+ -> C), and an AI-written classification + target audience based on the combination of signals plus cast/scene context.

### `POST /predict/shifts`
Finds turning points where the dominant emotion pivots. Uses a hand-tuned emotion-distance table so opposites (happiness vs sadness) score higher than near-neighbours (anger vs fear). Filters out low-confidence noise via `min_magnitude` + `min_confidence`.

### `POST /predict/persuasion`
Combines rhetorical strength (avg confidence), emotional appeal (% strong-emotion segments), consistency (top emotion share), and variety (normalized entropy) into a single 0..1 score with a Very Low -> Very High label. AI assessment writes a 2-3 sentence qualitative paragraph.

### Auto-enriched endpoints
`POST /predict/pipeline` and `POST /predict/youtube` now also return `cast` and `timeline` in the same JSON response, no extra round-trip needed for those two features.

---

## AI usage (Groq, free tier)

The Groq LLM is called in five places, with a shared lazy-loaded singleton client (`analysis/autotitle.py::_get_client`). The API key is read from a `.env` file at the repo root and **never committed** (the file is in `.gitignore`).

| Where | What it produces | Fallback if Groq fails |
|---|---|---|
| `/compare` | Auto-title for each side | `"Show A"` / `"Show B"` |
| `/predict/scenes` (when `generate_titles=true`) | One AI title per scene | `"Scene N"` |
| `/predict/trailer` (when `generate_taglines=true`) | Tagline per clip | `"A {mood} moment."` |
| `/predict/score` (when `use_ai_classification=true`) | Classification + target audience | Heuristic rule-based classifier |
| `/predict/persuasion` (when `use_ai_assessment=true`) | 2-3 sentence qualitative paragraph | Deterministic template |

Every AI call has a **try/except with a graceful fallback**, the endpoint never breaks when the LLM is unreachable.

---

## Data flow

```
                +----------------------------------------------+
                |  Input: Text / Audio / Video / CSV / YouTube |
                +--------------------+-------------------------+
                                     |
                +--------------------▼-------------------------+
                |   /predict/pipeline  or  /predict/youtube    |
                |   (Whisper -> MarianMT -> DeBERTa per segment) |
                +--------------------+-------------------------+
                                     |
       +-----------------------------+----------------------------+
       | Auto in response:           |   On-demand via endpoints: |
       |   - segments                |     /predict/cast          |
       |   - cast  (NER)             |     /predict/scenes        |
       |   - timeline                |     /predict/trailer       |
       +-----------------------------+     /predict/score         |
                                     |     /predict/shifts        |
                                     |     /predict/persuasion    |
                                     |     /compare (left+right)  |
                                     +----------------------------+
                                                  |
                                                  ▼
                                     +------------------------+
                                     |  Groq LLM (Llama-3.1)  |
                                     |, titles, taglines,   |
                                     |    classifications,    |
                                     |    persuasion summary  |
                                     +------------------------+
```

---

## Run locally

```powershell
# Activate the project venv
cd <repo-root>
.\.venv311\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

# Make sure GROQ_API_KEY is in .env (BOM-free UTF-8):
#   GROQ_API_KEY=gsk_...
# .env is gitignored.

# Start the API
uvicorn emotion_mlops.api.app:app --reload --port 8000

# Open the interactive Swagger UI
# http://localhost:8000/docs
```

---

## Security note

A real Groq API key was accidentally pasted once during development. It was **immediately revoked** through the Groq console and a fresh key was issued and placed only in `.env`. From that point on every dev test was run from `.env` rather than from inline values. `.env` is in `.gitignore`, so no key has ever reached the remote repository.

---

## Next sprint (frontend)

All eight features are designed to be consumed by a single Compare-focused frontend page with two parallel input panels (one per side, each supporting the four input types) and a full dashboard below: radar overlay, bar chart, winner badges, character comparison, timeline, trailer clip cards, content-score gauges, shift annotations, and a persuasion meter.

Backend is feature-complete; frontend will be written from scratch in one pass against this stable API.