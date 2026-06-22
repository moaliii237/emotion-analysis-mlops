# Package Plan, `emotion_mlops`

## Overview

`emotion_mlops` is a production-grade NLP package for sentence-level emotion
classification of TV transcripts. It supports audio/video input, speech-to-text
transcription, English-to-Dutch translation, and emotion classification via a
fine-tuned DeBERTa model. The package is designed for local, on-premise, and
cloud deployment.

**Version:** 0.1.0  
**Python:** 3.11+  
**License:** MIT

---

## Module Structure

```
src/emotion_mlops/
+-- __init__.py
+-- analysis/                   # Sprint 4, analytical features
|   +-- autotitle.py            # AI-generated scene/clip titles (Groq)
|   +-- cast_analyzer.py        # Named-character emotion profiling (spaCy NER)
|   +-- compare.py              # Side-by-side comparison of two pipeline results
|   +-- content_score.py        # Producer-style scorecard (grade, axes, audience)
|   +-- per_character.py        # Per-speaker emotion summary and top moments
|   +-- persuasion.py           # Rhetorical strength and emotional appeal scoring
|   +-- scenes.py               # Auto-detected emotional scene segmentation
|   +-- shifts.py               # Emotional turning-point detection
|   +-- summary.py              # AI 2-3 sentence verdict (Groq Llama)
|   +-- timeline.py             # Time-series emotion intensity chart data
|   +-- trailer.py              # Best-clip selection for short trailers
|   +-- __init__.py
+-- api/                        # FastAPI backend
|   +-- app.py                  # Main application, lifespan, all endpoints
|   +-- database.py             # PostgreSQL prediction logging
|   +-- schemas.py              # Pydantic request/response models
|   +-- __init__.py
+-- azure/                      # Azure ML integration
|   +-- config.py               # Azure config loader (azure_config.yaml)
|   +-- connect.py              # MLClient factory, workspace connection
|   +-- data_assets.py          # Data asset registration in Azure ML
|   +-- deploy_batch.py         # Batch endpoint deployment
|   +-- deploy_endpoint.py      # Online endpoint deployment (blue/green)
|   +-- environments.py         # Azure ML environment registration
|   +-- pipeline.py             # Training pipeline definition and submission
|   +-- register_best.py        # Register best model from sweep
|   +-- schedule.py             # Weekly retraining schedule
|   +-- score.py                # Scoring script for Azure ML endpoint
|   +-- sweep.py                # Hyperparameter sweep definition
|   +-- sweep_grid.py           # Sweep grid configuration
|   +-- train_hybrid.py         # Hybrid local + cloud training
+-- cli/                        # Typer CLI
|   +-- commands.py             # CLI command definitions
|   +-- infer.py                # Full pipeline CLI entry point
|   +-- main.py                 # CLI app factory
|   +-- __init__.py
+-- data/                       # Data pipeline
|   +-- data_ingestion.py       # Raw data loading and validation
|   +-- data_preprocessing.py   # Label mapping, cleaning, normalisation
|   +-- data_splitting.py       # Train/val/test split with stratification
|   +-- data_validation.py      # Schema and distribution checks
|   +-- __init__.py
+-- diarization/                # Speaker diarization
|   +-- diarizer.py             # pyannote.audio speaker turn detection
|   +-- __init__.py
+-- features/                   # Feature engineering
|   +-- feature_extraction.py   # Feature extraction utilities
|   +-- feature_scaling.py      # Scaling and normalisation
|   +-- feature_selection.py    # Feature selection helpers
|   +-- __init__.py
+-- feedback/                   # Continuous retraining flywheel (ILO 9.5D)
|   +-- auto_retrain.py         # Threshold-triggered retrain logic
|   +-- db.py                   # Feedback storage (PostgreSQL + SQLite fallback)
|   +-- retrain_trigger.py      # CLI/API retrain trigger with dry-run mode
|   +-- __init__.py
+-- frontend/
|   +-- app.py                  # Legacy Streamlit frontend
+-- models/                     # ML models
|   +-- classifier.py           # EmotionClassifier, DeBERTa inference wrapper
|   +-- model_definitions.py    # Architecture constants and label maps
|   +-- model_evaluation.py     # Evaluation script (F1, accuracy, confusion matrix)
|   +-- model_saving.py         # Save/load model artifacts
|   +-- model_training.py       # Fine-tuning script (WeightedTrainer, MLflow)
|   +-- model_tuning.py         # Hyperparameter tuning utilities
|   +-- transcriber.py          # Whisper ASR wrapper
|   +-- translator.py           # MarianMT EN->NL translation wrapper
|   +-- __init__.py
+-- monitoring/                 # Observability
|   +-- drift.py                # Data drift detection
|   +-- metrics.py              # Prometheus metrics (latency, prediction counts)
|   +-- __init__.py
+-- qa/                         # LLM Q&A over transcripts
|   +-- groq_qa.py              # Groq-hosted Llama Q&A endpoint
|   +-- __init__.py
+-- static/                     # Frontend assets (served by FastAPI StaticFiles)
|   +-- index.html              # Main SPA entry point
|   +-- app.js                  # Sprint 4 frontend (routing, all features)
|   +-- app.css                 # Sprint 4 styles
+-- summary/                    # Text summarisation
|   +-- summarizer.py           # distilbart-cnn-6-6 summarisation
|   +-- __init__.py
+-- tracking/
|   +-- experiment_tracking.py  # MLflow experiment tracking utilities
+-- ui/
|   +-- app.py                  # Streamlit UI (legacy)
+-- utils/                      # Shared utilities
    +-- configuration.py        # Config file loader (YAML)
    +-- helpers.py              # Shared helper functions
    +-- visualization.py        # Plotting utilities
    +-- __init__.py
```

---

## Public Interfaces

### `emotion_mlops.models.classifier`

```python
class EmotionClassifier:
    def __init__(model_path: str, model_version: str = "")
    def load() -> None
    def predict(text: str) -> tuple[str, float]
    def predict_all(text: str) -> dict[str, float]
    @property
    def is_loaded(self) -> bool
    @property
    def model_version(self) -> str
```

### `emotion_mlops.api.database`

```python
def init_db() -> None
def log_prediction(text: str, emotion: str, confidence: float, source: str) -> None
def get_recent_predictions(limit: int = 100) -> list[dict]
```

### `emotion_mlops.feedback.db`

```python
def init_db() -> None
def record_feedback(sentence, predicted_label, predicted_conf, is_correct,
                    model_version, user_label, source, file_id) -> int
def get_stats() -> dict
def export_corrections_for_retraining() -> list[dict]
def get_retrain_history(limit: int = 20) -> list[dict]
def should_retrain(threshold: int = 20) -> dict
```

### `emotion_mlops.api.app`, REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Static frontend (SPA) |
| `GET` | `/api` | API metadata and endpoint list |
| `GET` | `/health` | Model load status and version |
| `GET` | `/endpoint/info` | Azure ML endpoint name and traffic split |
| `GET` | `/predictions` | Recent predictions from database |
| `POST` | `/predict/text` | Single-sentence prediction (Azure ML endpoint) |
| `POST` | `/predict/text/full` | Single-sentence prediction with all label probabilities |
| `POST` | `/predict/cloud` | Explicit Azure ML endpoint inference |
| `POST` | `/predict/pipeline` | Full pipeline: audio/CSV -> emotions |
| `POST` | `/predict/youtube` | YouTube URL -> download -> pipeline |
| `POST` | `/predict/cast` | Named-character emotion profiling |
| `POST` | `/predict/timeline` | Emotion intensity time-series |
| `POST` | `/predict/scenes` | Emotional scene segmentation |
| `POST` | `/predict/trailer` | Best-clip selection for trailers |
| `POST` | `/predict/score` | Producer-style content scorecard |
| `POST` | `/predict/shifts` | Emotional turning-point detection |
| `POST` | `/predict/persuasion` | Rhetorical strength analysis |
| `POST` | `/predict/summary` | AI 2-3 sentence narrative verdict |
| `POST` | `/compare` | Side-by-side comparison of two results |
| `POST` | `/summarize` | Transcript summarisation |
| `POST` | `/ask` | Free-form Q&A about a transcript (Groq LLM) |
| `POST` | `/feedback` | Record user feedback on a prediction |
| `GET` | `/feedback/stats` | Aggregated feedback statistics |
| `GET` | `/feedback/export` | Export corrections for retraining |
| `GET` | `/feedback/auto-check` | Flywheel progress (corrections vs threshold) |
| `POST` | `/feedback/retrain` | Trigger retraining flywheel |

### `emotion_mlops.cli`

```bash
# Full pipeline on audio file
emotion-pipeline --config configs/config.yaml --audio data/audio.wav --out results.csv

# Full pipeline on transcript CSV
emotion-pipeline --config configs/config.yaml --transcript_csv data/transcript.csv --out results.csv

# Typer CLI
emotion-mlops predict --input transcript.txt
emotion-mlops train --data training_data.csv
```

### `emotion_mlops.azure.pipeline`

```python
def build_pipeline_job(
    data_asset: str = "emotion-raw",
    data_version: str = "1",
    epochs: int = 1,
    batch_size: int = 4,
    max_steps: int = 100,
    learning_rate: float = 2e-5,
    run_name: str = "azure-run",
    display_name: str = "emotion-training-pipeline",
) -> PipelineJob

def build_and_submit_pipeline(**kwargs) -> PipelineJob
```

---

## Dependencies

### Core (production)

| Package | Version | Purpose |
|---|---|---|
| `transformers` | ≥4.40.0 | DeBERTa, MarianMT models |
| `openai-whisper` | latest | Speech-to-text transcription |
| `torch` | ≥2.0.0 | PyTorch CPU/GPU inference |
| `fastapi` | ≥0.110.0 | REST API framework |
| `uvicorn[standard]` | ≥0.29.0 | ASGI server |
| `psycopg2-binary` | latest | PostgreSQL driver |
| `pydantic` | ≥2.0.0 | Data validation and schemas |
| `azure-ai-ml` | ≥1.13.0 | Azure ML SDK v2 |
| `azure-identity` | ≥1.15.0 | Azure authentication |
| `sentencepiece` | ≥0.2.0 | MarianMT tokenizer |
| `sacremoses` | latest | Text normalisation |
| `yt-dlp` | ≥2024.1.0 | YouTube audio download |
| `prometheus-fastapi-instrumentator` | latest | Prometheus metrics |
| `scikit-learn` | ≥1.4.0 | Metrics, class weights |
| `pandas` | ≥2.0.0 | Data manipulation |
| `typer` | ≥0.12.0 | CLI framework |
| `groq` | latest | Groq LLM API (Q&A, scene titles, verdict) |
| `spacy` | latest | NER for cast detection |
| `mlflow` | ≥2.12.0 | Experiment tracking |

### Development only

| Package | Purpose |
|---|---|
| `pytest` + `pytest-cov` | Testing and coverage |
| `black` | Code formatting |
| `flake8` | Linting |
| `isort` | Import sorting |
| `build` | Package building |

---

## Entry Points

Defined in `pyproject.toml`:

```toml
[project.scripts]
emotion-pipeline = "emotion_mlops.cli.infer:main"
emotion-mlops    = "emotion_mlops.cli.main:app"
```

---

## Configuration

Runtime settings in `configs/config.yaml`:

```yaml
device: cpu
whisper_model: base
mt_en_nl_model: Helsinki-NLP/opus-mt-en-nl
best_model_dir: models/best_model_v3
batch_size: 16
max_length: 192
```

Azure ML settings in `configs/azure_config.yaml`:

```yaml
workspace_name: <AZURE_WORKSPACE_NAME>
resource_group: <AZURE_RESOURCE_GROUP>
endpoint:
  name: emotion-classifier-endpoint
  traffic:
    blue: 60
    green: 40
```

Secrets (never committed to git):

| Variable | Where set | Purpose |
|---|---|---|
| `ML_ENDPOINT_URL` | env / Portainer / Container Apps | Azure ML endpoint URL |
| `ML_ENDPOINT_KEY` | env / Portainer / Container Apps | Azure ML endpoint key |
| `DATABASE_URL` | env / Portainer / Container Apps | PostgreSQL connection string |
| `GROQ_API_KEY` | env / Portainer / Container Apps | Groq LLM API key |
| `AUTO_RETRAIN_ENABLED` | env / Portainer | Enable auto-retrain flywheel (`1`/`0`) |

---

## Build and Distribution

```bash
# Build wheel and source distribution
python -m build

# Output:
# dist/emotion_mlops-0.1.0-py3-none-any.whl
# dist/emotion_mlops-0.1.0.tar.gz

# Install from wheel
pip install dist/emotion_mlops-0.1.0-py3-none-any.whl

# Install in editable mode for development
pip install -e .
```

---

## Testing

```bash
# Run unit tests with coverage
pytest tests/unit/ --cov=src/emotion_mlops --cov-report=term-missing

# Run integration tests
pytest tests/integration/ -v

# Run all tests
pytest -v
```

| Test file | Coverage |
|---|---|
| `tests/unit/` | FastAPI endpoints, schemas, classifier, feedback DB |
| `tests/integration/` | Sprint 4 endpoints, full pipeline |
| `conftest.py` | Shared fixtures (mock classifier, test client) |