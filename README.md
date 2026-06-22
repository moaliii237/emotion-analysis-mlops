## Team and my contribution

This was a 7-person Year 2 group project at Breda University. The lead developer was Zofia Pilitowska, with Ceren Kurt, Asma Moghimi, Arayah Coppens and others on the team. My part was the continuous-retraining feedback loop (the feedback module and the auto-retrain logic) and the command-line tool. I did not write the FastAPI app, the Azure deployment, the frontend or the test suite. I publish the full project here for my portfolio, with credit to the whole team.

---

# TV Emotion Analysis MLOps Application

A production-oriented NLP application for transcript-based emotion analysis of TV content.
This project extends our Block A prototype into a modular, deployable, and maintainable MLOps system for local, on-premise, and cloud deployment.

## Project Goal

Transform our Block A emotion-classification prototype into a production-ready MLOps application. The system processes audio, video, or transcript-based input and generates structured sentence-level emotion predictions. Block D focuses on packaging, deployment, monitoring, retraining, and documentation.

## Stakeholder Context

The primary stakeholder is our mentor / product owner in Block D. The use case is emotion analysis for TV transcripts, demonstrating the full MLOps lifecycle.

## Current Prototype

The pipeline supports:
- audio/video input
- speech-to-text transcription using Whisper
- English-to-Dutch translation using MarianMT
- sentence-level emotion classification using a fine-tuned DeBERTa model
- CSV output generation

---

- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [CLI](#cli)
- [API](#api)
- [Database](#database)
- [Feedback & Continuous Retraining](#feedback--continuous-retraining)
- [UI](#ui)
- [Logging](#logging)
- [Unit Tests](#unit-tests)
- [Docker](#docker)
- [Airflow](#airflow)
- [Portainer, On-Premise Deployment](#portainer--on-premise-deployment)
- [Local Training](#local-training)
- [Azure ML, MLOps Pipeline](#azure-ml--mlops-pipeline)
- [Cloud Deployment](#cloud-deployment)
- [CI/CD](#cicd)
- [Service URLs](#service-urls)
- [Emotion Labels](#emotion-labels)
- [Model Information](#model-information)
- [Contributing](#contributing)

---
## Documentation

Full API documentation is generated with Sphinx and published automatically to GitHub Pages on every push to `main`.

## Repository Structure

```text
2025-26D-FAI2-ADSAI-GROUP-NLP-GROUP-1/
  configs/
    config.yaml
    azure_config.yaml
    conda.yaml
  dags/
    data_preprocessing_dag.py
    model_training_dag.py
    retrain_dag.py
    feedback_retrain_dag.py
  data/
    data_test/
  dist/
    emotion_mlops-0.1.0-py3-none-any.whl
    emotion_mlops-0.1.0.tar.gz
  docs/
    package_plan.md
    source/                  # Sphinx docs source (build/ is generated)
  logs/
  models/
    best_model_v3/
  monitoring/
    grafana/
    prometheus/
  src/
    emotion_mlops/
      analysis/
      api/
        app.py
        database.py
        schemas.py
      azure/
      cli/
      data/
      feedback/
        auto_retrain.py      # Threshold-triggered auto-retrain logic
        db.py                # Feedback storage (PostgreSQL + SQLite fallback)
        retrain_trigger.py   # CLI/API retrain trigger with dry-run mode
        __init__.py
      models/
      monitoring/
      qa/
      static/                # Frontend HTML/CSS/JS
      summary/
      utils/
  tests/
    unit/
      conftest.py
      test_api.py
      test_classifier_real.py
      test_schemas.py
      test_dags.py
    integration/
      conftest.py
      test_sprint4_endpoints.py
  .github/workflows/
  Dockerfile
  Dockerfile.frontend
  docker-compose.yml
  docker-compose.server.yml
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  README.md
```

---

## Setup

### Prerequisites

- Python 3.11
- Docker Desktop
- Azure CLI (`az login` with BUas credentials)
- Git

### Install

```bash
git clone https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-nlp-group-1.git
cd 2025-26d-fai2-adsai-group-nlp-group-1

python -m venv .venv311

# Windows
.\.venv311\Scripts\Activate.ps1

# macOS/Linux
source .venv311/bin/activate

# 1. Install PyTorch (CPU), must come before requirements.txt
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    --only-binary=:all:

# 2. Install production dependencies
pip install -r requirements.txt

# 3. Install dev dependencies (MLflow, Airflow, matplotlib)
pip install -r requirements-dev.txt

# 4. Install Whisper separately (no binary wheel)
pip install openai-whisper

# 5. Install the package in editable mode
pip install -e .
```

### Install from wheel

```bash
pip install dist/emotion_mlops-0.1.0-py3-none-any.whl
```

### Environment variables

Create a `.env` file (never commit) or set in your shell:

```bash
export ML_ENDPOINT_URL="https://emotion-classifier-endpoint.westeurope.inference.ml.azure.com/score"
export ML_ENDPOINT_KEY="your-key-here"
export DATABASE_URL="postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope"
```

Windows PowerShell:
```powershell
$env:ML_ENDPOINT_URL="https://emotion-classifier-endpoint.westeurope.inference.ml.azure.com/score"
$env:ML_ENDPOINT_KEY="your-key-here"
$env:DATABASE_URL="postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope"
```

### Azure config

Create `config.json` at the repo root (do not commit, already in `.gitignore`):

```json
{
  "subscription_id": "<AZURE_SUBSCRIPTION_ID>",
  "resource_group": "<AZURE_RESOURCE_GROUP>",
  "workspace_name": "<AZURE_WORKSPACE_NAME>",
  "tenant_id": "YOUR_TENANT_ID"
}
```

---

## Configuration

All pipeline settings are in `configs/config.yaml`:

```yaml
device: cpu               # cpu or cuda
whisper_model: base       # tiny | base | small | medium
mt_en_nl_model: Helsinki-NLP/opus-mt-en-nl
best_model_dir: models/best_model_v3
batch_size: 16
max_length: 192
```

| Field | Description |
|---|---|
| `device` | Inference device. Use `cuda` if a GPU is available. |
| `whisper_model` | Whisper model size. `base` recommended for CPU. |
| `mt_en_nl_model` | HuggingFace model ID for EN->NL translation. |
| `best_model_dir` | Path to the fine-tuned DeBERTa model directory. |
| `batch_size` | Batch size for translation and classification. |
| `max_length` | Maximum token length for classification. |

---

## CLI

Three entry points are available after `pip install -e .`:

### Run the full pipeline on an audio file

```bash
emotion-pipeline \
    --config configs/config.yaml \
    --audio data/audio.wav \
    --out results.csv
```

### Run on an existing transcript CSV

```bash
emotion-pipeline \
    --config configs/config.yaml \
    --transcript_csv data/data_test/group_24_url_1_transcript.csv \
    --out data/data_test/group_24_url_1_results.csv
```

The transcript CSV must contain a `Sentence` (or `Text` / `Utterance`) column. `Start` and `End` columns are optional.

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file` | `logs/pipeline.log` | Path to log output file |

### Typer commands (after install)

```bash
emotion-mlops predict --input transcript.txt
emotion-mlops train --data training_data.csv
```

### Without installing the package

```powershell
$env:PYTHONPATH="src"
python -m emotion_mlops.cli.infer `
    --config configs/config.yaml `
    --transcript_csv data/data_test/group_24_url_1_transcript.csv `
    --out results.csv
```

---

## API

The FastAPI backend exposes the pipeline over HTTP.

### Start the backend locally

```powershell
$env:PYTHONPATH="src"
$env:ML_ENDPOINT_URL="https://emotion-classifier-endpoint.westeurope.inference.ml.azure.com/score"
$env:ML_ENDPOINT_KEY="your-key-here"
$env:DATABASE_URL="postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope"
python -m uvicorn emotion_mlops.api.app:app --port 8000 --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Frontend UI (static HTML) |
| `GET` | `/api` | API info and available endpoints |
| `GET` | `/health` | Model load status and version |
| `GET` | `/endpoint/info` | Azure ML endpoint traffic split and URI |
| `GET` | `/predictions` | Recent predictions from database |
| `POST` | `/predict/text` | Predict emotion from a single sentence (local model) |
| `POST` | `/predict/text/full` | Predict emotion with all label probabilities |
| `POST` | `/predict/cloud` | Predict via Azure ML Online Endpoint |
| `POST` | `/predict/pipeline` | Full pipeline: audio/CSV -> emotions |
| `POST` | `/predict/youtube` | Full pipeline from YouTube URL |
| `POST` | `/predict/timeline` | Build emotion timeline from segments |
| `POST` | `/predict/scenes` | Detect emotional scenes in transcript |
| `POST` | `/predict/trailer` | Auto-pick best clips for trailer |
| `POST` | `/predict/score` | Content scorecard for producers |
| `POST` | `/predict/shifts` | Detect emotional turning points |
| `POST` | `/predict/persuasion` | Score speaker persuasive power |
| `POST` | `/predict/cast` | Character emotion profiling via NER |
| `POST` | `/predict/summary` | AI verdict from full analysis |
| `POST` | `/compare` | Compare two pipeline results side-by-side |
| `POST` | `/feedback` | Record user feedback on a prediction |
| `GET` | `/feedback/stats` | Aggregated feedback statistics |
| `GET` | `/feedback/export` | Export user corrections for retraining |
| `GET` | `/feedback/auto-check` | Flywheel progress (corrections vs threshold) |
| `POST` | `/feedback/retrain` | Trigger retraining from feedback |
| `POST` | `/summarize` | Summarize transcript text |
| `POST` | `/ask` | Q&A about transcript using Groq LLM |

### Examples

```python
import requests, json

# Predict via cloud endpoint
r = requests.post('http://localhost:8000/predict/cloud',
    json={'text': 'I am so happy today!'})
print(json.dumps(r.json(), indent=2))
# {"text": "...", "emotion": "happiness", "confidence": 0.9818, "source": "azure-ml-endpoint"}

# Submit feedback
r = requests.post('http://localhost:8000/feedback', json={
    'sentence': 'I am so happy today!',
    'predicted_label': 'happiness',
    'predicted_conf': 0.98,
    'is_correct': True,
    'model_version': 'best_model_v3',
    'source': 'text'
})

# View recent predictions from database
r = requests.get('http://localhost:8000/predictions')
print(json.dumps(r.json(), indent=2))
```

---

## Database

All predictions and feedback are persisted in PostgreSQL automatically.

### Predictions table

Every call to `/predict/cloud` and `/predict/text` logs a row:

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL | Auto-incremented primary key |
| `timestamp` | TIMESTAMP | UTC time of prediction |
| `text` | TEXT | Input sentence |
| `emotion` | VARCHAR | Predicted emotion label |
| `confidence` | FLOAT | Model confidence (0-1) |
| `source` | VARCHAR | `local` or `azure-ml-endpoint` |

### Feedback table

Every call to `/feedback` logs a row:

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL | Auto-incremented primary key |
| `timestamp` | TIMESTAMP | UTC time of feedback |
| `sentence` | TEXT | Original input sentence |
| `predicted_label` | VARCHAR | What the model predicted |
| `predicted_conf` | FLOAT | Model confidence |
| `user_label` | VARCHAR | User's corrected label (if `is_correct=False`) |
| `is_correct` | INTEGER | 1 if user agreed, 0 if not |
| `model_version` | VARCHAR | Which model produced the prediction |
| `source` | VARCHAR | `audio`, `transcript`, or `text` |

### Local development database

```bash
docker-compose up -d postgres
```

Connection string:
```
DATABASE_URL=postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope
```

### Azure managed Postgres

Shared BUas server: `pg-students-y2d-2026.postgres.database.azure.com`

The `DATABASE_URL` is stored as a secret in Azure Container Apps and as an environment variable in Portainer, never in code or git.

```powershell
# Set DATABASE_URL as a secret in Container Apps
az containerapp secret set `
  --name nlp1-backend `
  --resource-group <AZURE_RESOURCE_GROUP> `
  --secrets db-url="YOUR_CONNECTION_STRING"

az containerapp update `
  --name nlp1-backend `
  --resource-group <AZURE_RESOURCE_GROUP> `
  --set-env-vars DATABASE_URL=secretref:db-url
```

---

## Feedback & Continuous Retraining

The application implements a full data flywheel for continuous retraining

### How it works

```
user submits feedback -> POST /feedback -> PostgreSQL feedback table
                                              v
                    Airflow DAG: emotion_feedback_retrain (daily 06:00)
                                              v
                    check_corrections_threshold (>=10 user corrections?)
                                    v yes
                    export corrections CSV + merge with GoEmotions raw data
                                              v
                    Azure ML: upload new versioned data asset
                                              v
                    Azure ML: submit training pipeline
                                              v
                    register_best_model -> deploy as green endpoint
```

### Submit feedback via API

```python
import requests

# User confirms prediction is correct
requests.post('http://localhost:8000/feedback', json={
    'sentence': 'I hate this show!',
    'predicted_label': 'anger',
    'predicted_conf': 0.91,
    'is_correct': True,
    'model_version': 'best_model_v3',
    'source': 'text'
})

# User corrects a wrong prediction
requests.post('http://localhost:8000/feedback', json={
    'sentence': 'Oh wow, I did not see that coming.',
    'predicted_label': 'happiness',
    'predicted_conf': 0.73,
    'is_correct': False,
    'user_label': 'surprise',        # user's correction
    'model_version': 'best_model_v3',
    'source': 'text'
})
```

### View feedback stats

```python
r = requests.get('http://localhost:8000/feedback/stats')
# {"total": 42, "agreement_rate": 0.857, "per_model": [...]}
```

### Automated retraining

The retraining pipeline runs automatically in two ways:

**1. Daily schedule (GitHub Actions)**

Every day at 06:00 UTC, the `retrain.yml` workflow checks whether the feedback
threshold has been reached (>= 10 new user corrections since the last retrain).
If yes, it automatically exports corrections, merges with raw training data,
uploads a new versioned data asset to Azure ML, and submits a training pipeline job.
No manual intervention required.

**2. Manual trigger**

```
GitHub -> Actions -> Retrain Pipeline -> Run workflow
model_version: v3
dry_run: false
```

Or via API (dry run, no Azure costs):
```python
requests.post('http://<SERVER_HOST>:2002/feedback/retrain?dry_run=true')
```

### Retraining flow

```
user clicks thumbs-down and selects correct label
        v
POST /feedback -> PostgreSQL feedback table
        v
GitHub Actions: retrain.yml (daily 06:00 UTC)
        v
check_corrections_threshold (>= 10 new corrections?)
        v yes
export corrections CSV + merge with raw data
        v
Azure ML: upload new versioned data asset (emotion-raw vYYYYMMDD)
        v
Azure ML: submit training pipeline (preprocess -> train -> evaluate)
        v
register best model -> deploy as green endpoint (blue/green strategy)
```

### Monitor flywheel progress

```python
r = requests.get('http://<SERVER_HOST>:2002/feedback/auto-check')
# {"new_corrections": 13, "threshold": 10, "ready": true, "history": [...]}
```

Or in the UI: **Features -> Monitor** -> Retrain Flywheel section.

---

## UI

The frontend is a static web application served directly by the FastAPI backend.

### Start locally

```powershell
$env:PYTHONPATH="src"
$env:ML_ENDPOINT_URL="..."
$env:ML_ENDPOINT_KEY="..."
$env:DATABASE_URL="..."
python -m uvicorn emotion_mlops.api.app:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

### Pages

| Page | Description |
|---|---|
| **Home** | Landing page with pipeline overview and feature highlights |
| **Analyse** | Run the pipeline on text, audio, transcript CSV, or YouTube URL |
| **Compare** | Side-by-side comparison of two analyses |
| **History** | Log of all pipeline runs stored in the browser |
| **About** | Project and team information |
| **Features -> Cast** | Named-character emotion profiling |
| **Features -> Timeline** | Emotion intensity time-series chart |
| **Features -> Scenes** | Auto-detected emotional scene segmentation |
| **Features -> Trailer** | Best-clip selection for trailers |
| **Features -> Score** | Producer-style content scorecard |
| **Features -> Shifts** | Emotional turning-point detection |
| **Features -> Persuasion** | Rhetorical strength analysis |
| **Features -> Monitor** | Feedback flywheel, model monitoring, and pipeline performance |

---

## Logging

Log files are written to the `logs/` directory at the repo root.

| File | Written by | Contains |
|---|---|---|
| `logs/api.log` | `src/emotion_mlops/api/app.py` | Model loading, request handling, pipeline timings |
| `logs/pipeline.log` | `src/emotion_mlops/cli/infer.py` | CLI pipeline runs, transcription, translation, classification |
| `logs/azure.log` | `src/emotion_mlops/azure/` | Azure ML job submissions, deployments |

### Log format

```
2026-05-11 09:16:44 | INFO     | emotion_mlops.api.app | [API] Whisper ready.
```

---

## Unit Tests

Tests are in the `tests/` directory and use `pytest`.

| File | What it tests |
|---|---|
| `tests/unit/test_api.py` | FastAPI endpoints (`/health`, `/predict/text`, `/predict/pipeline`) |
| `tests/unit/test_classifier_real.py` | DeBERTa classifier loading and prediction |
| `tests/unit/test_schemas.py` | Pydantic request/response schema validation |
| `tests/unit/test_dags.py` | Airflow DAG structure and task order |
| `tests/integration/test_sprint4_endpoints.py` | Sprint 4 endpoints, full pipeline |
| `tests/unit/conftest.py` | Shared fixtures (mock classifier, test client) |

### Run unit tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=src/emotion_mlops --cov-report=term-missing
```

### Run integration tests (requires running services)

```bash
# Tests on-prem, Azure Container Apps, monitoring, and Azure ML endpoint
python tests/integration_test_services.py
```

---

## Docker

### Local development (all services)

```bash
docker-compose up -d
```

| Service | Local URL | Description |
|---|---|---|
| `backend` | http://localhost:2002 | FastAPI API + Frontend UI |
| `postgres` | localhost:5432 | PostgreSQL database |
| `prometheus` | http://localhost:2003 | Metrics scraper |
| `grafana` | http://localhost:2004 (admin/admin) | Metrics dashboard |
| `airflow` | http://localhost:2005 (admin/admin) | Pipeline orchestration |
| `mlflow` | http://localhost:2006 | Experiment tracking |

### Start only what you need

```bash
# Just the database
docker-compose up -d postgres

# Backend + database
docker-compose up -d postgres backend
```

### Build images locally

```bash
docker build -f Dockerfile -t nlp-backend .
docker build -f Dockerfile.frontend -t nlp-frontend .
```

### Stop all services

```bash
docker-compose down
```

---
## Airflow

Apache Airflow orchestrates all data and retraining pipelines by triggering GitHub Actions workflows.

### Run locally

```powershell
docker run -d -p 2005:8080 `
  -v ${PWD}/dags:/opt/airflow/dags `
  -v ${PWD}/airflow-db:/opt/airflow `
  -e AIRFLOW__CORE__EXECUTOR=SequentialExecutor `
  -e AIRFLOW__CORE__LOAD_EXAMPLES=False `
  -e _AIRFLOW_WWW_USER_CREATE=true `
  -e _AIRFLOW_WWW_USER_USERNAME=admin `
  -e _AIRFLOW_WWW_USER_PASSWORD=admin `
  -e GITHUB_TOKEN="$env:GITHUB_TOKEN" `
  apache/airflow:2.9.0 standalone
Start-Sleep -Seconds 30
docker exec $(docker ps -q --filter "ancestor=apache/airflow:2.9.0") airflow users reset-password --username admin --password admin
Write-Host "Airflow ready at http://localhost:2005 - login: admin/admin"
```

Open http://localhost:2005, login: admin/admin

> Note: Airflow generates a random password on first run. Reset it with the command above.

### DAGs

| DAG | Schedule | Triggers | Description |
|---|---|---|---|
| `emotion_data_preprocessing` | manual | `data_preprocessing.yml` | Preprocess GoEmotions CSV and upload to Azure ML as versioned data asset |
| `emotion_model_training` | Monday 08:00 | `model_training.yml` | Submit Azure ML training pipeline -> register best model -> deploy as green |
| `emotion_full_retrain` | Sunday 00:00 | `data_preprocessing.yml` -> `model_training.yml` | Full weekly retraining: preprocess + train + register + deploy |
| `emotion_feedback_retrain` | Daily 06:00 | `retrain.yml` | Feedback flywheel, triggers retraining when user corrections >= 10 |

### How DAGs work

Each DAG triggers a GitHub Actions workflow via the GitHub API. This means:
- No Azure credentials needed in Airflow
- Full pipeline runs in GitHub Actions with OIDC authentication
- DAG logs show HTTP 204 on success

### Required environment variable

`GITHUB_TOKEN`, Personal Access Token with `workflow` scope. Set in Portainer stack env vars or locally before running Docker.

### On server (Portainer)

Airflow runs as part of the Portainer stack at http://<SERVER_HOST>:2005 (admin/admin).
---

## Portainer, On-Premise Deployment

The application runs on the BUas Lambda server (`<SERVER_HOST>`) via Portainer at `http://<SERVER_HOST>:3085/`.

### Stack: `emotionscope`

Services in the stack:

| Container | Image | Port |
|---|---|---|
| `emotionscope-backend-new` | `ghcr.io/.../nlp-backend:latest` | 2002 |
| `emotionscope-postgres` | `postgres:16` | 5433 |
| `emotionscope-prometheus-new` | `prom/prometheus:latest` | 2003 |
| `emotionscope-grafana-new` | `grafana/grafana:latest` | 2004 |
| `emotionscope-airflow` | `apache/airflow:2.9.0` | 2005 |

### Automated deployment

**GitOps polling**, Portainer polls the `deploy` branch every 5 minutes and redeploys when a new image is available.

**Webhook**, GitHub Actions triggers Portainer immediately after a successful build.

### Manual redeploy

1. Open Portainer at http://<SERVER_HOST>:3085
2. Go to **Stacks** -> `emotionscope`
3. Click **Editor** -> update compose if needed -> **Update the stack**

### On-prem service URLs

| Service | URL | Credentials |
|---|---|---|
| Backend API + Frontend | http://<SERVER_HOST>:2002 |, |
| API Docs (Swagger) | http://<SERVER_HOST>:2002/docs |, |
| Recent Predictions | http://<SERVER_HOST>:2002/predictions |, |
| Prometheus | http://<SERVER_HOST>:2003 |, |
| Grafana | http://<SERVER_HOST>:2004 | admin / admin |
| Airflow | http://<SERVER_HOST>:2005 | admin / admin |
| Portainer | http://<SERVER_HOST>:3085 |, |

### What changes require a rebuild?

| Change | Action needed |
|---|---|
| Python code in `src/` | Push to `main`, CI/CD rebuilds automatically |
| `requirements.txt` | Push to `main`, CI/CD rebuilds automatically |
| `configs/config.yaml` | Restart container in Portainer |
| `docker-compose.server.yml` | Update stack in Portainer -> **Editor** tab |
| Environment variables | Portainer -> Stack -> **Environment variables** |

---

## Local Training

```powershell
# 1. Preprocess data
python -m emotion_mlops.data.data_preprocessing `
    --input data/raw/goemotions_collapsed_majority.csv `
    --output data/processed/

# 2. Train
python -m emotion_mlops.models.model_training `
    --data data/processed/ `
    --output models/trained/ `
    --epochs 3 `
    --batch-size 16 `
    --max-steps -1 `
    --learning-rate 2e-5 `
    --run-name "local-run"

# 3. Evaluate
python -m emotion_mlops.models.model_evaluation `
    --model models/trained/ `
    --data data/processed/ `
    --output data/eval/ `
    --min-f1 0.30
```

---

## Azure ML, MLOps Pipeline

### Prerequisites

- Azure CLI installed and logged in: `az login`
- `config.json` at repo root (see [Setup](#setup))

### 1. Register Environment

```powershell
python src/emotion_mlops/azure/environments.py
```

### 2. Upload Data Assets

```powershell
python -m emotion_mlops.data.data_preprocessing `
    --input data/raw/goemotions_collapsed_majority.csv `
    --output data/processed/

python src/emotion_mlops/azure/data_assets.py
```

### 3. Run Training Pipeline

```powershell
python src/emotion_mlops/azure/pipeline.py `
    --epochs 3 `
    --batch-size 16 `
    --max-steps -1 `
    --learning-rate 2e-5
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `1` | Number of training epochs |
| `--batch-size` | `4` | Training batch size |
| `--max-steps` | `100` | Max steps (-1 = full training) |
| `--learning-rate` | `2e-5` | Learning rate |

### 4. Hyperparameter Sweep

```powershell
python src/emotion_mlops/azure/sweep.py --max-trials 8 --max-concurrent 2
```

Search space: `learning_rate`  in  [5e-6, 1e-5, 2e-5, 3e-5, 5e-5], `batch_size`  in  [16, 32, 64], `epochs`  in  [2, 3, 4]

### 5. Register Best Model

```powershell
python src/emotion_mlops/azure/register_best.py
```

Registers the model with the highest macro F1 from all completed runs.

### 6. Schedule Weekly Retraining

```powershell
python src/emotion_mlops/azure/schedule.py
```

Creates a weekly Azure ML schedule that triggers the training pipeline every Monday at 08:00.

### Azure ML workspace

- Portal: https://ml.azure.com -> workspace: `<AZURE_WORKSPACE_NAME>`
- Resource group: `<AZURE_RESOURCE_GROUP>`
- Subscription: `<AZURE_SUBSCRIPTION_ID>`

---

## Cloud Deployment

### Azure Container Apps

| Service | URL |
|---|---|
| Backend API + Frontend | https://nlp1-backend.politeriver-95d366e5.northeurope.azurecontainerapps.io |
| API Docs | https://nlp1-backend.politeriver-95d366e5.northeurope.azurecontainerapps.io/docs |
| Predictions (DB) | https://nlp1-backend.politeriver-95d366e5.northeurope.azurecontainerapps.io/predictions |

> Note: The frontend UI is served directly by the backend, no separate frontend container needed.

Environment: `cae-y2d-2026-nlp1` (North Europe)

### Manual deploy to Container Apps

```powershell
# Set registry credentials
az containerapp registry set `
  --name nlp1-backend `
  --resource-group <AZURE_RESOURCE_GROUP> `
  --server ghcr.io `
  --username $env:GHCR_USERNAME `
  --password $env:GHCR_TOKEN

# Update image
az containerapp update `
  --name nlp1-backend `
  --resource-group <AZURE_RESOURCE_GROUP> `
  --image ghcr.io/bredauniversityadsai/nlp1/nlp-backend:latest `
  --min-replicas 1
```

### Azure ML Online Endpoint

| Resource | Value |
|---|---|
| Endpoint name | `emotion-classifier-endpoint` |
| Scoring URI | `https://emotion-classifier-endpoint.westeurope.inference.ml.azure.com/score` |
| Traffic | blue: 60%, green: 40% |
| Auth | Bearer token (key-based) |

### Blue-Green deployment

```powershell
# Deploy new version as green
python src/emotion_mlops/azure/deploy_endpoint.py --model-version 2 --deployment-name green

# Canary: 10% traffic to green
python src/emotion_mlops/azure/deploy_endpoint.py --set-traffic blue=90 green=10

# Promote: full traffic to green
python src/emotion_mlops/azure/deploy_endpoint.py --set-traffic blue=0 green=100

# Rollback: back to blue
python src/emotion_mlops/azure/deploy_endpoint.py --set-traffic blue=100 green=0
```

---

## CI/CD

### Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `validate-branch.yml` | push / PR | Enforces branch naming convention |
| `lint.yml` | push / PR to main or develop | Runs `black`, `isort`, `flake8` |
| `test.yml` | push / PR to main or develop | Runs `pytest` with coverage |
| `deploy.yml` | push to `main` | Builds images -> GHCR -> Portainer webhook (on-prem) |
| `deploy_container_apps.yml` | push to `main` | Builds images -> GHCR -> Azure Container Apps (OIDC) |
| `deploy_model.yml` | push to `main` (azure/ changed) / manual | Deploys model to Azure ML endpoint |
| `retrain.yml` | daily 06:00 UTC + manual | Checks feedback threshold, triggers retraining if >=10 corrections |

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `ML_ENDPOINT_KEY` | Azure ML endpoint key |
| `GHCR_USERNAME` | GitHub Container Registry username |
| `PORTAINER_WEBHOOK_URL` | Portainer stack webhook for instant deploy |

### Required GitHub Variables

| Variable | Description |
|---|---|
| `AZURE_CLIENT_ID` | Azure service principal client ID (OIDC) |
| `AZURE_TENANT_ID` | Azure tenant ID (OIDC) |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID (OIDC) |

### Full CI/CD flow on push to main

```
push to main
    v
lint.yml        -> black + isort + flake8
test.yml        -> pytest --cov
    v
deploy.yml
    - build nlp-backend -> push to GHCR
    - build nlp-frontend -> push to GHCR
    - publish-deploy branch -> Portainer polls every 5 min
    - webhook -> Portainer instant redeploy

deploy_container_apps.yml
    - build + push -> GHCR
    - Azure login (OIDC)
    - az containerapp registry set (GHCR credentials)
    - az containerapp update -> nlp1-backend + nlp1-frontend

deploy_model.yml (only if azure/ files changed)
    - deploy_endpoint.py -> emotion-classifier-endpoint
```

### Branch naming

| Prefix | Purpose | Example |
|---|---|---|
| `feature/` | New functionality | `feature/online-endpoint` |
| `fix/` | Bug fixes | `fix/oom-batch-size` |
| `chore/` | Non-functional changes | `chore/update-deps` |

---
## Service URLs

### On-Premise (BUas Lambda server `<SERVER_HOST>`)

| Service | URL | Credentials |
|---|---|---|
| Frontend + Backend API | http://<SERVER_HOST>:2002 |, |
| API Docs (Swagger) | http://<SERVER_HOST>:2002/docs |, |
| Recent Predictions | http://<SERVER_HOST>:2002/predictions |, |
| Prometheus | http://<SERVER_HOST>:2003 |, |
| Grafana | http://<SERVER_HOST>:2004 | admin / admin |
| Airflow | http://<SERVER_HOST>:2005 | admin / admin |
| Portainer | http://<SERVER_HOST>:3085 |, |

### Cloud (Azure Container Apps)

| Service | URL |
|---|---|
| Backend API + Frontend UI | https://nlp1-backend.politeriver-95d366e5.northeurope.azurecontainerapps.io |
| API Docs | https://nlp1-backend.politeriver-95d366e5.northeurope.azurecontainerapps.io/docs |
| Recent Predictions | https://nlp1-backend.politeriver-95d366e5.northeurope.azurecontainerapps.io/predictions |

> The frontend UI is served directly by the backend, no separate frontend container is needed.

### Azure ML

| Resource | URL |
|---|---|
| ML Portal | https://ml.azure.com -> workspace: `<AZURE_WORKSPACE_NAME>` |
| Online Endpoint | https://emotion-classifier-endpoint.westeurope.inference.ml.azure.com/score |

---

## Emotion Labels

| Label | Description |
|---|---|
| `anger` | Expressions of frustration or rage |
| `disgust` | Expressions of revulsion |
| `fear` | Expressions of anxiety or dread |
| `happiness` | Expressions of joy or satisfaction |
| `sadness` | Expressions of sorrow or grief |
| `surprise` | Expressions of shock or astonishment |
| `neutral` | No strong emotional content |

---

## Model Information

| Property | Value |
|---|---|
| Model family | DeBERTa-v3-base |
| Training data | GoEmotions (58k rows, 7 emotion labels) |
| Task | Single-label sentence-level emotion classification |
| Best val F1 | 0.517 (sweep: lr=5e-5, bs=32, epochs=2) |
| Evaluation metrics | Macro F1, Accuracy, Precision/Recall per class, Confusion Matrix |
| Experiment tracking | MLflow (local) + Azure ML workspace |
| Blue deployment | Model version 1 (60% traffic) |
| Green deployment | Model version 2 (40% traffic) |

---

## Contributing

1. Create a branch: `feature/your-feature`, `fix/your-fix`, or `chore/your-task`
2. Make changes and ensure tests pass: `pytest`
3. Ensure linting passes: `black src/ tests/ && flake8 src/ tests/`
4. Open a pull request to `develop`
5. After review and merge, open a PR from `develop` to `main` to trigger deployment
