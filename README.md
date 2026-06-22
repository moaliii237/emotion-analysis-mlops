## Team and my contribution

This was a 7-person Year 2 group project at Breda University. The lead developer was Zofia Pilitowska, with Ceren Kurt, Asma Moghimi, Arayah Coppens and others on the team. My part was the continuous-retraining feedback loop (the feedback module and the auto-retrain logic) and the command-line tool. I did not write the FastAPI app, the Azure deployment, the frontend or the test suite. I publish the full project here for my portfolio, with credit to the whole team.

---

# TV Emotion Analysis MLOps Application

This is a production-style NLP system that reads a TV transcript (or an audio/video file) and labels every sentence with one of 7 emotions. The full pipeline is Whisper for speech-to-text, MarianMT for English-to-Dutch translation, and a fine-tuned DeBERTa model for the emotion label. It is served behind a FastAPI app with a static frontend, runs on Azure ML and an on-premise server, and retrains itself from user feedback.

The README below describes the whole system so the project makes sense, but the parts I personally built and own are the ones I go into detail on: the feedback flywheel that closes the retraining loop, the Typer command-line tool, and the `EmotionClassifier` class the API and CLI both use. Everything else was built by my teammates and I credit them for it.

## What I built

Three things, all in `src/emotion_mlops/`:

1. **The feedback flywheel** (`feedback/`). This is the loop that turns "user says the model was wrong" into "a better model on the live endpoint", with no manual steps and without ever pushing a worse model live. Files: `auto_retrain.py`, `retrain_trigger.py`, `merge_inputs.py`, and the storage and bookkeeping in `db.py`.
2. **The command-line tool** (`cli/commands.py`). A Typer app called `emotionscope` with `predict`, `predict-full`, `health` and `pipeline` commands. This is the way to use the model from a terminal with no browser.
3. **The `EmotionClassifier` class** (`models/classifier.py`). The wrapper around the fine-tuned DeBERTa model that loads it once, keeps it in memory, and returns either the top emotion plus confidence or the full 7-way probability distribution. The API and the CLI both call into this.

The rest of this document is the full system reference. My deep-dives are in the [Feedback and Continuous Retraining](#feedback-and-continuous-retraining), [CLI](#cli) and [Model Information](#model-information) sections.

## Why a feedback flywheel was the hard part

A model trained once on GoEmotions and never touched again drifts away from real TV transcripts. The obvious fix is "let users correct wrong predictions and retrain on the corrections", but doing that safely is the actual problem. If you retrain blindly, a handful of noisy or wrong corrections can push a worse model live and you would not even notice. So the design goal I set was: collect corrections, retrain only when there are enough of them, and promote the new model only if it is measurably better than the one already serving traffic. The rest of my work follows from that.

---

- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Feedback and Continuous Retraining](#feedback-and-continuous-retraining)
- [CLI](#cli)
- [API](#api)
- [Database](#database)
- [UI](#ui)
- [Docker](#docker)
- [Airflow](#airflow)
- [On-Premise Deployment](#on-premise-deployment)
- [Local Training](#local-training)
- [Azure ML Pipeline](#azure-ml-pipeline)
- [Cloud Deployment](#cloud-deployment)
- [CI/CD](#cicd)
- [Emotion Labels](#emotion-labels)
- [Model Information](#model-information)
- [Contributing](#contributing)

---

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
    feedback_exports/          # corrections + merge previews I write here
  dist/
    emotion_mlops-0.1.0-py3-none-any.whl
  models/
    best_model_v3/             # fine-tuned DeBERTa weights (NOT in git, see below)
  monitoring/
    grafana/
    prometheus/
  src/
    emotion_mlops/
      api/                     # FastAPI app (teammates)
      azure/                   # Azure ML training, sweep, deploy (teammates)
      cli/
        commands.py            # MINE: Typer CLI (predict / predict-full / health / pipeline)
        infer.py               # older argparse pipeline driver, CLI delegates to it
        main.py                # console-script entry point
      feedback/                # MINE: the whole flywheel
        auto_retrain.py        # threshold check -> retrain -> F1 gate -> promote
        retrain_trigger.py     # export corrections, upload asset, submit training job
        merge_inputs.py        # merge base data + corrections in-cloud
        db.py                  # feedback storage + retrain watermark/bookkeeping
      models/
        classifier.py          # MINE: EmotionClassifier + predict_emotions batch fn
        transcriber.py         # Whisper wrapper (teammates)
        translator.py          # MarianMT wrapper (teammates)
        model_training.py      # training entry point (teammates)
      static/                  # frontend (teammates)
  tests/                       # pytest suite (teammates)
  Dockerfile
  docker-compose.yml
  pyproject.toml
  requirements.txt
  README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or 3.11
- Docker Desktop (for the database and the full stack)
- Azure CLI (`az login`) only if you want to touch the cloud parts

### Install

```bash
git clone https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-nlp-group-1.git
cd 2025-26d-fai2-adsai-group-nlp-group-1

python -m venv .venv311
# Windows
.\.venv311\Scripts\Activate.ps1
# macOS/Linux
source .venv311/bin/activate

# 1. PyTorch CPU build first
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    --only-binary=:all:

# 2. Project dependencies
pip install -r requirements.txt

# 3. Install the package (gives you the emotionscope CLI)
pip install -e .
```

### What is not in the repo

A few things are deliberately not committed and you need to supply them:

- **The model weights.** `models/best_model_v3/` is the fine-tuned DeBERTa folder. It is too large for git, so it is gitignored. Without it the CLI and the local model endpoints will not load. You can retrain it yourself (see [Local Training](#local-training)) or point at the Azure ML endpoint instead.
- **The raw training data.** The GoEmotions CSV is not committed. The flywheel works around this by leaving the base data in Azure and only uploading corrections (more on that below).
- **Secrets.** `ML_ENDPOINT_URL`, `ML_ENDPOINT_KEY`, `DATABASE_URL` go in a `.env` file or your shell, never in git.

### Environment variables

```bash
export ML_ENDPOINT_URL="https://emotion-classifier-endpoint.westeurope.inference.ml.azure.com/score"
export ML_ENDPOINT_KEY="your-key-here"
export DATABASE_URL="postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope"
```

---

## Configuration

Pipeline settings live in `configs/config.yaml`:

```yaml
device: cpu               # cpu or cuda
whisper_model: base       # tiny | base | small | medium
mt_en_nl_model: Helsinki-NLP/opus-mt-en-nl
best_model_dir: models/best_model_v3
batch_size: 16
max_length: 192
```

---

## Feedback and Continuous Retraining

This is the part I am most proud of, so it gets the most detail. It lives in `src/emotion_mlops/feedback/`.

### The idea

Users correct wrong predictions in the UI (thumbs-down plus the right label). Those corrections pile up in a database. When enough of them exist, the system retrains on the base data plus the corrections, checks whether the new model is actually better, and only then swaps it onto the live endpoint. If the new model is not better, the old one stays. No human in the loop, but also no silent regression.

### How a feedback row becomes a better model

```
user clicks thumbs-down and picks the correct label
        |
POST /feedback  ->  feedback table (db.record_feedback)
        |
threshold check (auto_retrain.should_retrain): >= 10 NEW corrections
        |  yes, and no retrain already running
export corrections CSV (db.export_corrections_for_retraining)
        |
upload corrections as their own Azure ML data asset version (emotion-feedback)
        |
submit Azure ML job: merge base + corrections in-cloud, preprocess, train
        |
read the new model's macro-F1 tag
        |
F1 GATE: promote ONLY if new f1 >= max(live model f1, floor 0.50)
        |  pass
deploy new version to a fresh endpoint slot, shift 100% traffic to it
        |
write the result to retrain_log, advance the watermark
```

### The design decisions that mattered

**A watermark so corrections are never reused.** Each finished retrain records the highest feedback `id` it consumed, in the `retrain_log` table. The next run counts only corrections with an `id` above that watermark (`db.count_new_corrections`, `db.get_watermark`). Without this, every run would retrain on the same corrections again and the threshold check would be meaningless.

**Only real corrections count.** A "usable correction" is feedback where the user disagreed and gave a replacement label (`is_correct = 0 AND user_label IS NOT NULL`). A thumbs-up is useful for the agreement-rate stat but adds no new training signal, so it does not count toward the threshold. Default threshold is 10 (`DEFAULT_THRESHOLD`).

**An F1 gate so a worse model never goes live.** This is the safety mechanism. After training, `auto_retrain.py` reads the macro-F1 tag of the newly registered model and the macro-F1 of the model currently serving the most endpoint traffic, then promotes only if `new_f1 >= max(live_f1, 0.50)`. The `0.50` floor (`MIN_PROMOTE_F1`) means even if there is no live model to compare against, nothing below 0.50 ever gets promoted. If the new model has no F1 tag, it is registered but left for manual review instead of being auto-promoted.

**A merge that keeps the human's label.** `merge_inputs.py` concatenates the base data and the corrections, putting corrections last, then de-duplicates on the sentence keeping the last occurrence. So when the same sentence exists in both, the human correction wins over the original GoEmotions label. This runs inside the Azure job, so the base data stays in the cloud and never has to be downloaded.

**Two modes, dry-run by default.** Both `auto_retrain.py` and `retrain_trigger.py` default to `--dry-run`. Dry-run reads the real feedback database so the correction count is real, prints exactly what each step would do, and makes zero Azure calls so it costs no money. `--live` actually uploads, trains and promotes. I made dry-run the default on purpose: during a mentor demo you can show the whole flow and the real numbers without accidentally launching a 15-minute GPU job.

**Auto-firing is opt-in.** The `/feedback` endpoint can trigger a live retrain on its own, but only if `AUTO_RETRAIN_ENABLED=1` is set in the environment (`maybe_trigger_after_feedback`). By default, feedback is just recorded and you trigger retraining yourself. Same reason: I did not want a demo to set money on fire.

**An in-flight lock.** `db.is_retrain_in_flight()` returns true while a retrain row is in `started` or `training` status, so two retrains can never overlap.

### Storage

`feedback/db.py` uses PostgreSQL when `DATABASE_URL` is set and falls back to SQLite for local dev, so the same code works on the server and on a laptop with nothing installed. It manages two tables, `feedback` (the corrections) and `retrain_log` (the bookkeeping: which corrections each run used, the job name, the new and live F1, whether it promoted).

### Try it without spending anything

```bash
# Show the plan and the real correction count, no Azure calls
python -m emotion_mlops.feedback.auto_retrain --dry-run

# Lower the threshold to test the trigger logic
python -m emotion_mlops.feedback.auto_retrain --dry-run --threshold 5

# The simpler trigger (export + merge preview only in dry-run)
python -m emotion_mlops.feedback.retrain_trigger --dry-run
```

Live runs (`--live`) need the team's Azure modules present on the branch and will use real compute.

### Submit feedback via the API

```python
import requests

# User corrects a wrong prediction (this is the kind that counts toward retraining)
requests.post('http://localhost:8000/feedback', json={
    'sentence': 'Oh wow, I did not see that coming.',
    'predicted_label': 'happiness',
    'predicted_conf': 0.73,
    'is_correct': False,
    'user_label': 'surprise',        # the human correction
    'model_version': 'best_model_v3',
    'source': 'text',
})

# Check flywheel progress
r = requests.get('http://localhost:8000/feedback/auto-check')
# {"new_corrections": 7, "threshold": 10, "ready": false, ...}
```

---

## CLI

The command-line tool is `src/emotion_mlops/cli/commands.py`, a Typer app I built and exposed as the `emotionscope` console script (see `pyproject.toml` `[project.scripts]`). After `pip install -e .` it is on your path.

I chose Typer because the type hints on each command double as input validation and `--help` is generated from the docstrings, so I did not have to hand-write argument parsing or help text. The classifier is loaded lazily and cached, so running `--help` does not pay the cost of loading DeBERTa.

### Commands

```bash
# Single most likely emotion, with coloured output
emotionscope predict "I am absolutely furious!"

# Full 7-way probability distribution as a terminal bar chart
emotionscope predict-full "I'm both excited and scared"

# Same but machine-readable, for piping into scripts
emotionscope predict-full "I am angry" --json

# Check the model loads and report its version
emotionscope health

# Full pipeline: transcribe + translate + classify -> CSV
emotionscope pipeline --audio meeting.mp3 --config configs/config.yaml --out results.csv
```

`predict-full --json` is built for scripting:

```bash
emotionscope predict-full "I am angry" --json | jq '.top_emotion'
# "anger"
```

The `pipeline` command does not duplicate the pipeline logic. It delegates to the older `infer.py` driver, so there is one place where transcription, translation and classification actually happen.

### Without installing the package

```powershell
$env:PYTHONPATH="src"
python -m emotion_mlops.cli.commands predict "I am happy"
```

---

## API

The FastAPI backend (built by my teammates) exposes the pipeline and my classifier over HTTP. The feedback endpoints below are the ones wired into my flywheel.

### Start it locally

```powershell
$env:PYTHONPATH="src"
python -m uvicorn emotion_mlops.api.app:app --port 8000 --reload
```

Open http://localhost:8000/docs for the Swagger UI.

### Endpoints that touch my code

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict/text` | Top emotion for one sentence, via my `EmotionClassifier` |
| `POST` | `/predict/text/full` | All 7 label probabilities, via `predict_all` |
| `POST` | `/feedback` | Record a user correction |
| `GET` | `/feedback/stats` | Agreement rate and per-model breakdown |
| `GET` | `/feedback/export` | Export corrections for retraining |
| `GET` | `/feedback/auto-check` | Flywheel progress (new corrections vs threshold) |
| `POST` | `/feedback/retrain` | Trigger a retrain (supports `?dry_run=true`) |

The backend has many more endpoints (cloud prediction, timeline, scenes, summary, and so on) that my teammates built on top of the pipeline.

---

## Database

Predictions and feedback persist in PostgreSQL (SQLite fallback for local dev). The two tables my flywheel relies on:

### `feedback` table

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL | Primary key, also the watermark cursor |
| `timestamp` | TIMESTAMP | UTC time |
| `sentence` | TEXT | Input sentence |
| `predicted_label` | VARCHAR | What the model predicted |
| `predicted_conf` | FLOAT | Model confidence |
| `user_label` | VARCHAR | The human's corrected label, if any |
| `is_correct` | INTEGER | 1 = agreed, 0 = corrected |
| `model_version` | VARCHAR | Which model produced the prediction |
| `source` | VARCHAR | `audio`, `transcript`, or `text` |

### `retrain_log` table (my bookkeeping)

Records each retrain: the `corrections_from_id`/`corrections_to_id` range it consumed, the data version, the Azure job name, the new and live F1, whether it promoted, and a status (`started`, `training`, `registered`, `promoted`, `skipped`, `failed`). The watermark is `MAX(corrections_to_id)` over the completed rows.

### Local database

```bash
docker-compose up -d postgres
# DATABASE_URL=postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope
```

---

## UI

The frontend is a static web app served by the FastAPI backend (built by my teammates). The page that matters for my work is **Features -> Monitor**, which shows the feedback flywheel: how many new corrections exist, the threshold, and the retrain history coming from `retrain_log`.

---

## Docker

```bash
docker-compose up -d
```

| Service | Local URL | Description |
|---|---|---|
| `backend` | http://localhost:2002 | FastAPI API + frontend |
| `postgres` | localhost:5432 | Database |
| `prometheus` | http://localhost:2003 | Metrics |
| `grafana` | http://localhost:2004 (admin/admin) | Dashboards |
| `airflow` | http://localhost:2005 (admin/admin) | Orchestration |
| `mlflow` | http://localhost:2006 | Experiment tracking |

---

## Airflow

Apache Airflow orchestrates the data and retraining pipelines by triggering GitHub Actions workflows (so no Azure credentials live in Airflow). The DAG tied to my work is `emotion_feedback_retrain`, scheduled daily at 06:00, which runs the flywheel and triggers retraining when user corrections reach the threshold.

| DAG | Schedule | Description |
|---|---|---|
| `emotion_data_preprocessing` | manual | Preprocess GoEmotions and upload as a data asset |
| `emotion_model_training` | Monday 08:00 | Train, register best model, deploy as green |
| `emotion_full_retrain` | Sunday 00:00 | Full weekly preprocess + train + deploy |
| `emotion_feedback_retrain` | Daily 06:00 | My flywheel: retrain when corrections >= 10 |

---

## On-Premise Deployment

The app runs on the BUas Lambda server via Portainer as the `emotionscope` stack (backend, postgres, prometheus, grafana, airflow). Portainer polls the deploy branch and redeploys on a new image, and a GitHub Actions webhook can trigger an immediate redeploy. This was set up by my teammates.

---

## Local Training

If you want to produce your own `best_model_v3` instead of using the Azure endpoint:

```powershell
# 1. Preprocess
python -m emotion_mlops.data.data_preprocessing `
    --input data/raw/goemotions_collapsed_majority.csv `
    --output data/processed/

# 2. Train
python -m emotion_mlops.models.model_training `
    --data data/processed/ `
    --output models/trained/ `
    --epochs 3 --batch-size 16 --learning-rate 2e-5

# 3. Evaluate
python -m emotion_mlops.models.model_evaluation `
    --model models/trained/ --data data/processed/ --output data/eval/
```

Training, preprocessing and evaluation were written by my teammates.

---

## Azure ML Pipeline

My teammates built the Azure ML side: registering the training environment, uploading versioned data assets, the training pipeline, a hyperparameter sweep, model registration, and a weekly schedule. My flywheel reuses their `pipeline.py`, `data_assets.py` and `deploy_endpoint.py` rather than reimplementing them, so the retrain path and the normal training path share the same Azure code.

Sweep search space (theirs): `learning_rate` in [5e-6, 1e-5, 2e-5, 3e-5, 5e-5], `batch_size` in [16, 32, 64], `epochs` in [2, 3, 4].

---

## Cloud Deployment

The model is served from an Azure ML Online Endpoint with a blue-green setup, and the API runs on Azure Container Apps. My flywheel's promote step deploys a new model version to a fresh endpoint slot and shifts 100% of traffic to it, reusing the team's `deploy_endpoint.py`.

| Resource | Value |
|---|---|
| Endpoint name | `emotion-classifier-endpoint` |
| Scoring URI | `https://emotion-classifier-endpoint.westeurope.inference.ml.azure.com/score` |
| Strategy | blue-green |

---

## CI/CD

GitHub Actions handles linting, tests, image builds and deploys (set up by my teammates). The workflow relevant to me is `retrain.yml`, which runs daily and on manual dispatch, checks the feedback threshold, and triggers retraining when there are at least 10 new corrections.

---

## Emotion Labels

The model predicts one of 7 labels per sentence.

| Label | Meaning |
|---|---|
| `anger` | Frustration or rage |
| `disgust` | Revulsion |
| `fear` | Anxiety or dread |
| `happiness` | Joy or satisfaction |
| `sadness` | Sorrow or grief |
| `surprise` | Shock or astonishment |
| `neutral` | No strong emotion |

---

## Model Information

The classifier wrapper (`models/classifier.py`) is mine; the training that produced the weights was done by my teammates.

| Property | Value |
|---|---|
| Model family | DeBERTa-v3-base |
| Training data | GoEmotions, collapsed to 7 emotion labels |
| Task | Single-label sentence-level emotion classification |
| Best validation macro-F1 | about 0.517 (sweep: lr=5e-5, batch=32, epochs=2) |
| Max token length | 192 |
| Experiment tracking | MLflow + Azure ML |

### Honest note on the results

A macro-F1 of about 0.517 across 7 emotions is modest. GoEmotions is hard: classes are imbalanced, the neutral class dominates, and several emotions (fear, disgust) have far fewer examples than happiness or neutral. The model is good enough to demonstrate the MLOps lifecycle, which was the point of Block D, but it is not a state-of-the-art emotion classifier and I would not present it as one. The whole reason the feedback flywheel exists is that the first model is never the last word: real corrections from real transcripts are exactly the signal that pushes that number up over time, and the F1 gate makes sure each retrain only keeps the model if it actually does.

How `EmotionClassifier` works: it loads the tokenizer and model once at startup, moves them to GPU if available else CPU, runs a forward pass and a softmax, and returns either the top label with its confidence (`predict`) or the full probability dict over all 7 labels (`predict_all`). The label names come straight from the model's `id2label` config, lowercased, so the class does not hard-code them.

---

## Contributing

1. Branch: `feature/...`, `fix/...`, or `chore/...`
2. Make changes and run `pytest`
3. Run linting before pushing
4. Open a PR to `develop`, then `develop` to `main` to deploy
