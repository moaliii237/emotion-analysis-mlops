# Emotion Classification API

A FastAPI application that serves the fine-tuned DeBERTa emotion classifier over HTTP. It exposes a small set of endpoints for health checks and single-sentence emotion prediction.

## Structure

```
api/
+-- __init__.py
+-- schemas.py   # Pydantic request and response models
+-- app.py       # FastAPI routes and lifespan manager
```

The classifier itself lives in `src/emotion_mlops/models/classifier.py` as the `EmotionClassifier` class. The API only wires HTTP routes to it.

## Endpoints

### GET /

Returns basic API information: name, version, and a list of available endpoints. Useful as a simple liveness indicator.

### GET /health

Reports whether the model is currently loaded in memory. Intended for deployment probes and monitoring dashboards.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "best_model_v3"
}
```

### POST /predict/text

Predicts the emotion for a single input sentence and returns it together with a confidence score and the model version.

Request body:

```json
{"text": "Listen to me right now!"}
```

Response:

```json
{
  "text": "Listen to me right now!",
  "emotion": "anger",
  "confidence": 0.8734,
  "model_version": "best_model_v3"
}
```

Possible status codes:

- `200`, prediction successful
- `422`, invalid request body (missing, empty, or wrong-typed `text` field)
- `503`, the model is not currently loaded
- `500`, the model raised an unexpected exception during prediction

## Design notes

- **Model loaded once at startup.** The FastAPI `lifespan` context manager loads the model when the application starts and keeps it in memory for every subsequent request. This avoids the model-reload cost per request.
- **Dependency injection.** Endpoints receive the classifier through `Depends(get_classifier)`. This makes it straightforward to substitute a fake classifier in unit tests without modifying the route code.
- **Pydantic validation.** All request and response shapes are defined in `schemas.py`. Invalid inputs are rejected automatically with a 422 response before reaching the handler.
- **Auto-generated documentation.** Interactive Swagger UI is available at `/docs` and ReDoc at `/redoc` once the server is running.

## Running locally

From the project root:

**Windows (PowerShell)**

```powershell
$env:PYTHONPATH="src"
uvicorn emotion_mlops.api.app:app --reload --port 8000
```

**macOS / Linux**

```bash
PYTHONPATH=src uvicorn emotion_mlops.api.app:app --reload --port 8000
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) in a browser.

## Model files

The trained model must be available locally at `models/best_model_v3/` (relative to the project root). This folder is gitignored because of its size, so every contributor needs to place the model in their own local project directory before running the API.

The folder must contain the standard Hugging Face artifacts: `config.json`, `tokenizer.json`, `model.safetensors` (or `pytorch_model.bin`), and the accompanying tokenizer files.

## Testing

Fast tests (no trained model required, complete in a few seconds):

```bash
pytest tests/test_schemas.py tests/test_api.py -v
```

Integration tests (require `models/best_model_v3/` to be present locally):

```bash
pytest tests/test_classifier_real.py -v
```

Run the full test suite:

```bash
pytest -v
```

In CI, only the fast tests are executed. The integration tests are automatically skipped when the model folder is missing, using `pytest.mark.skipif` inside the test file.