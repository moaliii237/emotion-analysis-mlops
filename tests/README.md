## Running the tests

### Unit tests (mocked, fast, runs on every push in CI)

```bash
# From repository root
pytest tests/unit/ -v
```

With coverage:
```bash
pytest tests/unit/ -v --cov=src/emotion_mlops --cov-report=term-missing
```

### Integration tests (HTTP, requires live server)

In one terminal, start the API:
```powershell
$env:PYTHONPATH="src"
uvicorn emotion_mlops.api.app:app --reload --port 8000
```

In another terminal, run the suite:
```bash
pytest tests/integration/ -v
```

## Mocking strategy (unit tests)

The unit suite never touches:
- Real DeBERTa model (~500 MB)
- Real Whisper / MarianMT / pyannote.audio
- The Azure ML Online Endpoint
- The Groq LLM API
- The production SQLite/PostgreSQL feedback DB

All of these are patched in `tests/unit/conftest.py`:

| Component | Mocked as |
|-----------|-----------|
| `emotion_mlops.api.app.classifier` | `FakeClassifier` (keyword-based) |
| Whisper / MarianMT modules | `object()` placeholders |
| `emotion_mlops.analysis.summary.generate_summary` | Deterministic fake |
| `emotion_mlops.analysis.autotitle.generate_title` | Returns fallback label |
| `emotion_mlops.api.app._get_endpoint_url_and_key` | Returns fake creds |
| `requests.post` | `_fake_azure_post`, mirrors FakeClassifier logic |

The Azure ML mock is the most important addition, it intercepts the HTTP call that `/predict/text` makes to the Azure ML scoring endpoint and returns a response derived from the same keyword logic the FakeClassifier uses.

## Coverage configuration

Target: **80% on the testable surface**.

Configured in `.coveragerc`:

- **`omit`**, files with no realistic unit-test surface (Azure ML training scripts, audio pipelines, Groq-backed modules, real DB drivers, CLI tool)
- **`exclude_lines`**, error handlers (`raise HTTPException`, `except Exception`), defensive code paths, AI condition lines
- **`# pragma: no cover`**, function-level markers on:
  - Pipeline / YouTube endpoints (require audio files)
  - Lifespan startup (loads real Whisper / MarianMT)
  - Cloud-fallback paths inside endpoints
  - `_ai_classification`, `_ai_assessment`, `_generate_tagline` (Groq calls)

## Current status

- **67 unit tests**, all passing in CI (~6 s with coverage)
- **Coverage: ~81%** on the testable surface
- **Integration suite**, passes locally against a live server

## CI behavior

GitHub Actions runs the unit suite on every push to a non-protected branch and on every PR to `main` or `develop`. The workflow tolerates exit code 134 (a known race between torch's atexit handler and pytest-cov on Linux) when all tests have already passed.

See `.github/workflows/test.yml` for the exact command.