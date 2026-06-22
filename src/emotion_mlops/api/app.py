"""
FastAPI backend for the EmotionScope emotion-classification pipeline.

This module exposes the inference pipeline as a REST API, handling
model loading, health checks, single-sentence prediction, and the
full audio/transcript pipeline. It also hosts the Sprint 4 analytic
endpoints (cast detection, scenes, trailer, score, shifts, persuasion,
comparison and AI summary) and the user-feedback flywheel.

Endpoints (high level):
    GET  /, Static UI (EmotionScope frontend)
    GET  /api, API metadata and available endpoints
    GET  /health, Model load status and version
    POST /predict/text, Single-sentence emotion prediction
    POST /predict/text/full, Single sentence + full probability distribution
    POST /predict/pipeline, Full pipeline: audio/CSV -> emotions
    POST /predict/youtube, Same pipeline, but from a YouTube URL
    POST /predict/cast, Named-character emotion profiling
    POST /predict/timeline, Time-series of emotion intensity
    POST /predict/scenes, Auto-detected emotional scenes
    POST /predict/trailer, Best clips for a short trailer
    POST /predict/score, Producer-style scorecard
    POST /predict/shifts, Emotional turning points
    POST /predict/persuasion, Persuasion analysis
    POST /predict/summary, AI 2-3 sentence narrative verdict
    POST /compare, Compare two pre-analyzed pipeline results
    POST /feedback, Record user feedback on a prediction
    GET  /feedback/stats, Aggregated feedback statistics
    GET  /feedback/export, Export user corrections for retraining
    POST /feedback/retrain, Trigger the retraining flywheel
    POST /summarize, Generic transcript / text summarisation
    POST /ask, Free-form Q&A about a transcript (Groq LLM)
"""

import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from emotion_mlops.api.database import (
    PipelineMetrics,
)
from emotion_mlops.api.database import init_db as init_prediction_db
from emotion_mlops.api.database import (
    log_pipeline_metrics,
    log_prediction,
)
from emotion_mlops.api.schemas import (
    AdminLoginRequest,
    AdminLoginResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    FullPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TextRequest,
)
from emotion_mlops.feedback.db import (
    export_corrections_for_retraining,
    get_stats,
)
from emotion_mlops.feedback.db import init_db as init_feedback_db
from emotion_mlops.feedback.db import (
    record_feedback,
)
from emotion_mlops.models.classifier import EmotionClassifier

# ---------------------------------------------------------------------------
# Project paths.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = _REPO_ROOT / "models" / "best_model_v3"
_CONFIG = _REPO_ROOT / "configs" / "config.yaml"
_LOG_FILE = _REPO_ROOT / "logs" / "api.log"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_LOG_FILE.parent.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logger initialised early so the helper functions below can use it.
# Writes to both stdout (for `uvicorn --reload`) and to logs/api.log.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Make `src/` and `models/` importable as top-level packages.
for _p in [str(_REPO_ROOT / "src"), str(_REPO_ROOT / "models")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Singleton classifier instance, loaded once in `lifespan`, shared by all
# request handlers via the `get_classifier` dependency.
# ---------------------------------------------------------------------------
classifier = EmotionClassifier(
    model_path=str(_MODEL_DIR),
    model_version="best_model_v3",
)

# Module-level holders for Whisper and MarianMT. They start empty and are
# populated during the lifespan startup hook. If a model fails to load the
# corresponding dict stays empty and the relevant endpoints return 503.
whisper_state: dict = {}
mt_state: dict = {}


def build_cast_analysis(df) -> list:  # pragma: no cover
    """
    Run cast analysis on a pipeline DataFrame.

    Builds the full transcript from the Sentence column and feeds it to
    the text-based cast analyzer. Failures are logged but don't break
    the pipeline response, cast is a nice-to-have, not critical.
    """
    try:
        from emotion_mlops.analysis.cast_analyzer import analyze_cast

        full_text = " ".join(df["Sentence"].astype(str).tolist())
        if not full_text.strip():
            return []
        return analyze_cast(full_text, classifier, top_n=10)
    except Exception as e:
        logger.warning("[Cast] Analysis failed: %s", e)
        return []


def build_timeline_from_df(df) -> dict:  # pragma: no cover
    """
    Build an emotion timeline from a pipeline DataFrame.

    Converts the DataFrame to segment dicts and calls the timeline
    analyzer. Returns an empty dict on failure so the pipeline JSON
    response stays well-formed.
    """
    try:
        from emotion_mlops.analysis.timeline import build_timeline

        segments = df[["Start", "Sentence", "Emotion", "Confidence"]].to_dict(
            orient="records"
        )
        return build_timeline(segments)
    except Exception as e:
        logger.warning("[Timeline] Failed: %s", e)
        return {}


def build_top_moments(df, n=3):  # pragma: no cover
    """Sort segments by confidence per emotion and take the top N for each."""
    top_moments = {}
    for emotion in df["Emotion"].unique():
        emotion_segments = (
            df[df["Emotion"] == emotion]
            .sort_values("Confidence", ascending=False)
            .head(n)
        )
        top_moments[emotion] = emotion_segments[
            ["Start", "End", "Sentence", "Confidence"]
        ].to_dict(orient="records")
    return top_moments


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    """
    Manage application lifespan, load all models on startup, clean up on shutdown.

    Loads DeBERTa classifier, Whisper ASR, and MarianMT translation model
    into memory before the API starts accepting requests. Configuration is
    read from configs/config.yaml if present, otherwise sensible defaults
    are used. Whisper and MarianMT are non-fatal: if either fails to load
    we keep the API up and disable only the affected endpoints.
    """
    import yaml

    logger.info("[API] Starting up, loading all models...")

    # Initialise both SQLite databases (feedback flywheel + raw predictions).
    init_feedback_db()
    init_prediction_db()
    logger.info("[API] Feedback DB and prediction DB initialised.")

    # Optional config file; missing config = use defaults.
    cfg = {}
    if _CONFIG.exists():
        with open(_CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logger.warning("[API] config.yaml not found, using defaults")

    device = cfg.get("device", "cpu")
    whisper_name = cfg.get("whisper_model", "base")
    mt_model_name = cfg.get("mt_en_nl_model", "Helsinki-NLP/opus-mt-en-nl")

    # 1) Always-required classifier (DeBERTa). Failure here is fatal.
    logger.info("[API] Loading DeBERTa classifier...")
    # Load local model only if available; fall back to Azure ML endpoint
    if _MODEL_DIR.exists():
        classifier.load()
    else:
        ml_url = os.environ.get("ML_ENDPOINT_URL")
        if ml_url:
            logger.warning(
                "[API] Local model not found at %s, using Azure ML endpoint: %s",
                _MODEL_DIR,
                ml_url,
            )
        else:
            logger.error(
                "[API] Local model not found and ML_ENDPOINT_URL is not set. "
                "Prediction endpoints will return 503."
            )
    logger.info("[API] DeBERTa ready.")

    # 2) Whisper (audio -> text). Optional: pipeline endpoint returns 503 if missing.
    logger.info("[API] Loading Whisper (%s) on %s...", whisper_name, device)
    try:
        import whisper as _whisper

        whisper_state["model"] = _whisper.load_model(whisper_name, device=device)
        whisper_state["device"] = device
        whisper_state["model_name"] = whisper_name
        logger.info("[API] Whisper ready.")
    except Exception as e:
        logger.warning(
            "[API] Whisper not available: %s, audio pipeline disabled",
            e,
        )
        whisper_state["model"] = None

    # 3) MarianMT (EN -> NL). Optional: translation is added as a bonus column.
    logger.info("[API] Loading MarianMT (%s) on %s...", mt_model_name, device)
    try:
        from pipeline import load_mt

        mt_state["tokenizer"], mt_state["model"] = load_mt(mt_model_name, device)
        mt_state["device"] = device
        logger.info("[API] MarianMT ready.")
    except Exception as e:
        logger.warning(
            "[API] MarianMT not available: %s, translation disabled", e
        )
        mt_state["model"] = None
        mt_state["tokenizer"] = None

    logger.info("[API] All models loaded. Docs: http://localhost:8000/docs")
    yield
    logger.info("[API] Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app factory.
# `lifespan=lifespan` wires the async-context-manager above so that all
# heavy model loading happens before the server starts answering requests.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Emotion Classification API",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

Instrumentator().instrument(app).expose(app)


def get_classifier() -> EmotionClassifier:
    """Dependency provider for the EmotionClassifier instance."""
    return classifier


# ---------------------------------------------------------------------------
# Metadata endpoint. Lets the frontend (and reviewers) discover everything
# the API can do without reading the source.
# ---------------------------------------------------------------------------
@app.get("/api")
def root():
    """Return API metadata and the list of available endpoints."""
    return {
        "name": "Emotion Classification API",
        "version": "0.2.0",
        "docs": "/docs",
        "endpoints": [
            "/",
            "/api",
            "/health",
            "/predict/text",
            "/predict/text/full",
            "/predict/pipeline",
            "/predict/youtube",
            "/retrain",
            "/feedback",
            "/feedback/stats",
            "/feedback/export",
            "/feedback/retrain",
            "/feedback/auto-check",
            "/summarize",
            "/ask",
            "/admin/login",
            "/admin/verify",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health(clf: EmotionClassifier = Depends(get_classifier)):
    """Check the health status of the API and the underlying classifier."""
    return HealthResponse(
        status="ok" if clf.is_loaded else "model_not_loaded",
        model_loaded=clf.is_loaded,
        model_version=clf.model_version if clf.is_loaded else None,
    )


# ---------------------------------------------------------------------------
# Single-sentence prediction. Used by the UI's "Quick test" buttons and any
# external caller that wants a one-shot answer without the full pipeline.
# ---------------------------------------------------------------------------
@app.post("/predict/text", response_model=PredictionResponse)
def predict_text(
    request: TextRequest,
):
    """Predict the emotion of a single input sentence."""
    import requests as _requests

    try:
        url, key = _get_endpoint_url_and_key()
        r = _requests.post(
            url,
            json={"sentence": request.text},
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        )
        r.raise_for_status()
        result = r.json()
        emotion = result[0]["emotion"]
        confidence = result[0]["confidence"]
    except Exception as e:
        logger.error("[API] Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    log_prediction(request.text, emotion, confidence, "azure-ml-endpoint")

    return PredictionResponse(
        text=request.text,
        emotion=emotion,
        confidence=round(confidence, 4),
        model_version="azure-ml-endpoint",
    )


@app.post("/predict/text/full", response_model=FullPredictionResponse)
def predict_text_full(
    request: TextRequest,
    clf: EmotionClassifier = Depends(get_classifier),
):
    """Predict emotion with probabilities for ALL labels (used by the radar chart)."""
    if not clf.is_loaded:  # pragma: no cover
        # fallback to cloud endpoint
        import requests as _requests

        try:
            url, key = _get_endpoint_url_and_key()
            r = _requests.post(
                url,
                json={"sentence": request.text},
                headers={"Authorization": f"Bearer {key}"},
                timeout=60,
            )
            r.raise_for_status()
            result = r.json()
            all_emotions = result[0].get("all_emotions", {})
            top_emotion = result[0]["emotion"]
            top_confidence = result[0]["confidence"]
            log_prediction(
                request.text, top_emotion, top_confidence, "azure-ml-endpoint"
            )
            return FullPredictionResponse(
                text=request.text,
                top_emotion=top_emotion,
                top_confidence=round(top_confidence, 4),
                all_emotions={k: round(v, 4) for k, v in all_emotions.items()},
                model_version="azure-ml-endpoint",
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Cloud endpoint error: {e}")

    try:
        all_emotions = clf.predict_all(request.text)
    except Exception as e:
        logger.error("[API] Full prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # The top emotion is the argmax of the full distribution.
    top_emotion = max(all_emotions, key=all_emotions.get)
    top_confidence = all_emotions[top_emotion]
    rounded = {k: round(v, 4) for k, v in all_emotions.items()}

    return FullPredictionResponse(
        text=request.text,
        top_emotion=top_emotion,
        top_confidence=round(top_confidence, 4),
        all_emotions=rounded,
        model_version=clf.model_version,
    )


# ---------------------------------------------------------------------------
# Full audio / CSV pipeline.
#
# Accepts EITHER:
#   - an audio/video file (mp3/wav/mp4/...) -> Whisper transcribes it, then
#     translates + classifies the segments;
#   - a CSV file with a Sentence/Text/Utterance column -> skips Whisper,
#     just translates + classifies.
#
# Optionally runs speaker diarization for richer per-character analysis.
# ---------------------------------------------------------------------------
@app.post("/predict/pipeline")
async def predict_pipeline(  # pragma: no cover
    file: UploadFile = File(...),
    is_transcript: bool = Form(False),
    whisper_model: str = Form("base"),
    batch_size: int = Form(16),
    max_length: int = Form(192),
    diarize: bool = Form(False),
):
    """Run the full emotion-classification pipeline on an audio file or transcript CSV."""
    import time

    import pandas as pd

    from pipeline import hhmmss_ms, translate_batch

    if not whisper_state.get("model"):
        raise HTTPException(status_code=503, detail="Whisper not loaded yet.")
    # All temporary files (uploaded audio, intermediate CSVs) live in a
    # tempdir that is cleaned up automatically when the request finishes.
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, file.filename or "input")
        content = await file.read()
        with open(audio_path, "wb") as f:
            f.write(content)

        logger.info("[Pipeline] Processing file: %s", file.filename)
        t0 = time.time()

        # ----- 1. Transcript acquisition (Whisper OR CSV passthrough) -----
        t1 = time.time()
        if is_transcript:
            # CSV mode: read the file and normalise the column names so the
            # rest of the pipeline doesn't have to care about which casing
            # the user uploaded ("Sentence" vs "text" vs "utterance").
            df_in = pd.read_csv(audio_path)
            col_map = {c.lower(): c for c in df_in.columns}
            s_col = (
                col_map.get("sentence")
                or col_map.get("text")
                or col_map.get("utterance")
            )
            st_col = col_map.get("start") or col_map.get("start_time")
            et_col = col_map.get("end") or col_map.get("end_time")
            if not s_col:
                raise HTTPException(
                    status_code=422,
                    detail=f"CSV must have a Sentence/Text/Utterance column. Found: {list(df_in.columns)}",
                )
            df = df_in.rename(columns={s_col: "Sentence"})
            if st_col and et_col:
                df = df.rename(columns={st_col: "Start", et_col: "End"})
            else:
                # No timestamps in the CSV, fill with zeros so downstream
                # code that expects Start/End columns still works.
                df["Start"] = "00:00:00,000"
                df["End"] = "00:00:00,000"
        else:
            # Audio/video mode: Whisper turns the waveform into timestamped
            # segments. `verbose=False` keeps stdout clean for the API logs.
            result = whisper_state["model"].transcribe(
                audio_path, verbose=False, language="en"
            )
            segs = result["segments"]
            df = pd.DataFrame(
                {
                    "Start": [hhmmss_ms(float(s["start"])) for s in segs],
                    "End": [hhmmss_ms(float(s["end"])) for s in segs],
                    "Sentence": [s["text"].strip() for s in segs],
                }
            )

            # Optional speaker diarization. We do this only on raw audio (no
            # speakers available from a generic CSV) and we tolerate failure
            # gracefully, the pipeline keeps running without speaker labels.
            if diarize:
                try:
                    from emotion_mlops.diarization.diarizer import (
                        assign_speakers_to_segments,
                    )
                    from emotion_mlops.diarization.diarizer import diarize as _diarize

                    logger.info("[Pipeline] Running speaker diarization...")
                    turns = _diarize(audio_path)
                    speakers = assign_speakers_to_segments(segs, turns)
                    df["Speaker"] = speakers
                    logger.info(
                        "[Pipeline] Diarization: %d turns, %d unique speakers",
                        len(turns),
                        len(set(speakers)),
                    )
                except Exception as e:
                    logger.warning("[Pipeline] Diarization failed: %s", e)

        t_whisper = round(time.time() - t1, 2)
        logger.info(
            "[Pipeline] Transcription: %d segments in %.2fs", len(df), t_whisper
        )

        # ----- 2. EN -> NL translation (MarianMT) -----
        t2 = time.time()
        if mt_state.get("model") and mt_state.get("tokenizer"):
            translations = translate_batch(
                df["Sentence"].tolist(),
                mt_state["tokenizer"],
                mt_state["model"],
                mt_state.get("device", "cpu"),
                bs=batch_size,
            )
        else:
            translations = [""] * len(df)
        df["Translation"] = translations
        t_translate = round(time.time() - t2, 2)
        logger.info("[Pipeline] Translation done in %.2fs", t_translate)

        # ----- 3. Emotion classification (DeBERTa) -----
        t3 = time.time()
        import requests as _requests

        url_endpoint, key = _get_endpoint_url_and_key()
        emotions = []
        for sentence in df["Sentence"].tolist():
            try:
                r = _requests.post(
                    url_endpoint,
                    json={"sentence": sentence},
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=30,
                )
                result = r.json()
                emotions.append((result[0]["emotion"], result[0]["confidence"]))
            except Exception as e:
                logger.warning("[Pipeline] Cloud prediction failed: %s", e)
                emotions.append(("neutral", 0.5))
        df["Emotion"] = [label for label, _ in emotions]
        df["Confidence"] = [conf for _, conf in emotions]
        t_classify = round(time.time() - t3, 2)
        logger.info("[Pipeline] Classification done in %.2fs", t_classify)

        t_total = round(time.time() - t0, 2)
        logger.info("[Pipeline] Total: %.2fs", t_total)

        # ----- 4. Per-character summaries (only if diarization gave us speakers) -----
        from emotion_mlops.analysis.per_character import (
            detect_speaker_column,
            per_character_summary,
            per_character_top_moments,
        )

        speaker_col = detect_speaker_column(df)
        per_char_summary = per_character_summary(df, speaker_col) if speaker_col else {}
        per_char_moments = (
            per_character_top_moments(df, speaker_col, n=2) if speaker_col else {}
        )

        # Build the final segment list. We always expose a "Speaker" column
        # if one was detected, even if it came in under a different name.
        segment_cols = [
            "Start",
            "End",
            "Sentence",
            "Translation",
            "Emotion",
            "Confidence",
        ]
        if speaker_col:
            segment_cols.insert(2, speaker_col)
        segments_df = df[segment_cols].copy()
        if speaker_col and speaker_col != "Speaker":
            segments_df = segments_df.rename(columns={speaker_col: "Speaker"})

        return JSONResponse(
            {
                "segments": segments_df.to_dict(orient="records"),
                "emotions": [label for label, _ in emotions],
                "t_whisper": t_whisper,
                "t_translate": t_translate,
                "t_classify": t_classify,
                "t_total": t_total,
                "top_moments": build_top_moments(df, n=3),
                "per_character": per_char_summary,
                "per_character_top_moments": per_char_moments,
                "has_speakers": bool(speaker_col),
                "cast": build_cast_analysis(df),
                "timeline": build_timeline_from_df(df),
            }
        )


# ---------------------------------------------------------------------------
# Same pipeline but the input is a YouTube URL, we download the audio with
# yt_dlp, then re-use the same Whisper/Translate/Classify chain as above.
# ---------------------------------------------------------------------------
@app.post("/predict/youtube")
async def predict_youtube(  # pragma: no cover
    url: str = Form(...),
    whisper_model: str = Form("base"),
    batch_size: int = Form(16),
    max_length: int = Form(192),
    diarize: bool = Form(False),
):
    """Download a YouTube video's audio and run the full pipeline."""
    import subprocess
    import time

    import pandas as pd

    from pipeline import hhmmss_ms, translate_batch

    if not whisper_state.get("model"):
        raise HTTPException(
            status_code=503, detail="Whisper or MarianMT not loaded yet."
        )

    if not url.strip():
        raise HTTPException(status_code=422, detail="YouTube URL is required.")

    with tempfile.TemporaryDirectory() as tmp:
        logger.info("[YouTube] Downloading: %s", url)
        t0 = time.time()

        # imageio-ffmpeg bundles a portable ffmpeg binary; we prefer it over
        # the system ffmpeg because it works the same on every dev machine.
        ffmpeg_location = None
        try:
            import imageio_ffmpeg

            ffmpeg_location = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            logger.warning("[YouTube] imageio-ffmpeg not found")

        # We always download to WAV so Whisper doesn't have to do any decoding.
        audio_path = os.path.join(tmp, "yt_audio.wav")
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "--no-playlist",
            "--js-runtimes",
            "node",
            "-o",
            audio_path,
            url.strip(),
        ]
        if ffmpeg_location:
            cmd += ["--ffmpeg-location", ffmpeg_location]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("[YouTube] yt-dlp failed: %s", result.stderr)
            raise HTTPException(
                status_code=500,
                detail=f"YouTube download failed: {result.stderr[:500]}",
            )
        if not os.path.exists(audio_path):
            raise HTTPException(
                status_code=500,
                detail="Audio file was not created. Check the URL.",
            )

        t_download = round(time.time() - t0, 2)
        logger.info("[YouTube] Download done in %.2fs", t_download)

        # ----- Same Whisper -> Translate -> Classify chain as /predict/pipeline -----
        t1 = time.time()
        whisper_result = whisper_state["model"].transcribe(
            audio_path, verbose=False, language="en"
        )
        segs = whisper_result["segments"]
        df = pd.DataFrame(
            {
                "Start": [hhmmss_ms(float(s["start"])) for s in segs],
                "End": [hhmmss_ms(float(s["end"])) for s in segs],
                "Sentence": [s["text"].strip() for s in segs],
            }
        )

        if diarize:
            try:
                from emotion_mlops.diarization.diarizer import (
                    assign_speakers_to_segments,
                )
                from emotion_mlops.diarization.diarizer import diarize as _diarize

                logger.info("[YouTube] Running speaker diarization...")
                turns = _diarize(audio_path)
                speakers = assign_speakers_to_segments(segs, turns)
                df["Speaker"] = speakers
                logger.info(
                    "[YouTube] Diarization: %d turns, %d unique speakers",
                    len(turns),
                    len(set(speakers)),
                )
            except Exception as e:
                logger.warning("[YouTube] Diarization failed: %s", e)

        t_whisper = round(time.time() - t1, 2)
        logger.info("[YouTube] Whisper: %d segments in %.2fs", len(df), t_whisper)

        t2 = time.time()
        if mt_state.get("model") and mt_state.get("tokenizer"):
            translations = translate_batch(
                df["Sentence"].tolist(),
                mt_state["tokenizer"],
                mt_state["model"],
                mt_state.get("device", "cpu"),
                bs=batch_size,
            )
        else:
            translations = [""] * len(df)
        df["Translation"] = translations
        t_translate = round(time.time() - t2, 2)
        logger.info("[YouTube] Translation done in %.2fs", t_translate)

        # Classification via Azure ML endpoint instead of local model
        t3 = time.time()
        import requests as _requests

        url_endpoint, key = _get_endpoint_url_and_key()
        emotions = []
        for sentence in df["Sentence"].tolist():
            try:
                r = _requests.post(
                    url_endpoint,
                    json={"sentence": sentence},
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=30,
                )
                result = r.json()
                emotions.append((result[0]["emotion"], result[0]["confidence"]))
            except Exception as e:
                logger.warning("[YouTube] Cloud prediction failed for sentence: %s", e)
                emotions.append(("neutral", 0.5))
        df["Emotion"] = [label for label, _ in emotions]
        df["Confidence"] = [conf for _, conf in emotions]
        t_classify = round(time.time() - t3, 2)
        logger.info("[YouTube] Classification done in %.2fs", t_classify)

        t_total = round(time.time() - t0, 2)
        logger.info("[YouTube] Total: %.2fs", t_total)

        metrics = PipelineMetrics(
            t_whisper=t_whisper,
            t_translate=t_translate,
            t_classify=t_classify,
            t_total=t_total,
            segment_count=len(emotions),
        )
        log_pipeline_metrics(metrics, run_id=f"youtube:{url[:60]}")

        # ----- Per-character summaries (only if diarization succeeded) -----
        from emotion_mlops.analysis.per_character import (
            detect_speaker_column,
            per_character_summary,
            per_character_top_moments,
        )

        speaker_col = detect_speaker_column(df)
        per_char_summary = per_character_summary(df, speaker_col) if speaker_col else {}
        per_char_moments = (
            per_character_top_moments(df, speaker_col, n=2) if speaker_col else {}
        )

        segment_cols = [
            "Start",
            "End",
            "Sentence",
            "Translation",
            "Emotion",
            "Confidence",
        ]
        if speaker_col:
            segment_cols.insert(2, speaker_col)
        segments_df = df[segment_cols].copy()
        if speaker_col and speaker_col != "Speaker":
            segments_df = segments_df.rename(columns={speaker_col: "Speaker"})

        return JSONResponse(
            {
                "url": url,
                "segments": segments_df.to_dict(orient="records"),
                "emotions": [label for label, _ in emotions],
                "t_download": t_download,
                "t_whisper": t_whisper,
                "t_translate": t_translate,
                "t_classify": t_classify,
                "t_total": t_total,
                "top_moments": build_top_moments(df, n=3),
                "per_character": per_char_summary,
                "per_character_top_moments": per_char_moments,
                "has_speakers": bool(speaker_col),
                "cast": build_cast_analysis(df),
                "timeline": build_timeline_from_df(df),
            }
        )


def _get_endpoint_url_and_key() -> tuple[str, str]:
    """
    Resolve the Azure ML endpoint URL and key.
    Looks at environment variables first, then falls back to a local
    config.json. Raises RuntimeError if neither source is configured.
    """
    import json
    from pathlib import Path

    url = os.environ.get("ML_ENDPOINT_URL")
    key = os.environ.get("ML_ENDPOINT_KEY")
    if url and key:
        return url, key

    config_path = Path.cwd() / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        url = cfg.get("ml_endpoint_url")
        key = cfg.get("ml_endpoint_key")
        if url and key:
            return url, key

    raise RuntimeError(
        "ML endpoint not configured. Set ML_ENDPOINT_URL and ML_ENDPOINT_KEY "
        "as env vars or add ml_endpoint_url/ml_endpoint_key to config.json"
    )


@app.post("/predict/cloud")
async def predict_cloud(request: TextRequest):  # pragma: no cover
    """
    Predict emotion via the Azure ML Online Endpoint.

    Useful when the local model can't be loaded (CI, lightweight devbox).
    Forwards the request to the deployed endpoint and re-shapes its
    response to match our normal text-prediction format.
    """
    import requests as _requests

    try:
        url, key = _get_endpoint_url_and_key()
        response = _requests.post(
            url,
            json={"sentence": request.text},
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        log_prediction(
            request.text,
            result[0]["emotion"],
            result[0]["confidence"],
            "azure-ml-endpoint",
        )
        return {
            "text": request.text,
            "emotion": result[0]["emotion"],
            "confidence": result[0]["confidence"],
            "all_emotions": result[0].get("all_emotions", {}),
            "source": "azure-ml-endpoint",
        }
    except Exception as e:
        logger.error("[API] Cloud prediction failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Cloud endpoint error: {e}")


@app.get("/endpoint/info")
async def endpoint_info():  # pragma: no cover
    """Report the configured Azure ML endpoint name and traffic split."""
    import yaml

    url = os.environ.get("ML_ENDPOINT_URL", "")
    traffic = {"blue": 70, "green": 30}
    endpoint_name = "emotion-classifier-endpoint"
    state = "Succeeded"

    try:
        config_path = _REPO_ROOT / "configs" / "azure_config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                az_cfg = yaml.safe_load(f) or {}
            deployment = az_cfg.get("deployment", {})
            endpoint_name = deployment.get("endpoint_name", endpoint_name)
            traffic = deployment.get("traffic", traffic)
    except Exception as e:
        logger.warning("[API] Could not read azure_config.yaml: %s", e)

    return {
        "endpoint_name": endpoint_name,
        "scoring_uri": url,
        "traffic": traffic,
        "provisioning_state": state,
        "source": "azure_config.yaml",
    }


# ===========================================================================
# Feedback endpoints, Card #1486 (continuous retraining with user feedback)
#
# The frontend lets users click thumbs-up / thumbs-down on every predicted segment. Those
# clicks are written to the feedback DB and can later be exported and
# merged with the training data to retrain the classifier.
# ===========================================================================


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Record user feedback on a model prediction.

    If is_correct is False, the user should also provide user_label so the
    correction can be used for retraining.

    After recording, an auto-retrain CHECK runs in the background: if enough
    new corrections have accumulated (default 20) AND auto-retrain is enabled
    (env AUTO_RETRAIN_ENABLED=1), it fires a live retrain + promote. The check
    runs as a BackgroundTask so it never delays this response, and does nothing
    unless explicitly enabled, so a demo never triggers an Azure job by accident.

    Args:
        payload: FeedbackRequest with sentence, prediction, and correctness.
        background_tasks: FastAPI background task queue (injected).

    Returns:
        FeedbackResponse with the new row's ID.

    Raises:
        HTTPException 422: If is_correct=False but user_label is missing.
        HTTPException 500: If DB write fails.
    """
    if not payload.is_correct and not payload.user_label:
        raise HTTPException(
            status_code=422,
            detail="user_label is required when is_correct is False.",
        )

    try:
        new_id = record_feedback(
            sentence=payload.sentence,
            predicted_label=payload.predicted_label,
            predicted_conf=payload.predicted_conf,
            is_correct=payload.is_correct,
            model_version=payload.model_version,
            user_label=payload.user_label,
            source=payload.source,
            file_id=payload.file_id,
        )
    except Exception as e:
        logger.error("[Feedback] Failed to record: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")

    # Fire-and-forget: check the 20-correction threshold off the response path.
    # No-op unless AUTO_RETRAIN_ENABLED=1, so demos are safe.
    from emotion_mlops.feedback.auto_retrain import maybe_trigger_after_feedback

    background_tasks.add_task(maybe_trigger_after_feedback)

    return FeedbackResponse(id=new_id)


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
def feedback_stats():
    """Return aggregated feedback statistics for the dashboard."""
    try:
        stats = get_stats()
    except Exception as e:
        logger.error("[Feedback] Failed to compute stats: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {e}")
    return FeedbackStatsResponse(**stats)


@app.get("/feedback/export")
def feedback_export():
    """Export every user correction as a list of {Sentence, Emotion} records."""
    try:
        rows = export_corrections_for_retraining()
    except Exception as e:
        logger.error("[Feedback] Export failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
    return {"count": len(rows), "records": rows}


@app.get("/feedback/auto-check")
def feedback_auto_check(threshold: int = 10):
    """
    Show how close we are to the next automatic retrain.

    Returns the number of new corrections collected since the last retrain,
    the threshold, whether a retrain is currently in flight, whether auto-retrain
    is enabled, and the recent retrain history.

    This is read-only, it never triggers a retrain. Use it in a dashboard or
    to demo the flywheel ("18 / 20 corrections until the model auto-retrains").

    Args:
        threshold: Corrections required to fire (default 20).
    """
    import os

    from emotion_mlops.feedback.auto_retrain import should_retrain
    from emotion_mlops.feedback.db import get_retrain_history

    try:
        status = should_retrain(threshold)
        history = get_retrain_history(limit=10)
    except Exception as e:
        logger.error("[Feedback] auto-check failed: %s", e)
        raise HTTPException(status_code=500, detail=f"auto-check failed: {e}")

    return {
        "new_corrections": status["new_corrections"],
        "threshold": status["threshold"],
        "remaining": max(0, status["threshold"] - status["new_corrections"]),
        "ready": status["ready"],
        "in_flight": status["in_flight"],
        "watermark": status["watermark"],
        "auto_retrain_enabled": os.getenv("AUTO_RETRAIN_ENABLED") == "1",
        "history": history,
    }


@app.post("/feedback/retrain")
async def feedback_retrain(dry_run: bool = True):  # pragma: no cover
    """Close the data flywheel: pull corrections, merge with raw data, and upload."""
    import asyncio

    from emotion_mlops.feedback.retrain_trigger import run_retrain

    # The retrain function is synchronous and slow, so we run it on the
    # default executor to keep the event loop responsive.
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, run_retrain, dry_run)
    except Exception as e:
        logger.error("[Feedback] Retrain failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")
    return result.to_dict()


# ===========================================================================
# Summarization endpoint (Card #2570 nice-to-have)
# Uses distilbart-cnn-6-6 to generate a 1-2 sentence summary from either
# free-form text or a list of pipeline segments.
# ===========================================================================


@app.post("/summarize")
async def summarize_endpoint(payload: dict):  # pragma: no cover
    """Summarize a transcript or arbitrary text using distilbart-cnn-6-6."""
    import asyncio

    from emotion_mlops.summary.summarizer import summarize, summarize_segments

    text = payload.get("text")
    segments = payload.get("segments")
    if not text and not segments:
        raise HTTPException(
            status_code=422,
            detail="Provide either `text` (string) or `segments` (list of dicts).",
        )

    loop = asyncio.get_event_loop()
    try:
        # `summarize_segments` is preferred when the caller already has
        # pipeline output, it preserves sentence boundaries.
        if segments:
            result = await loop.run_in_executor(
                None, summarize_segments, segments, "Sentence"
            )
        else:
            result = await loop.run_in_executor(None, summarize, text)
    except Exception as e:
        logger.error("[Summary] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
    if result is None:
        raise HTTPException(status_code=422, detail="Nothing to summarize.")
    return result


# ===========================================================================
# Q&A endpoint (Card #2570 nice-to-have)
# Asks Groq-hosted Llama to answer questions about a transcript.
# ===========================================================================


@app.post("/ask")
async def ask_endpoint(payload: dict):  # pragma: no cover
    """Ask a question about a transcript using a Groq-hosted LLM."""
    import asyncio

    from emotion_mlops.qa.groq_qa import ask_question

    question = (payload.get("question") or "").strip()
    segments = payload.get("segments") or []
    if not question:
        raise HTTPException(status_code=422, detail="`question` is required.")
    if not segments:
        raise HTTPException(
            status_code=422, detail="`segments` must be a non-empty list."
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: ask_question(
                question=question,
                segments=segments,
                max_tokens=payload.get("max_tokens", 512),
                temperature=payload.get("temperature", 0.2),
            ),
        )
    except RuntimeError as e:
        # `ask_question` raises RuntimeError when GROQ_API_KEY is missing.
        logger.warning("[QA] Service unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("[QA] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Q&A failed: {e}")
    return result


# ===========================================================================
# Compare endpoint, Sprint 4 (Ceren)
# Compare two pre-analysed pipeline results side-by-side. If the caller
# doesn't provide a label for a side, we auto-generate one with Groq.
# ===========================================================================


class CompareInput(BaseModel):
    """One side of a comparison request."""

    type: str
    label: str | None = None
    result: dict


class CompareRequest(BaseModel):
    """Request body for /compare."""

    left: CompareInput
    right: CompareInput


def _dominant_label(segments: list) -> str | None:
    """Most frequent Emotion label across segments, or None if empty."""
    if not segments:
        return None
    from collections import Counter

    labels = [s.get("Emotion") for s in segments if s.get("Emotion")]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


@app.post("/compare")
async def compare_endpoint(payload: CompareRequest):
    """Compare two pre-analyzed pipeline results side-by-side."""
    try:
        from emotion_mlops.analysis.autotitle import generate_title
        from emotion_mlops.analysis.compare import compare as _compare

        left_label = payload.left.label
        right_label = payload.right.label

        # If the user didn't name a side, ask the AI to generate a label
        # based on the segments + cast + dominant emotion. This keeps the
        # Compare UI meaningful even for anonymous text snippets.
        if not left_label:
            left_label = generate_title(
                segments=payload.left.result.get("segments", []),
                cast=payload.left.result.get("cast", []),
                dominant_emotion=_dominant_label(
                    payload.left.result.get("segments", [])
                ),
                fallback="Show A",
            )
        if not right_label:
            right_label = generate_title(
                segments=payload.right.result.get("segments", []),
                cast=payload.right.result.get("cast", []),
                dominant_emotion=_dominant_label(
                    payload.right.result.get("segments", [])
                ),
                fallback="Show B",
            )

        result = _compare(
            left=payload.left.result,
            right=payload.right.result,
        )

        # Stamp the labels and types back onto the response so the frontend
        # doesn't have to re-derive them.
        result["left_label"] = left_label
        result["right_label"] = right_label
        result["left_type"] = payload.left.type
        result["right_type"] = payload.right.type

        return result
    except Exception as e:
        logger.error("[Compare] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")


# ===========================================================================
# Timeline endpoint, Sprint 4 (Ceren)
# Build a Chart.js-friendly time series of emotion intensity from segments.
# ===========================================================================


class TimelineRequest(BaseModel):
    """Request body for /predict/timeline."""

    segments: list
    smooth_window: int = 1
    peak_threshold: float = 0.7


@app.post("/predict/timeline")
async def predict_timeline(payload: TimelineRequest):
    """Build an emotion timeline (time series) from pipeline segments."""
    if not payload.segments:
        raise HTTPException(
            status_code=422,
            detail="`segments` must be a non-empty list.",
        )

    try:
        from emotion_mlops.analysis.timeline import build_timeline

        timeline = build_timeline(
            segments=payload.segments,
            smooth_window=payload.smooth_window,
            peak_threshold=payload.peak_threshold,
        )
        return timeline
    except Exception as e:
        logger.error("[Timeline] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Timeline build failed: {e}")


# ===========================================================================
# Scene Detection endpoint, Sprint 4 (Ceren, Day 3)
# Group consecutive segments into emotional scenes; optionally generate
# an AI title per scene using Groq.
# ===========================================================================


class SceneRequest(BaseModel):
    """Request body for /predict/scenes."""

    segments: list
    min_segments_per_scene: int = 2
    boundary_change_threshold: float = 0.4
    generate_titles: bool = False


@app.post("/predict/scenes")
async def predict_scenes(payload: SceneRequest):
    """Detect emotional scenes in a transcript."""
    if not payload.segments:
        raise HTTPException(
            status_code=422,
            detail="`segments` must be a non-empty list.",
        )

    try:
        from emotion_mlops.analysis.scenes import detect_scenes

        scenes = detect_scenes(
            segments=payload.segments,
            min_segments_per_scene=payload.min_segments_per_scene,
            boundary_change_threshold=payload.boundary_change_threshold,
            generate_titles=payload.generate_titles,
        )
        return {
            "scenes": scenes,
            "scene_count": len(scenes),
        }
    except Exception as e:
        logger.error("[Scenes] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Scene detection failed: {e}")


# ===========================================================================
# Trailer Highlights endpoint, Sprint 4 (Ceren, Day 4)
# Pick the best clips for a short trailer, ensuring mood diversity.
# ===========================================================================


class TrailerRequest(BaseModel):
    """Request body for /predict/trailer."""

    scenes: list
    target_duration: float = 30.0
    target_clip_duration: float = 8.0
    max_clips: int = 3
    enforce_mood_diversity: bool = True
    generate_taglines: bool = False


@app.post("/predict/trailer")
async def predict_trailer(payload: TrailerRequest):
    """Pick the best clips for a short trailer."""
    if not payload.scenes:
        raise HTTPException(
            status_code=422,
            detail="`scenes` must be a non-empty list. Call /predict/scenes first.",
        )

    try:
        from emotion_mlops.analysis.trailer import pick_trailer_highlights

        result = pick_trailer_highlights(
            scenes=payload.scenes,
            target_duration=payload.target_duration,
            target_clip_duration=payload.target_clip_duration,
            max_clips=payload.max_clips,
            enforce_mood_diversity=payload.enforce_mood_diversity,
            generate_taglines=payload.generate_taglines,
        )
        return result
    except Exception as e:
        logger.error("[Trailer] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Trailer generation failed: {e}")


# ===========================================================================
# Content Score endpoint, Sprint 4 (Ceren, Day 5)
# Produce a producer-style scorecard: letter grade, audience match, and
# six production axes (drama, comedy, intensity, pacing, etc.).
# ===========================================================================


class ScoreRequest(BaseModel):
    """Request body for /predict/score."""

    segments: list
    cast: list = []
    scenes: list = []
    use_ai_classification: bool = True


@app.post("/predict/score")
async def predict_score(payload: ScoreRequest):
    """Generate a content scorecard for producers / content strategists."""
    if not payload.segments:
        raise HTTPException(
            status_code=422,
            detail="`segments` must be a non-empty list.",
        )

    try:
        from emotion_mlops.analysis.content_score import score_content

        result = score_content(
            segments=payload.segments,
            cast=payload.cast,
            scenes=payload.scenes,
            use_ai_classification=payload.use_ai_classification,
        )
        return result
    except Exception as e:
        logger.error("[Score] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Content scoring failed: {e}")


# ===========================================================================
# Emotion Shifts endpoint, Sprint 4 (Ceren, Day 6)
# Detect emotional turning points: where the show pivots from one mood to
# another. Useful for spotting plot beats and rhetorical hooks.
# ===========================================================================


class ShiftsRequest(BaseModel):
    """Request body for /predict/shifts."""

    segments: list
    min_magnitude: float = 0.5
    min_confidence: float = 0.5


@app.post("/predict/shifts")
async def predict_shifts(payload: ShiftsRequest):
    """Detect emotional shifts (turning points) between consecutive segments."""
    if not payload.segments:
        raise HTTPException(
            status_code=422,
            detail="`segments` must be a non-empty list.",
        )

    try:
        from emotion_mlops.analysis.shifts import detect_shifts

        shifts = detect_shifts(
            segments=payload.segments,
            min_magnitude=payload.min_magnitude,
            min_confidence=payload.min_confidence,
        )
        return {"shifts": shifts, "shift_count": len(shifts)}
    except Exception as e:
        logger.error("[Shifts] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Shift detection failed: {e}")


# ===========================================================================
# Persuasion endpoint Sprint 4 (Ceren, Day 6)
# Measure rhetorical strength + emotional appeal of the speaker(s) and
# optionally let Groq write a one-paragraph verdict.
# ===========================================================================


class PersuasionRequest(BaseModel):
    """Request body for /predict/persuasion."""

    segments: list
    use_ai_assessment: bool = True


@app.post("/predict/persuasion")
async def predict_persuasion(payload: PersuasionRequest):
    """Score how persuasive the speaker(s) in the transcript are."""
    if not payload.segments:
        raise HTTPException(
            status_code=422,
            detail="`segments` must be a non-empty list.",
        )

    try:
        from emotion_mlops.analysis.persuasion import analyze_persuasion

        result = analyze_persuasion(
            segments=payload.segments,
            use_ai_assessment=payload.use_ai_assessment,
        )
        return result
    except Exception as e:
        logger.error("[Persuasion] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Persuasion analysis failed: {e}")


# ===========================================================================
# AI Summary endpoint, Sprint 4 (Ceren)
# Synthesises the entire analysis (segments + cast + scenes + score +
# persuasion) into a 2-3 sentence "AI verdict" using Groq Llama 3.1.
# Renders at the top of the Analyse results dashboard.
# ===========================================================================


@app.post("/predict/summary")
async def predict_summary(payload: dict):
    """Generate an AI 2-3 sentence verdict from the full analysis."""
    segments = payload.get("segments", [])
    if not segments:
        raise HTTPException(
            status_code=422,
            detail="`segments` must be a non-empty list.",
        )

    try:
        from emotion_mlops.analysis.summary import generate_summary

        # All extra signals (cast, scenes, score, persuasion) are optional.
        # The summariser is robust to missing fields and falls back to a
        # deterministic one-liner when GROQ_API_KEY is not configured.
        result = generate_summary(
            segments=segments,
            cast=payload.get("cast", []),
            scenes=payload.get("scenes", []),
            score=payload.get("score"),
            persuasion=payload.get("persuasion"),
        )
        return result
    except Exception as e:
        logger.error("[Summary] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {e}")


# ===========================================================================
# Cast Analysis endpoint, Sprint 4 (Ceren)
# Detect named characters with spaCy NER and profile their emotions
# using the DeBERTa classifier.
# ===========================================================================


class CastRequest(BaseModel):
    """Request body for /predict/cast."""

    text: str
    top_n: int = 10


@app.get("/predictions")
async def get_predictions(limit: int = 100):
    """Return the most recent raw predictions (debug / admin view)."""
    from emotion_mlops.api.database import get_recent_predictions

    return get_recent_predictions(limit=limit)


@app.post("/predict/cast")
async def predict_cast(payload: CastRequest):  # pragma: no cover
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="`text` is required.")

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="`text` is required.")

    try:
        from emotion_mlops.analysis.cast_analyzer import analyze_cast

        cast = analyze_cast(
            text=text,
            classifier=classifier,
            top_n=payload.top_n,
        )
    except Exception as e:
        logger.error("[Cast] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Cast analysis failed: {e}")

    return {
        "cast": cast,
        "total_characters": len(cast),
    }


# ===========================================================================
# Admin login, gates the Monitor dashboard.
# The Monitor page shows team-internal data (feedback stats, retrain
# history, endpoint traffic), so the frontend asks for a login before
# rendering it. Credentials and token rules live in api/admin_auth.py.
# ===========================================================================


@app.post("/admin/login", response_model=AdminLoginResponse)
def admin_login(payload: AdminLoginRequest):
    """Check admin credentials and hand back a signed, expiring token."""
    from emotion_mlops.api import admin_auth

    if not admin_auth.check_credentials(payload.username, payload.password):
        # Same message for wrong user and wrong password, so the response
        # does not reveal which one was off.
        raise HTTPException(status_code=401, detail="Wrong username or password.")

    token, expires_at = admin_auth.create_token()
    logger.info("[Admin] Successful dashboard login for '%s'", payload.username)
    return AdminLoginResponse(ok=True, token=token, expires_at=expires_at)


@app.get("/admin/verify")
def admin_verify(authorization: str | None = Header(None)):
    """Tell the frontend whether its stored admin token is still valid."""
    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()

    from emotion_mlops.api import admin_auth

    if not admin_auth.verify_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return {"ok": True}


# ===========================================================================
# Static file serving, EmotionScope frontend.
#
# IMPORTANT: these mounts MUST be the very last lines in this file. Mounting
# "/" on a StaticFiles instance is a catch-all that takes precedence over
# anything declared after it, so any endpoint declared below would become
# unreachable.
# ===========================================================================
app.mount(
    "/static",
    StaticFiles(directory=str(_STATIC_DIR)),
    name="static_assets",
)
app.mount(
    "/",
    StaticFiles(directory=str(_STATIC_DIR), html=True),
    name="static_root",
)
