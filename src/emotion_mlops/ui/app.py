"""
EmotionScope, Streamlit frontend for the emotion classification pipeline.

Pages:
  1. Analyse, run the pipeline on a file or YouTube URL
  2. Dashboard, operational and performance metrics with bottleneck tracing
  3. History, persistent run log (stored in runs.json next to this file)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# Suppress noisy transformers __path__ deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Logging
LOG_FILE = Path(__file__).parent / "emotionscope.log"
REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_FILE = REPO_ROOT / "logs" / "ui.log"
BACKEND_LOG = REPO_ROOT / "logs" / "api.log"
(REPO_ROOT / "logs").mkdir(exist_ok=True)


def setup_logging() -> None:
    """Configure root logger, writes to console and to emotionscope.log."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


setup_logging()
logger = logging.getLogger("emotionscope")

st.set_page_config(
    page_title="EmotionScope",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
EMOTION_COLORS = {
    "anger": "#2542b9",
    "disgust": "#6aab3a",
    "fear": "#3f2d5a",
    "happiness": "#950f31",
    "sadness": "#3a6fb5",
    "surprise": "#dd7cad",
    "neutral": "#aaaaaa",
}

RUNS_FILE = Path(__file__).parent / "runs.json"
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
MODEL_DIR = REPO_ROOT / "models" / "best_model_v3"
API_URL = "http://localhost:8000"


# Backend auto-start
def ensure_backend() -> None:
    if st.session_state.get("_backend_ready"):
        return

    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        if r.status_code == 200:
            st.session_state["_backend_ready"] = True
            return
    except requests.exceptions.ConnectionError:
        pass

    if st.session_state.get("_backend_proc") is not None:
        return

    logger.info("Launching uvicorn subprocess...")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    backend_log_fh = open(BACKEND_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "emotion_mlops.api.app:app",
            "--port",
            "8000",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=backend_log_fh,
        stderr=backend_log_fh,
    )
    st.session_state["_backend_proc"] = proc
    logger.info("Uvicorn PID %d, logs at %s", proc.pid, BACKEND_LOG)
    # <- function ends here, no loop


# CSS, light, single font family

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #f8f8f6;
    color: #1a1a1a;
}
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e5e5;
}
h1, h2, h3 { font-weight: 600; }

.page-title { font-size: 1.8rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.2rem; }
.page-sub   { font-size: 0.9rem; color: #6b6b6b; margin-bottom: 2rem; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #999;
    border-bottom: 1px solid #e5e5e5;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

.kpi-card { background: #ffffff; border: 1px solid #e5e5e5; border-radius: 8px; padding: 1.2rem 1.4rem; }
.kpi-val  { font-size: 2rem; font-weight: 600; color: #1a1a1a; line-height: 1.1; }
.kpi-lbl  { font-size: 0.75rem; color: #999; margin-bottom: 0.3rem; }

.trace-row   { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.6rem; }
.trace-label { width: 200px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #6b6b6b; flex-shrink: 0; }
.trace-bg    { flex: 1; background: #efefef; border-radius: 3px; height: 6px; }
.trace-fill  { height: 6px; border-radius: 3px; background: #1a1a1a; }
.trace-time  { width: 55px; text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #1a1a1a; flex-shrink: 0; }

.bottleneck-box   { background: #fff8ed; border: 1px solid #f0c060; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.8rem; }
.bottleneck-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #b07800; margin-bottom: 0.3rem; }

.log-box {
    background: #f4f4f2;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.7;
    color: #333;
    max-height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.stButton > button {
    background: #1a1a1a; color: #ffffff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; letter-spacing: 0.04em;
    border: none; padding: 0.6rem 1.8rem; border-radius: 6px;
}
.stButton > button:hover { background: #333; }

[data-testid="stFileUploadDropzone"] {
    background: #ffffff !important;
    border: 1px dashed #ccc !important;
    border-radius: 8px !important;
}
.stProgress > div > div { background: #1a1a1a !important; }
hr { border-color: #e5e5e5; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# Persistence


def load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        logger.debug("runs.json not found, starting with empty history")
        return []
    try:
        with open(RUNS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            r["timestamp"] = datetime.fromisoformat(r["timestamp"])
        logger.debug("Loaded %d run(s) from %s", len(data), RUNS_FILE)
        return data
    except Exception as exc:
        logger.warning("Could not load runs.json: %s", exc)
        return []


def save_runs(runs: list[dict]) -> None:
    rows = []
    for r in runs:
        row = dict(r)
        row["timestamp"] = r["timestamp"].isoformat()
        rows.append(row)
    with open(RUNS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.debug("Saved %d run(s) to %s", len(runs), RUNS_FILE)


# Session state

if "runs" not in st.session_state:
    st.session_state.runs = load_runs()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# Log paths once per session only
if "paths_logged" not in st.session_state:
    logger.debug("REPO_ROOT resolved to: %s", REPO_ROOT)
    logger.debug("CONFIG_PATH: %s (exists=%s)", CONFIG_PATH, CONFIG_PATH.exists())
    logger.debug("MODEL_DIR:   %s (exists=%s)", MODEL_DIR, MODEL_DIR.exists())
    st.session_state["paths_logged"] = True

ensure_backend()


# Chart helpers


def fmt_time(s: float) -> str:
    h = int(s // 3600)
    s %= 3600
    m = int(s // 60)
    s %= 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def trace_row(label: str, secs: float, total: float) -> None:
    pct = min(secs / total * 100, 100) if total > 0 else 0
    st.markdown(
        f'<div class="trace-row">'
        f'<span class="trace-label">{label}</span>'
        f'<div class="trace-bg"><div class="trace-fill" style="width:{pct:.1f}%"></div></div>'
        f'<span class="trace-time">{secs:.2f}s</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def emotion_bar_chart(emotions: list[str]) -> go.Figure:
    counts = {e: emotions.count(e) for e in EMOTIONS if emotions.count(e) > 0}
    df = pd.DataFrame(
        {"emotion": list(counts.keys()), "count": list(counts.values())}
    ).sort_values("count")
    fig = px.bar(
        df,
        x="count",
        y="emotion",
        orientation="h",
        color="emotion",
        color_discrete_map=EMOTION_COLORS,
        template="plotly_white",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=8, b=8),
        font_family="IBM Plex Sans",
        yaxis_title="",
        xaxis_title="Segments",
        xaxis=dict(gridcolor="#efefef"),
        yaxis=dict(gridcolor="#efefef"),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def emotion_pie_chart(emotions: list[str]) -> go.Figure:
    counts = {e: emotions.count(e) for e in EMOTIONS if emotions.count(e) > 0}
    fig = go.Figure(
        go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            marker_colors=[EMOTION_COLORS[e] for e in counts],
            hole=0.55,
            textinfo="percent",
            textfont=dict(family="IBM Plex Mono", size=11),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(family="IBM Plex Sans", size=11)),
        margin=dict(l=0, r=0, t=8, b=8),
    )
    return fig


def latency_trend_chart(runs: list[dict]) -> go.Figure:
    successful = [r for r in runs if r["status"] == "success"]
    if not successful:
        return go.Figure()
    df = pd.DataFrame(
        [
            {
                "Timestamp": r["timestamp"],
                "Whisper": r["t_whisper"],
                "Translation": r["t_translate"],
                "Classification": r["t_classify"],
                "Total": r["t_total"],
            }
            for r in successful
        ]
    )
    fig = go.Figure()
    palette = {
        "Whisper": "#e08c2a",
        "Translation": "#3a6fb5",
        "Classification": "#2a9d6a",
        "Total": "#1a1a1a",
    }
    for col, color in palette.items():
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df[col],
                name=col,
                mode="lines+markers",
                line=dict(color=color, width=1.8),
                marker=dict(size=5),
            )
        )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="IBM Plex Sans",
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#efefef"),
        yaxis=dict(gridcolor="#efefef", title="Seconds"),
        margin=dict(l=0, r=0, t=8, b=8),
        hovermode="x unified",
    )
    return fig


# Log viewer helper


def read_log_tail(n: int = 40) -> str:
    if not LOG_FILE.exists():
        return "(no log file yet)"
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(could not read log: {exc})"


# YouTube helper


def download_youtube_audio(url: str, out_dir: str) -> Optional[str]:
    """Download audio from YouTube via yt-dlp. Returns WAV path or None."""
    logger.info("[YouTube] Downloading audio from: %s", url)

    ffmpeg_location = None
    try:
        import imageio_ffmpeg

        ffmpeg_location = imageio_ffmpeg.get_ffmpeg_exe()
        logger.debug(
            "[YouTube] ffmpeg resolved via imageio-ffmpeg: %s", ffmpeg_location
        )
    except ImportError:
        logger.warning("[YouTube] imageio-ffmpeg not found, relying on system ffmpeg")

    out_path = os.path.join(out_dir, "yt_audio.wav")
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
        "none",
        "-o",
        out_path,
        url,
    ]
    if ffmpeg_location:
        cmd += ["--ffmpeg-location", ffmpeg_location]

    logger.debug("[YouTube] Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("[YouTube] yt-dlp failed:\n%s", result.stderr)
        st.error(f"yt-dlp error:\n{result.stderr}")
        return None

    logger.info("[YouTube] Audio saved to: %s", out_path)
    return out_path


# Pipeline runner, sends file to FastAPI backend


def run_pipeline(
    audio_path,
    config_path,
    out_csv,
    is_transcript=False,
    whisper_model=None,
    batch_size=None,
    device=None,
    progress_cb=None,
):
    """
    Sends the audio/CSV file to the FastAPI backend and returns results.
    """

    def _progress(pct: int, msg: str) -> None:
        logger.info("[Progress %d%%] %s", pct, msg)
        if progress_cb:
            progress_cb(pct, msg)

    logger.info("=" * 60)
    logger.info("Sending pipeline request to: %s", API_URL)

    # -- Check backend is reachable -------------------------------------------
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code != 200 or not health.json().get("model_loaded"):
            st.error(
                "FastAPI backend is not ready yet, model still loading. Wait a moment and try again."
            )
            return None
    except requests.exceptions.ConnectionError:
        # One retry before giving up
        st.warning("Backend still starting up, retrying in 5 s...")
        time.sleep(5)
        try:
            health = requests.get(f"{API_URL}/health", timeout=5)
            if health.status_code != 200 or not health.json().get("model_loaded"):
                st.error(
                    "Backend not ready yet. Check backend.log at the repo root for errors."
                )
                return None
        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot connect to the backend at {API_URL}\n\n"
                f"Check **{BACKEND_LOG}** for the startup error.\n\n"
                "Or start it manually in a second terminal:\n\n"
                "```powershell\n$env:PYTHONPATH='src'\n"
                "uvicorn emotion_mlops.api.app:app --port 8000\n```"
            )
            return None

    _progress(5, "Backend ready, uploading file...")

    # -- Read config for settings ---------------------------------------------
    import yaml

    cfg = {}
    if Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    whisper_model = whisper_model or cfg.get("whisper_model", "base")
    batch_size = batch_size or int(cfg.get("batch_size", 16))
    max_length = int(cfg.get("max_length", 192))

    # -- Send file to backend -------------------------------------------------
    _progress(10, "Processing on backend, this may take a moment...")
    try:
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{API_URL}/predict/pipeline",
                files={"file": (Path(audio_path).name, f)},
                data={
                    "is_transcript": str(is_transcript).lower(),
                    "whisper_model": whisper_model,
                    "batch_size": batch_size,
                    "max_length": max_length,
                },
                timeout=600,
            )
    except requests.exceptions.Timeout:
        st.error("Request timed out. The file may be too large.")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        logger.exception("[Pipeline] Request failed: %s", exc)
        return None

    _progress(95, "Done, loading results...")

    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        st.error(f"Backend error ({response.status_code}): {detail}")
        return None

    data = response.json()

    # -- Rebuild DataFrame ----------------------------------------------------
    df = pd.DataFrame(data["segments"])
    emotions = data["emotions"]

    # Save CSV locally so the download button still works
    df[["Start", "End", "Sentence", "Translation", "Emotion"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )

    t_whisper = data["t_whisper"]
    t_translate = data["t_translate"]
    t_classify = data["t_classify"]
    t_total = data["t_total"]

    logger.info("Pipeline complete: %d segments in %.2fs", len(emotions), t_total)
    logger.info("=" * 60)

    _progress(100, "Done.")

    return {
        "df": df,
        "segments": [
            (row["Start"], row["End"], row["Sentence"], row["Emotion"])
            for row in data["segments"]
        ],
        "emotions": emotions,
        "t_whisper": t_whisper,
        "t_translate": t_translate,
        "t_classify": t_classify,
        "t_total": t_total,
    }


# Sidebar

with st.sidebar:
    st.markdown("**EmotionScope**")
    st.markdown(
        '<div style="font-size:0.8rem;color:#999;margin-bottom:1.5rem;">TV emotion analysis pipeline</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Page", ["Analyse", "Dashboard", "History"], label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.78rem;color:#999;line-height:2;">'
        'Model: <span style="color:#1a1a1a">DeBERTa-v3-large</span><br>'
        'ASR: <span style="color:#1a1a1a">Whisper</span><br>'
        'MT: <span style="color:#1a1a1a">opus-mt-en-nl</span><br>'
        'Labels: <span style="color:#1a1a1a">7 emotions</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    for e, c in EMOTION_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">'
            f'<div style="width:9px;height:9px;border-radius:50%;background:{c};flex-shrink:0;"></div>'
            f'<span style="font-size:0.78rem;color:#6b6b6b;text-transform:capitalize;">{e}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Show backend status in sidebar
    st.markdown("---")
    if st.session_state.get("_backend_ready"):
        st.markdown(
            '<div style="font-size:0.75rem;color:#2a9d6a;">● Backend running</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.75rem;color:#e08c2a;">● Backend starting…</div>',
            unsafe_allow_html=True,
        )
        if BACKEND_LOG.exists():
            st.caption(f"Logs: {BACKEND_LOG}")


# ===========================================================================
# PAGE: ANALYSE
# ===========================================================================
if page == "Analyse":

    st.markdown('<div class="page-title">Analyse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Run the pipeline on an audio file, video file, transcript CSV, or a YouTube URL.</div>',
        unsafe_allow_html=True,
    )

    col_in, col_cfg = st.columns([3, 2], gap="large")

    with col_in:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Input type",
            ["Upload audio / video", "Upload transcript CSV", "YouTube URL"],
            label_visibility="collapsed",
        )

        uploaded_file = None
        yt_url = ""

        if input_mode == "Upload audio / video":
            uploaded_file = st.file_uploader(
                "Supported formats: WAV, MP3, MP4, MKV",
                type=["wav", "mp3", "mp4", "mkv"],
                label_visibility="visible",
            )
        elif input_mode == "Upload transcript CSV":
            uploaded_file = st.file_uploader(
                "Required columns: Sentence (or Text / Utterance). Optional: Start, End.",
                type=["csv"],
                label_visibility="visible",
            )
        elif input_mode == "YouTube URL":
            yt_url = st.text_input(
                "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
            )

    with col_cfg:
        st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
        whisper_size = st.selectbox(
            "Whisper model",
            ["tiny", "base", "small", "medium"],
            index=1,
            help=(
                "tiny, 39 MB,  ~1 GB RAM, fastest, lowest accuracy\n"
                "base, 74 MB,  ~1 GB RAM, good balance (recommended for CPU)\n"
                "small, 244 MB, ~2 GB RAM, better accuracy, slower\n"
                "medium, 769 MB, ~5 GB RAM, high accuracy, slow on CPU"
            ),
        )
        batch_size = st.slider("Batch size", 4, 32, 16, step=4)
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        config_override = st.text_input(
            "Config path (optional)", placeholder=str(CONFIG_PATH)
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run pipeline", disabled=st.session_state.pipeline_running)

    if run_btn and not st.session_state.pipeline_running:
        st.session_state.pipeline_running = True

        config_path = config_override.strip() or str(CONFIG_PATH)
        logger.info(
            "Run button clicked, input_mode=%s config=%s", input_mode, config_path
        )

        if not Path(config_path).exists():
            logger.error("Config not found: %s", config_path)
            st.error(f"Config file not found: {config_path}")
            st.session_state.pipeline_running = False
            st.stop()

        with tempfile.TemporaryDirectory() as tmp:
            audio_path: Optional[str] = None
            is_transcript = False

            if input_mode == "YouTube URL":
                if not yt_url.strip():
                    st.error("Enter a YouTube URL.")
                    st.session_state.pipeline_running = False
                    st.stop()
                with st.spinner("Downloading audio from YouTube..."):
                    audio_path = download_youtube_audio(yt_url.strip(), tmp)
                if not audio_path:
                    st.session_state.pipeline_running = False
                    st.stop()

            elif input_mode == "Upload audio / video":
                if not uploaded_file:
                    st.error("Upload a file first.")
                    st.session_state.pipeline_running = False
                    st.stop()
                audio_path = os.path.join(tmp, uploaded_file.name)
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.read())
                logger.info(
                    "File written to temp: %s (%d bytes)",
                    audio_path,
                    os.path.getsize(audio_path),
                )

            elif input_mode == "Upload transcript CSV":
                if not uploaded_file:
                    st.error("Upload a CSV first.")
                    st.session_state.pipeline_running = False
                    st.stop()
                audio_path = os.path.join(tmp, "transcript.csv")
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.read())
                is_transcript = True
                logger.info("Transcript CSV written to temp: %s", audio_path)

            out_csv = os.path.join(tmp, "output.csv")

            prog = st.progress(0, text="Starting pipeline...")
            status = st.empty()
            result = run_pipeline(
                audio_path,
                config_path,
                out_csv,
                is_transcript=is_transcript,
                whisper_model=whisper_size,
                batch_size=batch_size,
                device=device,
                progress_cb=lambda pct, msg: (
                    prog.progress(pct, text=msg),
                    status.markdown(
                        f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#6b6b6b;">{msg}</span>',
                        unsafe_allow_html=True,
                    ),
                ),
            )
            prog.progress(100, text="Done.")
            prog.empty()
            status.empty()

            st.session_state.pipeline_running = False

            if result is None:
                st.stop()

            run_record = {
                "run_id": f"run_{len(st.session_state.runs)+1:03d}",
                "timestamp": datetime.now(),
                "input_type": input_mode,
                "segments": len(result["emotions"]),
                "t_whisper": result["t_whisper"],
                "t_translate": result["t_translate"],
                "t_classify": result["t_classify"],
                "t_total": result["t_total"],
                "emotions": result["emotions"],
                "status": "success",
            }
            st.session_state.runs.append(run_record)
            save_runs(st.session_state.runs)
            st.session_state.last_result = result

            if Path(out_csv).exists():
                with open(out_csv, "rb") as f:
                    st.session_state.last_csv = f.read()

    # Results
    res = st.session_state.last_result
    if res:
        st.markdown("---")
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

        emotions = res["emotions"]
        dominant = max(set(emotions), key=emotions.count) if emotions else "-"

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-lbl">Segments</div><div class="kpi-val">{len(emotions)}</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            dom_col = EMOTION_COLORS.get(dominant, "#1a1a1a")
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-lbl">Dominant emotion</div><div class="kpi-val" style="color:{dom_col}">{dominant}</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-lbl">Total runtime</div><div class="kpi-val">{res["t_total"]}s</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-lbl">Unique emotions</div><div class="kpi-val">{len(set(emotions))}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.markdown(
                '<div class="section-label">Distribution</div>', unsafe_allow_html=True
            )
            st.plotly_chart(
                emotion_bar_chart(emotions),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with c2:
            st.markdown(
                '<div class="section-label">Composition</div>', unsafe_allow_html=True
            )
            st.plotly_chart(
                emotion_pie_chart(emotions),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown(
            '<div class="section-label">Step trace</div>', unsafe_allow_html=True
        )
        trace_row("Whisper ASR", res["t_whisper"], res["t_total"])
        trace_row("MarianMT translation", res["t_translate"], res["t_total"])
        trace_row("DeBERTa classification", res["t_classify"], res["t_total"])
        overhead = max(
            round(
                res["t_total"]
                - res["t_whisper"]
                - res["t_translate"]
                - res["t_classify"],
                2,
            ),
            0,
        )
        trace_row("Overhead / I/O", overhead, res["t_total"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">Transcript</div>', unsafe_allow_html=True
        )
        st.dataframe(
            res["df"][["Start", "End", "Sentence", "Translation", "Emotion"]],
            use_container_width=True,
            height=340,
        )

        if "last_csv" in st.session_state:
            st.download_button(
                "Download CSV",
                data=st.session_state.last_csv,
                file_name=f"emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Run log"):
            st.markdown(
                f'<div class="log-box">{read_log_tail(60)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Full log: {LOG_FILE}")


# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================
elif page == "Dashboard":

    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Operational metrics and bottleneck analysis across all pipeline runs.</div>',
        unsafe_allow_html=True,
    )

    runs = st.session_state.runs
    successful = [r for r in runs if r["status"] == "success"]

    if not runs:
        st.info("No runs yet. Go to Analyse to run the pipeline.")
        st.stop()

    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    success_rate = round(len(successful) / len(runs) * 100, 1)
    avg_total = round(
        sum(r["t_total"] for r in successful) / max(len(successful), 1), 2
    )
    avg_segs = round(
        sum(r["segments"] for r in successful) / max(len(successful), 1), 1
    )
    total_segs = sum(r["segments"] for r in runs)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-lbl">Total runs</div><div class="kpi-val">{len(runs)}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-lbl">Success rate</div><div class="kpi-val">{success_rate}%</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-lbl">Avg latency</div><div class="kpi-val">{avg_total}s</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-lbl">Avg segments / run</div><div class="kpi-val">{avg_segs}</div></div>',
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-lbl">Total segments</div><div class="kpi-val">{total_segs}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Latency over time</div>', unsafe_allow_html=True
    )
    st.plotly_chart(
        latency_trend_chart(runs),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c_step, c_drift = st.columns(2, gap="large")

    with c_step:
        st.markdown(
            '<div class="section-label">Average step duration</div>',
            unsafe_allow_html=True,
        )
        avg_w = sum(r["t_whisper"] for r in successful) / max(len(successful), 1)
        avg_t = sum(r["t_translate"] for r in successful) / max(len(successful), 1)
        avg_c = sum(r["t_classify"] for r in successful) / max(len(successful), 1)
        avg_tot = avg_w + avg_t + avg_c

        fig_step = go.Figure(
            go.Bar(
                x=["Whisper ASR", "Translation", "Classification"],
                y=[avg_w, avg_t, avg_c],
                marker_color=["#e08c2a", "#3a6fb5", "#2a9d6a"],
                marker_line_width=0,
            )
        )
        fig_step.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="IBM Plex Sans",
            showlegend=False,
            yaxis=dict(gridcolor="#efefef", title="Avg seconds"),
            margin=dict(l=0, r=0, t=8, b=8),
        )
        st.plotly_chart(
            fig_step, use_container_width=True, config={"displayModeBar": False}
        )

        bottleneck = max(
            [("Whisper ASR", avg_w), ("Translation", avg_t), ("Classification", avg_c)],
            key=lambda x: x[1],
        )
        pct_bn = round(bottleneck[1] / avg_tot * 100) if avg_tot > 0 else 0
        st.markdown(
            f'<div class="bottleneck-box">'
            f'<div class="bottleneck-title">Bottleneck identified</div>'
            f"<b>{bottleneck[0]}</b>, {bottleneck[1]:.2f}s avg ({pct_bn}% of total)<br>"
            f'<span style="font-size:0.8rem;color:#6b6b6b;">Consider GPU acceleration or model quantisation to reduce this step.</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with c_drift:
        st.markdown(
            '<div class="section-label">Emotion share across runs</div>',
            unsafe_allow_html=True,
        )
        if len(successful) < 2:
            st.caption("Need at least 2 successful runs to show this chart.")
        else:
            rows = []
            for r in successful:
                total_e = len(r["emotions"]) or 1
                for e in EMOTIONS:
                    rows.append(
                        {
                            "Run": r["run_id"],
                            "Emotion": e,
                            "Share": r["emotions"].count(e) / total_e * 100,
                        }
                    )
            df_drift = pd.DataFrame(rows)
            fig_drift = px.area(
                df_drift,
                x="Run",
                y="Share",
                color="Emotion",
                color_discrete_map=EMOTION_COLORS,
                template="plotly_white",
            )
            fig_drift.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="IBM Plex Sans",
                legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                yaxis=dict(gridcolor="#efefef", title="% share"),
                margin=dict(l=0, r=0, t=8, b=8),
            )
            st.plotly_chart(
                fig_drift, use_container_width=True, config={"displayModeBar": False}
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Application log"):
        st.markdown(
            f'<div class="log-box">{read_log_tail(50)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Full log: {LOG_FILE}")


# ===========================================================================
# PAGE: HISTORY
# ===========================================================================
elif page == "History":

    st.markdown('<div class="page-title">History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">All pipeline runs, persisted in runs.json.</div>',
        unsafe_allow_html=True,
    )

    runs = st.session_state.runs
    if not runs:
        st.info("No runs recorded yet.")
        st.stop()

    f1, f2 = st.columns([2, 1])
    with f1:
        status_filter = st.multiselect(
            "Status", ["success", "error"], default=["success", "error"]
        )
    with f2:
        sort_by = st.selectbox(
            "Sort", ["Newest first", "Oldest first", "Fastest", "Slowest"]
        )

    filtered = [r for r in runs if r["status"] in status_filter]
    if sort_by == "Newest first":
        filtered = sorted(filtered, key=lambda r: r["timestamp"], reverse=True)
    elif sort_by == "Oldest first":
        filtered = sorted(filtered, key=lambda r: r["timestamp"])
    elif sort_by == "Fastest":
        filtered = sorted(filtered, key=lambda r: r["t_total"])
    elif sort_by == "Slowest":
        filtered = sorted(filtered, key=lambda r: r["t_total"], reverse=True)

    if not filtered:
        st.caption("No runs match the current filter.")
        st.stop()

    for r in filtered:
        status_color = "#2a9d6a" if r["status"] == "success" else "#d94f3d"
        dom = max(set(r["emotions"]), key=r["emotions"].count) if r["emotions"] else "-"

        with st.expander(
            f"{r['run_id']}   {r['timestamp'].strftime('%Y-%m-%d %H:%M')}   {r['segments']} segments   {r['t_total']}s"
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-lbl">Status</div><div class="kpi-val" style="font-size:1.2rem;color:{status_color}">{r["status"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                dom_col = EMOTION_COLORS.get(dom, "#1a1a1a")
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-lbl">Dominant emotion</div><div class="kpi-val" style="font-size:1.2rem;color:{dom_col}">{dom}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-lbl">Input type</div><div class="kpi-val" style="font-size:1.2rem">{r["input_type"]}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="section-label">Step trace</div>', unsafe_allow_html=True
            )
            trace_row("Whisper ASR", r["t_whisper"], r["t_total"])
            trace_row("MarianMT translation", r["t_translate"], r["t_total"])
            trace_row("DeBERTa classification", r["t_classify"], r["t_total"])

            st.markdown(
                '<div class="section-label" style="margin-top:1rem;">Emotion distribution</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                emotion_bar_chart(r["emotions"]),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"hist_{r['run_id']}",
            )
