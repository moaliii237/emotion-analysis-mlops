"""
EmotionScope CLI, terminal interface for the emotion classifier.

A user-friendly command-line tool built with Typer. Supports single-sentence
inference, full probability distributions, health checks, and the full
audio/CSV pipeline.

Run `python -m emotion_mlops.cli.command --help` to see all commands.

Examples
--------
    # Predict the emotion of a sentence
    emotionscope predict "I am absolutely furious!"

    # Get the full probability distribution
    emotionscope predict-full "I'm so happy today" --verbose

    # Check if the model is loaded
    emotionscope health

    # Run the full pipeline on an audio file
    emotionscope pipeline --audio meeting.mp3 --out results.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

# ---- Logger setup --------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_FILE = _REPO_ROOT / "logs" / "cli.log"
_LOG_FILE.parent.mkdir(exist_ok=True)

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


# ---- Typer app -----------------------------------------------------------
app = typer.Typer(
    name="emotionscope",
    help="EmotionScope, terminal interface for the 7-emotion classifier.",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)


# Lazy classifier loader, avoids loading DeBERTa just to print --help.
_classifier_cache = {}


def _get_classifier(model_dir: Optional[str] = None):
    """Load (and cache) the EmotionClassifier on first use."""
    key = model_dir or "default"
    if key in _classifier_cache:
        return _classifier_cache[key]

    from emotion_mlops.models.classifier import EmotionClassifier

    if model_dir is None:
        model_dir = str(_REPO_ROOT / "models" / "best_model_v3")

    logger.info("[CLI] Loading classifier from %s", model_dir)
    clf = EmotionClassifier(model_path=model_dir, model_version="best_model_v3")
    clf.load()
    _classifier_cache[key] = clf
    return clf


# ==========================================================================
# Command 1, predict
# ==========================================================================
@app.command()
def predict(
    text: str = typer.Argument(..., help="Sentence to classify."),
    model_dir: Optional[str] = typer.Option(
        None,
        "--model-dir",
        "-m",
        help="Override the model directory.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Display extra information about the prediction.",
    ),
):
    """
    Predict the [bold]single most likely emotion[/bold] for a sentence.

    Returns the top emotion and its confidence as readable text. Useful for
    quick smoke checks from the terminal.

    Example
    -------
        emotionscope predict "I am absolutely furious!"
    """
    if not text.strip():
        typer.echo(
            typer.style("Error: input text is empty.", fg=typer.colors.RED, bold=True)
        )
        raise typer.Exit(code=1)

    clf = _get_classifier(model_dir)
    emotion, confidence = clf.predict(text)

    # Map every emotion to a colour for pretty output.
    colour_map = {
        "anger": typer.colors.RED,
        "happiness": typer.colors.GREEN,
        "sadness": typer.colors.BLUE,
        "fear": typer.colors.MAGENTA,
        "surprise": typer.colors.YELLOW,
        "disgust": typer.colors.BRIGHT_RED,
        "neutral": typer.colors.WHITE,
    }
    styled_emotion = typer.style(
        emotion.upper(),
        fg=colour_map.get(emotion, typer.colors.WHITE),
        bold=True,
    )
    styled_conf = typer.style(f"{confidence:.1%}", fg=typer.colors.CYAN, bold=True)

    if verbose:
        typer.echo("-" * 60)
        typer.echo(
            typer.style("EmotionScope - Prediction", fg=typer.colors.CYAN, bold=True)
        )
        typer.echo("-" * 60)
        typer.echo(f"Input    : {text}")
        typer.echo(f"Model    : {clf.model_version}")
        typer.echo(f"Emotion  : {styled_emotion}")
        typer.echo(f"Confidence: {styled_conf}")
        typer.echo("-" * 60)
    else:
        typer.echo(f"{styled_emotion} ({styled_conf})")


# ==========================================================================
# Command 2, predict-full
# ==========================================================================
@app.command("predict-full")
def predict_full(
    text: str = typer.Argument(..., help="Sentence to classify."),
    model_dir: Optional[str] = typer.Option(
        None,
        "--model-dir",
        "-m",
        help="Override the model directory.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output the full distribution as JSON (machine-readable).",
    ),
):
    """
    Show the [bold]full probability distribution[/bold] over all 7 emotions.

    Prints a horizontal bar chart in the terminal, or JSON if `--json` is set.

    Example
    -------
        emotionscope predict-full "I'm so excited today!"
    """
    if not text.strip():
        typer.echo(
            typer.style("Error: input text is empty.", fg=typer.colors.RED, bold=True)
        )
        raise typer.Exit(code=1)

    clf = _get_classifier(model_dir)
    distribution = clf.predict_all(text)

    if json_output:
        # Machine-readable output for scripts and piping.
        top_emotion = max(distribution, key=distribution.get)
        payload = {
            "text": text,
            "top_emotion": top_emotion,
            "top_confidence": round(distribution[top_emotion], 4),
            "all_emotions": {k: round(v, 4) for k, v in distribution.items()},
            "model_version": clf.model_version,
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    # Pretty bar chart, sorted by probability descending.
    typer.echo("-" * 60)
    typer.echo(typer.style(f'Input: "{text}"', fg=typer.colors.CYAN))
    typer.echo("-" * 60)

    sorted_items = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
    max_bar = 40  # max characters in the bar
    for emotion, prob in sorted_items:
        bar_len = int(round(prob * max_bar))
        bar = "#" * bar_len + "-" * (max_bar - bar_len)
        styled_emo = typer.style(f"{emotion:<10}", bold=(emotion == sorted_items[0][0]))
        styled_pct = typer.style(f"{prob:>6.1%}", fg=typer.colors.GREEN)
        typer.echo(f"{styled_emo} {bar} {styled_pct}")
    typer.echo("-" * 60)


# ==========================================================================
# Command 3, health
# ==========================================================================
@app.command()
def health(
    model_dir: Optional[str] = typer.Option(
        None,
        "--model-dir",
        "-m",
        help="Override the model directory.",
    ),
):
    """
    Verify the classifier loads correctly and report its version.

    Useful as a smoke check before running batch jobs or demos.

    Example
    -------
        emotionscope health
    """
    try:
        clf = _get_classifier(model_dir)
        typer.echo(
            typer.style("OK: Model loaded successfully", fg=typer.colors.GREEN, bold=True)
        )
        typer.echo(f"  Version : {clf.model_version}")
        typer.echo(f"  Path    : {clf.model_path}")
    except Exception as e:
        typer.echo(
            typer.style(f"FAILED: Model failed to load: {e}", fg=typer.colors.RED, bold=True)
        )
        raise typer.Exit(code=1)


# ==========================================================================
# Command 4, pipeline (delegates to infer.py)
# ==========================================================================
@app.command()
def pipeline(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to the YAML config file."
    ),
    out: str = typer.Option(..., "--out", "-o", help="Where to write the output CSV."),
    audio: Optional[str] = typer.Option(
        None, "--audio", "-a", help="Path to an audio/video file."
    ),
    transcript_csv: Optional[str] = typer.Option(
        None, "--transcript-csv", "-t", help="Path to a transcript CSV (skips Whisper)."
    ),
):
    """
    Run the [bold]full audio/CSV pipeline[/bold]: transcribe + translate + classify.

    Either `--audio` or `--transcript-csv` must be supplied.

    Example
    -------
        emotionscope pipeline --audio meeting.mp3 --config configs/config.yaml --out results.csv
    """
    if not audio and not transcript_csv:
        typer.echo(
            typer.style(
                "Error: provide either --audio or --transcript-csv",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(code=1)

    # Delegate to the existing argparse-based infer.main by patching sys.argv.
    import sys

    from emotion_mlops.cli.infer import main as infer_main

    argv_backup = sys.argv[:]
    sys.argv = ["infer.py", "--config", config, "--out", out]
    if audio:
        sys.argv += ["--audio", audio]
    if transcript_csv:
        sys.argv += ["--transcript_csv", transcript_csv]

    try:
        infer_main()
    finally:
        sys.argv = argv_backup


# ==========================================================================
# Entry point
# ==========================================================================
if __name__ == "__main__":
    app()
