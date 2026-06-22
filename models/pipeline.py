"""
Complete NLP Pipeline for Emotion Classification (English -> Dutch)
-----------------------------------------------------------------
This script processes raw video/audio input end-to-end:
1. Extracts audio from video (if needed)
2. Transcribes English speech to text (Whisper)
3. Translates English -> Dutch (MarianMT)
4. Classifies emotions using a fine-tuned Transformer model
5. Outputs a CSV file with: Start Time | End Time | Sentence | Translation | Emotion
"""

import logging
import os
import argparse
import yaml
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import whisper
import subprocess
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)

logger = logging.getLogger(__name__)


# ---------------------------
# Logging Setup
# ---------------------------

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configure root logger with console (and optional file) handler."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger.debug("Logging initialised at level %s", log_level.upper())


# ---------------------------
# Utility Functions
# ---------------------------

def load_yaml(path: str):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hhmmss_ms(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS,mmm format."""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h = s // 3600
    s %= 3600
    m = s // 60
    s %= 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class Segment:
    start: float
    end: float
    text: str


# ---------------------------
# Step 1: Speech-to-Text (English)
# ---------------------------

def transcribe_audio(audio_path: str, whisper_model_name: str = "base", device: str = "cpu") -> List[Segment]:
    """Transcribe English speech using Whisper."""
    logger.info("[Whisper] Loading model '%s' on %s", whisper_model_name, device)
    model = whisper.load_model(whisper_model_name, device=device)

    logger.info("[Whisper] Transcribing (English): %s", audio_path)
    result = model.transcribe(audio_path, verbose=False, language="en")

    segments = [
        Segment(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip())
        for seg in result["segments"]
    ]
    logger.info("[Whisper] Produced %d segment(s)", len(segments))
    return segments


# ---------------------------
# Step 2: Translation (English -> Dutch)
# ---------------------------

def load_mt(model_name: str, device: str):
    """Load MarianMT model for translation."""
    logger.info("[MT] Loading MarianMT model '%s' on %s", model_name, device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizer, model


@torch.no_grad()
def translate_batch(sentences: List[str], tokenizer: MarianTokenizer, model: MarianMTModel, device: str, bs: int = 16) -> List[str]:
    """Translate English sentences to Dutch in batches."""
    logger.info("[MT] Translating %d sentence(s) EN->NL (batch_size=%d)", len(sentences), bs)
    outputs = []
    for i in tqdm(range(0, len(sentences), bs), desc="[MT] Translating EN->NL"):
        batch = sentences[i:i + bs]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        generated = model.generate(**encoded, max_new_tokens=256)
        decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
        outputs.extend(decoded)
    logger.debug("[MT] Translation complete, %d output(s) produced", len(outputs))
    return outputs


# ---------------------------
# Step 3: Emotion Classification
# ---------------------------

@torch.no_grad()
def predict_emotions(texts: List[str], model_dir: str, device: str, max_length: int = 192, bs: int = 16) -> List[Tuple[str, float]]:
    """Predict emotion labels and confidence scores for English text."""
    logger.info("[Emotion] Loading local model from '%s' on %s", model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    id2label = {i: str(l).lower() for i, l in model.config.id2label.items()}
    logger.debug("[Emotion] Label map: %s", id2label)

    preds: List[Tuple[str, float]] = []
    for i in tqdm(range(0, len(texts), bs), desc="[Emotion] Predicting"):
        batch = texts[i:i + bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        top = probs.argmax(dim=-1).cpu().numpy().tolist()
        confs = probs.max(dim=-1).values.cpu().numpy().tolist()
        preds.extend([(id2label[t], float(c)) for t, c in zip(top, confs)])

    logger.info("[Emotion] Predicted labels for %d sample(s)", len(preds))
    return preds


# ---------------------------
# Main Pipeline
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-end NLP pipeline for emotion classification (English->Dutch).")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--audio", required=False, help="Path to audio/video file (wav/mp3/mp4)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--transcript_csv", required=False,
                        help="Optional: existing transcript CSV (Start, End, Sentence). Skips Whisper if provided.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity (default: INFO)")
    _DEFAULT_LOG = Path(__file__).resolve().parents[2] / "logs" / "pipeline.log"
    parser.add_argument("--log-file", default=str(_DEFAULT_LOG), help="Path to log file (default: logs/pipeline.log)")
    args = parser.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    cfg = load_yaml(args.config)
    device = cfg.get("device", "cpu")
    whisper_model = cfg.get("whisper_model", "base")
    mt_model_name = cfg.get("mt_en_nl_model", "Helsinki-NLP/opus-mt-en-nl")
    emotion_model_dir = cfg.get("best_model_dir", "./best_model_v3")
    bs = int(cfg.get("batch_size", 16))
    max_len = int(cfg.get("max_length", 192))

    logger.debug("Config loaded: device=%s, whisper=%s, mt=%s, emotion_dir=%s, bs=%d, max_len=%d",
                 device, whisper_model, mt_model_name, emotion_model_dir, bs, max_len)

    # 0) Input validation
    if not args.transcript_csv and not args.audio:
        logger.error("Either --audio or --transcript_csv must be provided.")
        raise SystemExit(1)

    # Optional: extract audio from video
    if args.audio and args.audio.lower().endswith((".mp4", ".mov", ".mkv")):
        logger.info("[Preprocess] Extracting audio from video: %s", args.audio)
        audio_path = "./data/audio.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", args.audio, "-vn", "-ac", "1", "-ar", "16000", audio_path],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.error("[Preprocess] ffmpeg failed:\n%s", result.stderr.decode())
            raise SystemExit(1)
        logger.info("[Preprocess] Audio extracted to %s", audio_path)
        args.audio = audio_path

    # 1) Transcription
    if args.transcript_csv:
        logger.info("[Input] Using existing transcript: %s", args.transcript_csv)
        df = pd.read_csv(args.transcript_csv)
        col_map = {c.lower(): c for c in df.columns}
        s_col = col_map.get("sentence") or col_map.get("text") or col_map.get("utterance")
        st_col = col_map.get("start") or col_map.get("start time") or col_map.get("start_time")
        et_col = col_map.get("end") or col_map.get("end time") or col_map.get("end_time")
        if not s_col:
            logger.error("Transcript CSV must contain a 'Sentence' (or 'text'/'utterance') column.")
            raise SystemExit(1)
        if not st_col or not et_col:
            logger.warning("No Start/End columns found in transcript CSV; defaulting to 00:00:00,000")
            df["Start"] = "00:00:00,000"
            df["End"] = "00:00:00,000"
        else:
            df = df.rename(columns={st_col: "Start", et_col: "End"})
        df = df.rename(columns={s_col: "Sentence"})
        logger.info("[Input] Loaded %d row(s) from transcript CSV", len(df))
    else:
        logger.info("[Step 1] Running Whisper transcription")
        segments = transcribe_audio(args.audio, whisper_model_name=whisper_model, device=device)
        df = pd.DataFrame({
            "Start": [hhmmss_ms(s.start) for s in segments],
            "End":   [hhmmss_ms(s.end)   for s in segments],
            "Sentence": [s.text for s in segments],
        })

    # 2) Translation (English -> Dutch)
    logger.info("[Step 2] Running EN->NL translation")
    tokenizer, model = load_mt(mt_model_name, device)
    translations = translate_batch(df["Sentence"].astype(str).tolist(), tokenizer, model, device, bs=bs)
    df["Translation"] = translations

    # 3) Emotion Classification (on English)
    logger.info("[Step 3] Running emotion classification")
    results = predict_emotions(df["Sentence"].astype(str).tolist(), emotion_model_dir, device,
                               max_length=max_len, bs=bs)
    df["Emotion"] = [label for label, _ in results]
    df["EmotionConfidence"] = [conf for _, conf in results]

    # 4) Save output
    out_cols = ["Start", "End", "Sentence", "Translation", "Emotion", "EmotionConfidence"]
    df[out_cols].to_csv(args.out, index=False, encoding="utf-8-sig")
    logger.info("[Done] Output saved to: %s", args.out)
    logger.debug("Preview:\n%s", df[out_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
