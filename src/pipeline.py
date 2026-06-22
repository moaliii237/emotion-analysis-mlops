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

import argparse
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List

import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
import whisper
import yaml
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
)

os.environ["PATH"] = (
    os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ["PATH"]
)

# ---------------------------
# Logging Setup
# ---------------------------


def setup_logger(log_file: str = "pipeline.log") -> logging.Logger:
    """Configure and return the module logger with file and console handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers if setup_logger is called more than once
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


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


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio using the ffmpeg binary bundled with imageio_ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        path,
        "-f",
        "s16le",  # raw 16-bit PCM
        "-ac",
        "1",  # mono
        "-ar",
        str(sr),  # 16kHz sample rate
        "pipe:1",  # stream to stdout
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_audio(
    audio_path: str, whisper_model_name: str = "base", device: str = "cpu"
) -> List[Segment]:
    logger.info("Loading Whisper model '%s' on device '%s'", whisper_model_name, device)
    model = whisper.load_model(whisper_model_name, device=device)

    logger.info("Transcribing (English): %s", audio_path)
    audio_array = load_audio(audio_path)  # <- decode with bundled ffmpeg
    audio_tensor = torch.from_numpy(
        audio_array
    )  # <- convert to tensor so Whisper skips load_audio
    result = model.transcribe(
        audio_tensor, verbose=False, language="en"
    )  # <- pass tensor, NOT audio_path

    segments = [
        Segment(
            start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip()
        )
        for seg in result["segments"]
    ]
    logger.info("Transcription complete, %d segments extracted", len(segments))
    return segments


# ---------------------------
# Step 2: Translation (English -> Dutch)
# ---------------------------


def load_mt(model_name: str, device: str):
    """Load MarianMT model for translation."""
    logger.info("Loading MarianMT model '%s' on device '%s'", model_name, device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    logger.info("MarianMT model loaded successfully")
    return tokenizer, model


@torch.no_grad()
def translate_batch(
    sentences: List[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: str,
    bs: int = 16,
) -> List[str]:
    """Translate English sentences to Dutch in batches."""
    logger.info("Translating %d sentences EN->NL (batch size: %d)", len(sentences), bs)
    outputs = []
    for i in tqdm(range(0, len(sentences), bs), desc="[MT] Translating EN->NL"):
        batch = sentences[i : i + bs]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        generated = model.generate(**encoded, max_new_tokens=256)
        decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
        outputs.extend(decoded)
    logger.info("Translation complete, %d sentences translated", len(outputs))
    return outputs


# ---------------------------
# Step 3: Emotion Classification
# ---------------------------


@torch.no_grad()
def predict_emotions(
    texts: List[str], model_dir: str, device: str, max_length: int = 192, bs: int = 16
) -> List[str]:
    """Predict emotion labels for English text."""
    logger.info(
        "Loading local emotion model from '%s' on device '%s'", model_dir, device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    id2label = {i: str(l).lower() for i, l in model.config.id2label.items()}
    logger.info(
        "Emotion model loaded, %d classes: %s", len(id2label), list(id2label.values())
    )

    preds = []
    for i in tqdm(range(0, len(texts), bs), desc="[Emotion] Predicting"):
        batch = texts[i : i + bs]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        top_indices = probs.argmax(dim=-1).cpu().numpy().tolist()
        top_probs = probs.max(dim=-1).values.cpu().numpy().tolist()

        for idx, conf in zip(top_indices, top_probs):
            label = id2label[idx]
            if conf < 0.5:
                logger.warning(
                    "Low confidence (%.2f) for predicted label '%s', result may be unreliable",
                    conf,
                    label,
                )
            preds.append(label)

    logger.info("Emotion prediction complete, %d predictions made", len(preds))
    return preds


# ---------------------------
# Main Pipeline
# ---------------------------


def main():
    logger.info("Pipeline starting")

    parser = argparse.ArgumentParser(
        description="End-to-end NLP pipeline for emotion classification (English->Dutch)."
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--audio", required=False, help="Path to audio/video file (wav/mp3/mp4)"
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--transcript_csv",
        required=False,
        help="Optional: existing transcript CSV (Start, End, Sentence). Skips Whisper if provided.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger.info("Config loaded from '%s'", args.config)

    device = cfg.get("device", "cpu")
    whisper_model = cfg.get("whisper_model", "base")
    mt_model_name = cfg.get("mt_en_nl_model", "Helsinki-NLP/opus-mt-en-nl")
    emotion_model_dir = cfg.get("best_model_dir", "./best_model_v3")
    bs = int(cfg.get("batch_size", 16))
    max_len = int(cfg.get("max_length", 192))

    logger.info(
        "Runtime config, device: %s | whisper: %s | MT model: %s | emotion model: %s | batch: %d | max_len: %d",
        device,
        whisper_model,
        mt_model_name,
        emotion_model_dir,
        bs,
        max_len,
    )

    # 0) Input handling
    if not args.transcript_csv and not args.audio:
        logger.error("Neither --audio nor --transcript_csv was provided. Exiting.")
        raise SystemExit("Error: Either --audio or --transcript_csv must be provided.")

    # Optional: extract audio from video
    if args.audio and args.audio.lower().endswith((".mp4", ".mov", ".mkv")):
        logger.info("Video input detected, extracting audio with ffmpeg")
        audio_path = "./data/audio.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                args.audio,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                audio_path,
            ]
        )
        logger.info("Audio extracted to '%s'", audio_path)
        args.audio = audio_path

    # 1) Transcription
    if args.transcript_csv:
        logger.info(
            "Using existing transcript CSV: '%s' (skipping Whisper)",
            args.transcript_csv,
        )
        df = pd.read_csv(args.transcript_csv)
        col_map = {c.lower(): c for c in df.columns}
        s_col = (
            col_map.get("sentence") or col_map.get("text") or col_map.get("utterance")
        )
        st_col = (
            col_map.get("start")
            or col_map.get("start time")
            or col_map.get("start_time")
        )
        et_col = (
            col_map.get("end") or col_map.get("end time") or col_map.get("end_time")
        )
        if not s_col:
            logger.error("Transcript CSV is missing a recognisable sentence column")
            raise SystemExit("Transcript CSV must contain a 'Sentence' column.")
        if not st_col or not et_col:
            logger.warning(
                "No Start/End time columns found in transcript CSV, filling with zeros"
            )
            df["Start"] = "00:00:00,000"
            df["End"] = "00:00:00,000"
        else:
            df = df.rename(columns={st_col: "Start", et_col: "End"})
        df = df.rename(columns={s_col: "Sentence"})
        logger.info("Transcript loaded, %d rows", len(df))
    else:
        logger.info("Step 1/3, Transcribing audio with Whisper")
        segments = transcribe_audio(
            args.audio, whisper_model_name=whisper_model, device=device
        )
        df = pd.DataFrame(
            {
                "Start": [hhmmss_ms(s.start) for s in segments],
                "End": [hhmmss_ms(s.end) for s in segments],
                "Sentence": [s.text for s in segments],
            }
        )

    # 2) Translation (English -> Dutch)
    logger.info("Step 2/3, Translating English -> Dutch")

    tokenizer, model = load_mt(mt_model_name, device)
    translations = translate_batch(
        df["Sentence"].astype(str).tolist(), tokenizer, model, device, bs=bs
    )
    df["Translation"] = translations

    # 3) Emotion Classification (on English)
    logger.info("Step 3/3, Classifying emotions")

    emotions = predict_emotions(
        df["Sentence"].astype(str).tolist(),
        emotion_model_dir,
        device,
        max_length=max_len,
        bs=bs,
    )
    df["Emotion"] = emotions

    # 4) Save output
    out_cols = ["Start", "End", "Sentence", "Translation", "Emotion"]
    df[out_cols].to_csv(args.out, index=False, encoding="utf-8-sig")

    logger.info("Output saved to '%s' (%d rows)", args.out, len(df))
    logger.info(
        "Preview of first 10 rows:\n%s", df[out_cols].head(10).to_string(index=False)
    )
    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
