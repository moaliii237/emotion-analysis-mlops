import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_FILE = _REPO_ROOT / "logs" / "pipeline.log"
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


def hhmmss_ms(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="Emotion classification pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--transcript_csv", required=False)
    args = parser.parse_args()

    from emotion_mlops.models.classifier import predict_emotions
    from emotion_mlops.models.transcriber import transcribe_audio
    from emotion_mlops.models.translator import load_mt, translate_batch

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cpu")
    whisper_model = cfg.get("whisper_model", "base")
    mt_model = cfg.get("mt_en_nl_model", "Helsinki-NLP/opus-mt-en-nl")
    model_dir = cfg.get("best_model_dir")
    bs = int(cfg.get("batch_size", 16))
    max_len = int(cfg.get("max_length", 192))

    if not args.transcript_csv and not args.audio:
        raise SystemExit("Provide either --audio or --transcript_csv")

    if args.audio and args.audio.lower().endswith((".mp4", ".mov", ".mkv")):
        audio_path = "data/audio.wav"
        logger.info("[CLI] Converting video to audio: %s", args.audio)
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
        args.audio = audio_path

    if args.transcript_csv:
        logger.info("[CLI] Loading transcript CSV: %s", args.transcript_csv)
        df = pd.read_csv(args.transcript_csv)
    else:
        logger.info("[CLI] Transcribing audio: %s", args.audio)
        segments = transcribe_audio(args.audio, whisper_model, device)
        df = pd.DataFrame(
            {
                "Start": [hhmmss_ms(s.start) for s in segments],
                "End": [hhmmss_ms(s.end) for s in segments],
                "Sentence": [s.text for s in segments],
            }
        )
        logger.info("[CLI] Transcription complete: %d segments", len(df))

    logger.info("[CLI] Translating %d segments...", len(df))
    tokenizer, mt = load_mt(mt_model, device)
    df["Translation"] = translate_batch(
        df["Sentence"].tolist(), tokenizer, mt, device, bs
    )

    logger.info("[CLI] Classifying emotions...")
    df["Emotion"] = predict_emotions(
        df["Sentence"].tolist(), model_dir, device, max_len, bs
    )

    df[["Start", "End", "Sentence", "Translation", "Emotion"]].to_csv(
        args.out, index=False, encoding="utf-8-sig"
    )
    logger.info("[CLI] Saved results to: %s", args.out)
    print(df[["Start", "End", "Sentence", "Translation", "Emotion"]].head(10))


if __name__ == "__main__":
    main()
