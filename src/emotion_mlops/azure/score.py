"""
Scoring script for Azure ML Online AND Batch Endpoint.

Online: called with JSON string, returns list of dicts
Batch:  called with list of file paths (mini-batch), returns pd.DataFrame
"""

import glob
import json
import logging
import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL2ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

model = None
tokenizer = None


def init():
    """Load model once on container start."""
    global model, tokenizer
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    subdirs = glob.glob(os.path.join(model_dir, "*/"))
    if subdirs:
        model_dir = subdirs[0]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    logger.info("Model loaded from: %s", model_dir)


def _predict_sentence(sentence: str) -> dict:
    """Run inference on a single sentence."""
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = probs.argmax().item()
    confidence = probs[pred_id].item()

    return {
        "sentence": sentence,
        "emotion": ID2LABEL[pred_id],
        "confidence": round(confidence, 4),
        "all_emotions": {
            ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))
        },
    }


def run(mini_batch):
    """
    Called by Azure ML for both online and batch endpoints.

    Online endpoint:  mini_batch is a JSON string
    Batch endpoint:   mini_batch is a list of file paths (mini-batch of CSVs)

    Returns:
        Online: list of dicts
        Batch:  pd.DataFrame (Azure ML saves this to predictions.csv)
    """
    # -- Online endpoint: JSON string input ------------------------------------
    if isinstance(mini_batch, str):
        try:
            data = json.loads(mini_batch)
        except json.JSONDecodeError as e:
            return [{"error": f"Invalid JSON: {e}"}]

        sentences = data.get("sentences") or (
            [data["sentence"]] if "sentence" in data else None
        )
        if not sentences:
            return [{"error": "Request must contain 'sentence' or 'sentences' key"}]

        return [_predict_sentence(s) for s in sentences]

    results = []
    for file_path in mini_batch:
        try:
            df = pd.read_csv(file_path)

            text_col = next(
                (
                    c
                    for c in ["Sentence", "Text", "Utterance", "text", "sentence"]
                    if c in df.columns
                ),
                None,
            )
            if text_col is None:
                logger.warning("No text column found in %s", file_path)
                continue

            for sentence in df[text_col].dropna():
                result = _predict_sentence(str(sentence))
                result["source_file"] = os.path.basename(file_path)
                results.append(result)

        except Exception as e:
            logger.error("Failed processing %s: %s", file_path, e)
            results.append(
                {
                    "sentence": "",
                    "emotion": "error",
                    "confidence": 0.0,
                    "source_file": str(file_path),
                    "error": str(e),
                }
            )

    return pd.DataFrame(results)
