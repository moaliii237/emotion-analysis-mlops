"""
Model evaluation script for emotion classification.

Evaluates a trained model on test data and logs metrics to Azure ML.
Registers the model in Azure ML if F1 exceeds threshold.

Usage:
    python -m emotion_mlops.models.model_evaluation \
        --model models/trained/ \
        --data data/processed/ \
        --output data/eval/ \
        --min-f1 0.30
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def get_run():
    """
    Get Azure ML Run context if running in Azure ML, otherwise return None.

    Returns:
        Azure ML Run object or None if running locally.
    """
    try:
        from azureml.core.run import Run

        run = Run.get_context()
        if hasattr(run, "experiment"):
            return run
    except Exception:
        pass
    return None


def register_model(model_dir: str, f1: float) -> None:
    try:
        import os

        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.identity import ManagedIdentityCredential

        # Azure ML automatically injects these env vars in every job
        subscription_id = os.environ["AZUREML_ARM_SUBSCRIPTION"]
        resource_group = os.environ["AZUREML_ARM_RESOURCEGROUP"]
        workspace_name = os.environ["AZUREML_ARM_WORKSPACE_NAME"]

        client = MLClient(
            credential=ManagedIdentityCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        model = Model(
            path=model_dir,
            name="emotion-classifier",
            description=f"DistilBERT emotion classifier, test_f1={f1:.4f}",
            type="custom_model",
            tags={"test_f1": str(round(f1, 4)), "framework": "distilbert"},
        )
        registered = client.models.create_or_update(model)
        logger.info(
            "Model registered: %s v%s (test_f1=%.4f)",
            registered.name,
            registered.version,
            f1,
        )
    except Exception as e:
        logger.warning("Model registration failed: %s", e)


def evaluate(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    min_f1: float = 0.30,
) -> float:
    """
    Evaluate trained model on test set and log metrics to Azure ML.

    Logs test F1 and accuracy to Azure ML Run context (visible in
    Jobs -> Metrics tab). Saves full classification report to output_dir.
    Registers the model in Azure ML if F1 exceeds threshold.

    Args:
        model_dir: Path to saved model directory.
        data_dir: Directory containing test.csv.
        output_dir: Directory to save evaluation results.
        min_f1: Minimum F1 score threshold for model registration.

    Returns:
        Macro F1 score on test set.

    Raises:
        SystemExit: If F1 score is below min_f1 threshold.
    """
    run = get_run()

    if run:
        logger.info("Azure ML Run context found, logging to portal")
    else:
        logger.info("Running locally, metrics logged to console only")

    logger.info("Loading model from: %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    logger.info("Loading test data from: %s", data_dir)
    test_df = pd.read_csv(Path(data_dir) / "test.csv")
    logger.info("Test size: %d", len(test_df))

    # Run predictions
    predictions = []
    for text in test_df["Sentence"].tolist():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_id = outputs.logits.argmax(dim=1).item()
        predictions.append(ID2LABEL[pred_id])

    true_labels = test_df["Emotion"].tolist()

    # Metrics
    f1 = f1_score(
        [LABEL2ID[e] for e in true_labels],
        [LABEL2ID[e] for e in predictions],
        average="macro",
    )
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)

    # Log to Azure ML
    if run:
        run.log("test_f1", f1)
        run.log("test_accuracy", acc)

    logger.info("Test F1: %.4f | Test Accuracy: %.4f", f1, acc)
    logger.info("\n%s", report)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {"f1": f1, "accuracy": acc, "report": report}
    with open(Path(output_dir) / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results saved to: %s", output_dir)

    # Gate, register if above threshold, fail pipeline if below
    if f1 < min_f1:
        logger.warning(
            "F1 %.4f below threshold %.2f, model will not be registered",
            f1,
            min_f1,
        )
        raise SystemExit(1)

    logger.info(
        "F1 %.4f above threshold %.2f, registering model in Azure ML",
        f1,
        min_f1,
    )
    register_model(model_dir, f1)
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate emotion classifier")
    parser.add_argument(
        "--model", required=True, help="Path to trained model directory"
    )
    parser.add_argument("--data", required=True, help="Directory with test.csv")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--min-f1", type=float, default=0.30)
    args = parser.parse_args()

    evaluate(
        model_dir=args.model,
        data_dir=args.data,
        output_dir=args.output,
        min_f1=args.min_f1,
    )
