"""
Model training script for emotion classification.

Fine-tunes DistilBERT on GoEmotions data with class weights
to handle class imbalance and achieve better macro F1.

Supports both pipeline runs (Azure ML Run context) and
sweep runs (MLflow, required for sweep trials table).

Usage:
    python -m emotion_mlops.models.model_training \
        --data data/processed/ \
        --output models/trained/ \
        --epochs 3 \
        --learning-rate 2e-5 \
        --batch-size 16 \
        --max-steps -1
"""

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from emotion_mlops.models.model_saving import save_model

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
MODEL_NAME = "microsoft/deberta-v3-base"


def get_run():
    """
    Get Azure ML Run context if running in Azure ML pipeline, otherwise None.

    Returns:
        Azure ML Run object or None if running locally or in sweep.
    """
    try:
        from azureml.core.run import Run

        run = Run.get_context()
        if hasattr(run, "experiment"):
            return run
    except Exception:
        pass
    return None


def log_metric(run, key: str, value: float) -> None:
    """
    Log a metric to Azure ML.

    MLflow is logged first, required for sweep trials table to show values.
    run.log is added as backup for pipeline context metrics tab.

    Args:
        run: Azure ML Run context or None.
        key: Metric name.
        value: Metric value.
    """
    # MLflow first, only way to show metrics in sweep trials table
    try:
        mlflow.log_metric(key, value)
        logger.debug("Logged %s=%.4f via MLflow", key, value)
    except Exception as e:
        logger.warning("MLflow log failed for %s: %s", key, e)

    # run.log as backup, for pipeline context metrics tab
    if run:
        try:
            run.log(key, value)
        except Exception:
            pass


class AzureMLCallback(TrainerCallback):
    """
    Logs per-step metrics for learning curve visualisation.

    Uses log_metric() which works in both pipeline and sweep context.
    """

    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                if isinstance(v, float):
                    log_metric(self.run, k, v)


class EmotionDataset(Dataset):
    """PyTorch dataset for emotion classification."""

    def __init__(
        self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128
    ):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """
    Custom Trainer that uses class weights to handle class imbalance.

    Computes weighted cross-entropy loss so minority classes (disgust, fear)
    are not ignored in favour of dominant classes (happiness, neutral).
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute weighted cross-entropy loss.

        Args:
            model: The model being trained.
            inputs: Batch inputs including labels.
            return_outputs: Whether to return model outputs alongside loss.

        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs=True.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = CrossEntropyLoss(
            weight=(
                self.class_weights.to(model.device)
                if self.class_weights is not None
                else None
            )
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_steps: int = -1,
    run_name: str = "",
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> None:
    """
    Fine-tune DistilBERT on emotion classification data with class weights.

    Logs metrics via MLflow (sweep) and Azure ML Run context (pipeline).
    Learning curves visible in Azure ML portal -> Jobs -> Metrics tab.

    Args:
        data_dir: Directory containing train.csv and val.csv.
        output_dir: Directory to save trained model.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        batch_size: Training batch size.
        max_steps: Max training steps (-1 = full training).
        run_name: Name tag for this run.
        weight_decay: L2 regularisation strength (default 0.01).
        warmup_ratio: Fraction of training spent warming up LR (default 0.1).
    """
    run = get_run()

    if run:
        run.tag("run_name", run_name or "azure-run")
        logger.info("Azure ML Run context found")
    else:
        logger.info("No pipeline context, sweep or local mode")

    # Log hyperparameters
    log_metric(run, "epochs", epochs)
    log_metric(run, "learning_rate", learning_rate)
    log_metric(run, "batch_size", batch_size)
    log_metric(run, "weight_decay", weight_decay)
    log_metric(run, "warmup_ratio", warmup_ratio)

    logger.info("Loading data from: %s", data_dir)
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    val_df = pd.read_csv(Path(data_dir) / "val.csv")
    logger.info("Train size: %d | Val size: %d", len(train_df), len(val_df))

    # Class weights, fix imbalance.
    # Only weight the emotion classes actually present in this training split.
    # Small feedback batches can be missing a class entirely; passing the full
    # 7-class list to compute_class_weight then raises
    # "classes should have valid labels that are in y". So we compute balanced
    # weights for the classes that ARE present and give any missing class a
    # neutral weight of 1.0, keeping a full length-7 weight vector for the loss.
    all_labels = np.array([LABEL2ID[e] for e in train_df["Emotion"]])
    present_classes = np.unique(all_labels)
    present_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=all_labels,
    )
    weights = np.ones(len(LABEL2ID), dtype=float)
    for cls, w in zip(present_classes, present_weights):
        weights[int(cls)] = w
    class_weights = torch.tensor(weights, dtype=torch.float)
    logger.info("Class weights: %s", dict(zip(LABEL2ID.keys(), weights.round(2))))
    if len(present_classes) < len(LABEL2ID):
        missing = set(LABEL2ID.values()) - set(present_classes.tolist())
        logger.warning(
            "Training data is missing %d emotion class(es): %s "
            "(given neutral weight 1.0)",
            len(missing),
            [ID2LABEL[c] for c in missing],
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | CUDA available: %s", device, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info(
            "GPU: %s | Memory: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    train_dataset = EmotionDataset(
        train_df["Sentence"].tolist(),
        [LABEL2ID[e] for e in train_df["Emotion"]],
        tokenizer,
    )
    val_dataset = EmotionDataset(
        val_df["Sentence"].tolist(),
        [LABEL2ID[e] for e in val_df["Emotion"]],
        tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        report_to="none",
        max_steps=max_steps,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,
        seed=42,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        run_name=run_name or "local-run",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        class_weights=class_weights,
        callbacks=[AzureMLCallback(run)],
    )

    logger.info("Starting training...")
    trainer.train()

    # Compute and log final val metrics
    preds = trainer.predict(val_dataset)
    pred_labels = preds.predictions.argmax(axis=1)
    true_labels = [LABEL2ID[e] for e in val_df["Emotion"]]

    f1 = f1_score(true_labels, pred_labels, average="macro")
    acc = accuracy_score(true_labels, pred_labels)

    log_metric(run, "val_f1", f1)
    log_metric(run, "val_accuracy", acc)
    logger.info("Val F1: %.4f | Val Accuracy: %.4f", f1, acc)

    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_model(trainer.model, tokenizer, output_dir)
    logger.info("Model saved to: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument("--data", required=True, help="Directory with train/val CSVs")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (-1 = full training)",
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="L2 regularisation strength (default: 0.01)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of training spent warming up LR (default: 0.1)",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        run_name=args.run_name,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )
