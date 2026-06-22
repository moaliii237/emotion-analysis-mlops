import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def safe_name(value: str) -> str:
    """Convert labels into MLflow-safe metric names."""
    return str(value).replace(" ", "_").replace("/", "_").lower()


def run_tracking_test(
    dataset_path: str = "data/processed_subset/train.csv",
    output_dir: str = "outputs/tracking",
    experiment_name: str = "emotion-mlops-tracking-test",
    run_name: str = "majority-baseline-validation",
) -> None:
    """
    Run an MLflow tracking experiment for the EmotionScope emotion-classification pipeline.

    This experiment validates automated tracking of:
    - parameters,
    - validation metrics,
    - artifacts,
    - visualisations,
    - reproducible run metadata.
    """

    # Close any previously active run. This prevents "run already active" errors
    # when the script is re-run from VS Code or an interactive session.
    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.set_experiment(experiment_name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    if "Emotion" not in df.columns:
        raise ValueError("Dataset must contain an 'Emotion' column.")

    y = df["Emotion"].astype(str)
    X = df.drop(columns=["Emotion"])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    labels = sorted(y.unique())

    model = DummyClassifier(strategy="most_frequent")

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    runtime_seconds = time.perf_counter() - start_time
    runtime_ms = runtime_seconds * 1000

    majority_class = model.classes_[model.class_prior_.argmax()]

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_val, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_val, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_val, y_pred, labels=labels)

    # --------------------
    # Save artifacts
    # --------------------

    confusion_matrix_csv = output_path / "confusion_matrix.csv"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(confusion_matrix_csv)

    classification_report_csv = output_path / "classification_report.csv"
    report = classification_report(
        y_val,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(classification_report_csv)

    dataset_preview_csv = output_path / "dataset_preview.csv"
    df.head(20).to_csv(dataset_preview_csv, index=False)

    train_distribution_csv = output_path / "train_emotion_distribution.csv"
    y_train.value_counts().to_csv(train_distribution_csv)

    validation_distribution_csv = output_path / "validation_emotion_distribution.csv"
    y_val.value_counts().to_csv(validation_distribution_csv)

    prediction_distribution_csv = output_path / "prediction_distribution.csv"
    pd.Series(y_pred).value_counts().to_csv(prediction_distribution_csv)

    # --------------------
    # Save visualisations
    # --------------------

    metrics_plot = output_path / "validation_metrics.png"
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["accuracy", "macro_f1", "weighted_f1", "precision", "recall"],
        [accuracy, macro_f1, weighted_f1, macro_precision, macro_recall],
    )
    plt.ylim(0, 1)
    plt.title("Validation Metrics - Majority Baseline")
    plt.ylabel("Score")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(metrics_plot)
    plt.close()

    confusion_matrix_plot = output_path / "confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix - Validation Set")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(confusion_matrix_plot)
    plt.close()

    distribution_plot = output_path / "emotion_distribution.png"
    plt.figure(figsize=(8, 5))
    y_val.value_counts().reindex(labels).plot(kind="bar")
    plt.title("Validation Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(distribution_plot)
    plt.close()

    # --------------------
    # MLflow tracking
    # --------------------

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("project", "EmotionScope")
        mlflow.set_tag("pipeline_stage", "experiment_tracking")
        mlflow.set_tag("task", "single-label emotion classification")
        mlflow.set_tag("model_type", "majority-class baseline")
        mlflow.set_tag(
            "mlflow.note.content",
            "Validation-based MLflow tracking run for the EmotionScope MLOps pipeline. "
            "This run logs parameters, validation metrics, artifacts, and visualisations "
            "for a majority-class baseline.",
        )

        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("target_column", "Emotion")
        mlflow.log_param("model_family_reference", "DeBERTa-v3-large")
        mlflow.log_param("baseline_model", "DummyClassifier")
        mlflow.log_param("baseline_strategy", "most_frequent")
        mlflow.log_param("majority_class", majority_class)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("stratified_split", True)
        mlflow.log_param("num_rows_total", len(df))
        mlflow.log_param("num_rows_train", len(y_train))
        mlflow.log_param("num_rows_validation", len(y_val))
        mlflow.log_param("num_classes", y.nunique())
        mlflow.log_param("classes", ", ".join(labels))
        mlflow.log_param("asr_model_reference", "Whisper")
        mlflow.log_param("translation_model_reference", "opus-mt-en-nl")

        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_macro_f1", macro_f1)
        mlflow.log_metric("val_weighted_f1", weighted_f1)
        mlflow.log_metric("val_macro_precision", macro_precision)
        mlflow.log_metric("val_macro_recall", macro_recall)
        mlflow.log_metric("runtime_seconds", runtime_seconds)
        mlflow.log_metric("runtime_ms", runtime_ms)
        mlflow.log_metric("rows_processed", len(y_val))
        mlflow.log_metric("unique_emotions", y.nunique())

        per_class_f1 = f1_score(
            y_val,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        for label, score in zip(labels, per_class_f1):
            mlflow.log_metric(f"val_f1_{safe_name(label)}", score)

        for emotion, count in y_val.value_counts().items():
            mlflow.log_metric(f"val_count_{safe_name(emotion)}", int(count))

        mlflow.log_artifact(str(confusion_matrix_csv))
        mlflow.log_artifact(str(classification_report_csv))
        mlflow.log_artifact(str(dataset_preview_csv))
        mlflow.log_artifact(str(train_distribution_csv))
        mlflow.log_artifact(str(validation_distribution_csv))
        mlflow.log_artifact(str(prediction_distribution_csv))
        mlflow.log_artifact(str(metrics_plot))
        mlflow.log_artifact(str(confusion_matrix_plot))
        mlflow.log_artifact(str(distribution_plot))

        print(f"Logged run: {run_name}")
        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation macro F1: {macro_f1:.4f}")
        print(f"Validation weighted F1: {weighted_f1:.4f}")
        print(f"Majority class: {majority_class}")
        print(f"Runtime: {runtime_ms:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLflow experiment tracking for the EmotionScope baseline model."
    )
    parser.add_argument(
        "--dataset",
        default="data/processed_subset/train.csv",
        help="Path to dataset CSV containing an Emotion column.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tracking",
        help="Directory where tracking artifacts are saved.",
    )
    parser.add_argument(
        "--experiment-name",
        default="emotion-mlops-tracking-test",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        default="majority-baseline-validation",
        help="MLflow run name.",
    )

    args = parser.parse_args()

    run_tracking_test(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
