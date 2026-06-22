"""
Azure ML training pipeline for emotion classification.

Defines and submits an end-to-end pipeline:
    preprocess -> train -> evaluate

Uses Kubernetes compute target (lambda-2).

Usage:
    python src/emotion_mlops/azure/pipeline.py [options]

Examples:
    # Default run
    python src/emotion_mlops/azure/pipeline.py

    # Comparison baseline
    python src/emotion_mlops/azure/pipeline.py \
        --epochs 1 --batch-size 4 --max-steps 100 \
        --learning-rate 2e-5 --display-name comparison-baseline

    # Full training
    python src/emotion_mlops/azure/pipeline.py \
        --epochs 3 --batch-size 16 --max-steps -1 \
        --display-name full-training-v1

    # Subset test
    python src/emotion_mlops/azure/pipeline.py \
        --data-asset emotion-train-subset --data-version 1 \
        --max-steps 50 --display-name subset-test
"""

import argparse
import logging
from pathlib import Path

from azure.ai.ml import Input, Output, command, dsl
from azure.ai.ml.constants import AssetTypes

from emotion_mlops.azure.config import (
    get_compute_name,
    get_env_name,
    get_instance_type,
    get_raw_data_asset,
)
from emotion_mlops.azure.connect import get_ml_client

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_FILE = _REPO_ROOT / "logs" / "azure.log"
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

# -- Configuration, loaded from configs/azure_config.yaml ---------------------
COMPUTE_NAME = get_compute_name()
ENV_NAME = get_env_name()
DEFAULT_DATA_ASSET, DEFAULT_DATA_VERSION = get_raw_data_asset()
INSTANCE_TYPE = get_instance_type()


def get_preprocess_component(env: str) -> command:
    """
    Create the data preprocessing pipeline component.

    Reads a raw transcript CSV, cleans it, and splits it into
    train, validation, and test sets.

    Args:
        env: Azure ML environment name and version string.

    Returns:
        Command component for data preprocessing.
    """
    return command(
        name="preprocess_data",
        display_name="Preprocess Data",
        description="Clean and prepare transcript CSV for training",
        inputs={
            "raw_data": Input(
                type=AssetTypes.URI_FILE, description="Raw transcript CSV"
            ),
        },
        outputs={
            "processed_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
        },
        code=str(_REPO_ROOT / "src"),
        command=(
            "python -m emotion_mlops.data.data_preprocessing "
            "--input ${{inputs.raw_data}} "
            "--output ${{outputs.processed_data}}"
        ),
        environment=env,
        compute=COMPUTE_NAME,
        resources={"instance_type": INSTANCE_TYPE},
    )


def get_train_component(
    env: str,
    epochs: int = 1,
    batch_size: int = 4,
    max_steps: int = 100,
    learning_rate: float = 2e-5,
    run_name: str = "azure-run",
) -> command:
    """
    Create the model training pipeline component.

    Fine-tunes DistilBERT on preprocessed emotion classification data.
    Logs per-step and per-epoch metrics to Azure ML Run context.

    Args:
        env: Azure ML environment name and version string.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        max_steps: Max training steps (-1 for full training).
        learning_rate: Learning rate for optimizer.
        run_name: Name tag for this run, visible in Azure ML logs.

    Returns:
        Command component for model training.
    """
    return command(
        name="train_model",
        display_name="Train Model",
        description="Fine-tune DistilBERT on emotion classification data",
        inputs={
            "processed_data": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={
            "model_output": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
        },
        code=str(_REPO_ROOT / "src"),
        command=(
            "python -m emotion_mlops.models.model_training "
            "--data ${{inputs.processed_data}} "
            "--output ${{outputs.model_output}} "
            f"--epochs {epochs} "
            f"--batch-size {batch_size} "
            f"--max-steps {max_steps} "
            f"--learning-rate {learning_rate} "
            f"--run-name {run_name}"
        ),
        environment=env,
        compute=COMPUTE_NAME,
        resources={"instance_type": INSTANCE_TYPE},
    )


def get_evaluate_component(env: str) -> command:
    """
    Create the model evaluation pipeline component.

    Evaluates the trained model on the test set and logs metrics to Azure ML.

    Args:
        env: Azure ML environment name and version string.

    Returns:
        Command component for model evaluation.
    """
    return command(
        name="evaluate_model",
        display_name="Evaluate Model",
        description="Evaluate model F1 and accuracy",
        inputs={
            "model_output": Input(type=AssetTypes.URI_FOLDER),
            "processed_data": Input(type=AssetTypes.URI_FOLDER),
        },
        outputs={
            "eval_output": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
        },
        code=str(_REPO_ROOT / "src"),
        command=(
            "python -m emotion_mlops.models.model_evaluation "
            "--model ${{inputs.model_output}} "
            "--data ${{inputs.processed_data}} "
            "--output ${{outputs.eval_output}} "
            "--min-f1 0.5"
        ),
        environment=env,
        compute=COMPUTE_NAME,
        resources={"instance_type": INSTANCE_TYPE},
    )


def get_merge_component(env: str) -> command:
    """
    Create the feedback-merge pipeline component.

    Concatenates the base training data with the user feedback corrections
    into a single raw CSV. Both inputs are mounted by Azure ML as local files,
    so no local base data is required in CI. Used only when the retrain is
    triggered with feedback corrections.

    Args:
        env: Azure ML environment name and version string.

    Returns:
        Command component that emits the merged raw CSV.
    """
    return command(
        name="merge_feedback",
        display_name="Merge Feedback Corrections",
        description="Concatenate base training data with user feedback corrections",
        inputs={
            "base_data": Input(type=AssetTypes.URI_FILE),
            "corrections": Input(type=AssetTypes.URI_FILE),
        },
        outputs={
            "merged_data": Output(type=AssetTypes.URI_FILE, mode="rw_mount"),
        },
        code=str(_REPO_ROOT / "src"),
        command=(
            "python -m emotion_mlops.feedback.merge_inputs "
            "--base ${{inputs.base_data}} "
            "--corrections ${{inputs.corrections}} "
            "--output ${{outputs.merged_data}}"
        ),
        environment=env,
        compute=COMPUTE_NAME,
        resources={"instance_type": INSTANCE_TYPE},
    )


def build_pipeline_job(
    data_asset_name: str = DEFAULT_DATA_ASSET,
    data_asset_version: str = DEFAULT_DATA_VERSION,
    env_name: str = ENV_NAME,
    compute_name: str = COMPUTE_NAME,
    epochs: int = 1,
    batch_size: int = 4,
    max_steps: int = 100,
    learning_rate: float = 2e-5,
    run_name: str = "azure-run",
    display_name: str = "emotion-training-pipeline",
    corrections_asset: str | None = None,
    corrections_version: str | None = None,
):
    """
    Build and return the pipeline job without submitting.

    Used by schedule.py to attach the pipeline to a schedule,
    and by build_and_submit_pipeline to submit it directly.

    Args:
        data_asset_name: Name of the registered data asset in Azure ML.
        data_asset_version: Version of the data asset to use.
        env_name: Azure ML environment name and version string.
        compute_name: Azure ML compute target name.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        max_steps: Max training steps (-1 for full training).
        learning_rate: Learning rate for optimizer.
        run_name: Name tag for this run.
        display_name: Display name for the pipeline job in Azure ML portal.

    Returns:
        Pipeline job object ready to be submitted/ scheduled.
    """
    preprocess = get_preprocess_component(env_name)  # test
    train = get_train_component(
        env_name, epochs, batch_size, max_steps, learning_rate, run_name
    )
    evaluate = get_evaluate_component(env_name)

    use_merge = corrections_asset is not None and corrections_version is not None

    input_data = Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:{data_asset_name}:{data_asset_version}",
    )

    if use_merge:
        merge = get_merge_component(env_name)

        @dsl.pipeline(
            name="emotion-training-pipeline",
            description="Emotion classification: merge -> preprocess -> train -> evaluate",
            default_compute=compute_name,
        )
        def emotion_pipeline(input_data: Input, corrections_data: Input):
            merge_step = merge(base_data=input_data, corrections=corrections_data)
            preprocess_step = preprocess(raw_data=merge_step.outputs.merged_data)
            train_step = train(processed_data=preprocess_step.outputs.processed_data)
            evaluate_step = evaluate(
                model_output=train_step.outputs.model_output,
                processed_data=preprocess_step.outputs.processed_data,
            )
            return {"eval_output": evaluate_step.outputs.eval_output}

        corrections_data = Input(
            type=AssetTypes.URI_FILE,
            path=f"azureml:{corrections_asset}:{corrections_version}",
        )
        pipeline_job = emotion_pipeline(
            input_data=input_data, corrections_data=corrections_data
        )
    else:

        @dsl.pipeline(
            name="emotion-training-pipeline",
            description="Emotion classification: preprocess -> train -> evaluate",
            default_compute=compute_name,
        )
        def emotion_pipeline(input_data: Input):
            preprocess_step = preprocess(raw_data=input_data)
            train_step = train(processed_data=preprocess_step.outputs.processed_data)
            evaluate_step = evaluate(
                model_output=train_step.outputs.model_output,
                processed_data=preprocess_step.outputs.processed_data,
            )
            return {"eval_output": evaluate_step.outputs.eval_output}

        pipeline_job = emotion_pipeline(input_data=input_data)

    pipeline_job.display_name = display_name
    return pipeline_job


def build_and_submit_pipeline(
    data_asset_name: str = DEFAULT_DATA_ASSET,
    data_asset_version: str = DEFAULT_DATA_VERSION,
    env_name: str = ENV_NAME,
    compute_name: str = COMPUTE_NAME,
    epochs: int = 1,
    batch_size: int = 4,
    max_steps: int = 100,
    learning_rate: float = 2e-5,
    run_name: str = "azure-run",
    display_name: str = "emotion-training-pipeline",
    corrections_asset: str | None = None,
    corrections_version: str | None = None,
):
    """
    Build and submit the emotion classification training pipeline to Azure ML.

    Builds the pipeline graph using build_pipeline_job() and submits it
    as a job on the Kubernetes compute target. The pipeline runs:
        preprocess -> train -> evaluate

    Args:
        data_asset_name: Name of the registered data asset in Azure ML.
        data_asset_version: Version of the data asset to use.
        env_name: Azure ML environment name and version string.
        compute_name: Azure ML compute target name.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        max_steps: Max training steps (-1 for full training).
        learning_rate: Learning rate for optimizer.
        run_name: Name tag for this run.
        display_name: Display name visible in Azure ML portal Jobs tab.

    Raises:
        Exception: If pipeline submission fails.
    """
    logger.info("Connecting to Azure ML workspace...")
    client = get_ml_client()

    logger.info("Building pipeline job: %s", display_name)
    pipeline_job = build_pipeline_job(
        data_asset_name=data_asset_name,
        data_asset_version=data_asset_version,
        env_name=env_name,
        compute_name=compute_name,
        epochs=epochs,
        batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        run_name=run_name,
        display_name=display_name,
        corrections_asset=corrections_asset,
        corrections_version=corrections_version,
    )

    logger.info(
        "Submitting pipeline with data asset: %s v%s%s",
        data_asset_name,
        data_asset_version,
        (
            f" + corrections {corrections_asset}:{corrections_version}"
            if corrections_asset
            else ""
        ),
    )
    submitted = client.jobs.create_or_update(pipeline_job)

    logger.info("Pipeline submitted: %s", submitted.name)
    logger.info("View in Azure ML: %s", submitted.studio_url)
    print(f"\nPipeline submitted: {submitted.name}")
    print(f"View in Azure ML:   {submitted.studio_url}")
    return submitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit emotion training pipeline to Azure ML"
    )
    parser.add_argument(
        "--data-asset",
        default=DEFAULT_DATA_ASSET,
        help=f"Data asset name (default: {DEFAULT_DATA_ASSET})",
    )
    parser.add_argument(
        "--data-version",
        default=DEFAULT_DATA_VERSION,
        help=f"Data asset version (default: {DEFAULT_DATA_VERSION})",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max training steps, -1 for full training (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--run-name",
        default="azure-run",
        help="Name tag for this run (default: azure-run)",
    )
    parser.add_argument(
        "--display-name",
        default="emotion-training-pipeline",
        help="Display name in Azure ML portal (default: emotion-training-pipeline)",
    )
    args = parser.parse_args()

    build_and_submit_pipeline(
        data_asset_name=args.data_asset,
        data_asset_version=args.data_version,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        run_name=args.run_name,
        display_name=args.display_name,
    )
