"""
Hyperparameter tuning sweep job for emotion classification.

Runs model_training.py multiple times with different hyperparameter
combinations and selects the best model based on validation F1 score.

Previous results:
- 3 epochs, lr=2e-5, bs=16 -> val_f1=0.51 (best so far)
- 5 epochs -> overfitting (val_f1=0.49)
- GPU now available, faster training

Usage:
    python src/emotion_mlops/azure/sweep.py
    python src/emotion_mlops/azure/sweep.py --max-trials 6 --max-concurrent 2
    python src/emotion_mlops/azure/sweep.py --data-asset emotion-raw --data-version 1
"""

import argparse
import logging
from pathlib import Path

from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import Choice

from emotion_mlops.azure.config import (
    get_compute_name,
    get_env_name,
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

COMPUTE_NAME = get_compute_name()
ENV_NAME = get_env_name()
INSTANCE_TYPE = "gpu"  # Use GPU for faster sweep trials
DEFAULT_DATA_ASSET, DEFAULT_DATA_VERSION = get_raw_data_asset()


def run_sweep(
    data_asset_name: str = DEFAULT_DATA_ASSET,
    data_asset_version: str = DEFAULT_DATA_VERSION,
    env_name: str = ENV_NAME,
    compute_name: str = COMPUTE_NAME,
    max_total_trials: int = 8,
    max_concurrent_trials: int = 2,
) -> None:
    """
    Submit a hyperparameter sweep job to Azure ML using GPU compute.

    Searches over learning rate, batch size, epochs, and weight decay
    to find the best model based on validation F1 score.

    Search space is informed by previous results:
    - 3 epochs with lr=2e-5, bs=16 gave val_f1=0.51
    - 5 epochs caused overfitting
    - GPU allows larger batch sizes and more trials

    Args:
        data_asset_name: Name of the registered data asset.
        data_asset_version: Version of the data asset to use.
        env_name: Azure ML environment name and version string.
        compute_name: Azure ML compute target name.
        max_total_trials: Total number of trials to run.
        max_concurrent_trials: Maximum parallel trials.
    """
    logger.info("Connecting to Azure ML workspace...")
    client = get_ml_client()

    input_data = Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:{data_asset_name}:{data_asset_version}",
    )

    # Base training job, GPU instance, full training
    base_job = command(
        code=str(_REPO_ROOT / "src"),
        command=(
            "python -m emotion_mlops.data.data_preprocessing "
            "--input ${{inputs.input_data}} "
            "--output data/processed/ && "
            "python -m emotion_mlops.models.model_training "
            "--data data/processed/ "
            "--learning-rate ${{search_space.learning_rate}} "
            "--batch-size ${{search_space.batch_size}} "
            "--epochs ${{search_space.epochs}} "
            "--max-steps -1 "
            "--output models/sweep_output/"
        ),
        inputs={"input_data": input_data},
        environment=env_name,
        compute=compute_name,
        resources={"instance_type": INSTANCE_TYPE},
        display_name="emotion-sweep-trial-gpu",
    )

    # Wide search space, GPU can handle larger batches and more epochs
    sweep_job = base_job.sweep(
        sampling_algorithm="random",
        primary_metric="val_f1",
        goal="maximize",
        search_space={
            "learning_rate": Choice([5e-6, 1e-5, 2e-5, 3e-5, 5e-5]),
            "batch_size": Choice([16, 32, 64]),
            "epochs": Choice([2, 3, 4]),
        },
        max_total_trials=max_total_trials,
        max_concurrent_trials=max_concurrent_trials,
    )

    sweep_job.display_name = "emotion-hyperparameter-sweep-gpu"
    sweep_job.description = (
        "GPU sweep, lr=[5e-6..5e-5], batch=[16,32,64], epochs=[2,3,4]. "
        "Full training. Based on best: lr=2e-5, bs=16, epochs=3 -> val_f1=0.51."
    )

    logger.info(
        "Submitting GPU sweep, %d trials, %d concurrent",
        max_total_trials,
        max_concurrent_trials,
    )
    submitted = client.jobs.create_or_update(sweep_job)

    logger.info("Sweep submitted: %s", submitted.name)
    logger.info("View in Azure ML: %s", submitted.studio_url)
    print(f"\nSweep submitted: {submitted.name}")
    print(f"View in Azure ML: {submitted.studio_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit GPU hyperparameter sweep to Azure ML"
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
        "--max-trials",
        type=int,
        default=8,
        help="Total number of trials (default: 8)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Parallel trials (default: 2)",
    )
    parser.add_argument(
        "--instance-type",
        default="gpu",
        choices=["gpu", "cpu-large", "cpu-xl"],
        help="Kubernetes instance type (default: gpu)",
    )
    args = parser.parse_args()

    # Override instance type if specified
    if args.instance_type != "gpu":
        INSTANCE_TYPE = args.instance_type

    run_sweep(
        data_asset_name=args.data_asset,
        data_asset_version=args.data_version,
        max_total_trials=args.max_trials,
        max_concurrent_trials=args.max_concurrent,
    )
