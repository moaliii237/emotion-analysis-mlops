import argparse
import logging
from pathlib import Path

from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import BanditPolicy, Choice

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
INSTANCE_TYPE = "gpu"
DEFAULT_DATA_ASSET, DEFAULT_DATA_VERSION = get_raw_data_asset()


def run_grid_sweep(
    data_asset_name: str = DEFAULT_DATA_ASSET,
    data_asset_version: str = DEFAULT_DATA_VERSION,
    env_name: str = ENV_NAME,
    compute_name: str = COMPUTE_NAME,
    max_concurrent_trials: int = 3,
) -> None:
    """
    Submit a grid-search hyperparameter sweep to Azure ML.

    Uses exhaustive Grid sampling over a focused 4-dimensional search space,
    with Bandit early termination to prune underperforming trials.

    Args:
        data_asset_name: Name of the registered data asset.
        data_asset_version: Version of the data asset to use.
        env_name: Azure ML environment name and version string.
        compute_name: Azure ML compute target name.
        max_concurrent_trials: Maximum parallel trials.
    """
    logger.info("Connecting to Azure ML workspace...")
    client = get_ml_client()

    input_data = Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:{data_asset_name}:{data_asset_version}",
    )

    # Base trial: preprocess + train with hyperparameters from search space.
    # weight_decay flows through model_training.py's --weight-decay CLI arg
    # (added in this branch). Default behaviour preserved when not set.
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
            "--weight-decay ${{search_space.weight_decay}} "
            "--max-steps -1 "
            "--output models/sweep_output/"
        ),
        inputs={"input_data": input_data},
        environment=env_name,
        compute=compute_name,
        resources={"instance_type": INSTANCE_TYPE},
        display_name="deberta-lr-sweep",
    )

    # Wide grid around the previous winner (lr=2e-5, bs=8, ep=3, wd=0.05, f1=0.5243).
    # Skips lr=1e-5 (failed in first sweep). Caps bs at 16 to avoid OOM on DeBERTa-v3-base.
    # 3 * 2 * 2 * 2 = 24 combinations.
    sweep_job = base_job.sweep(
        sampling_algorithm="grid",
        primary_metric="val_f1",
        goal="maximize",
        search_space={
            "learning_rate": Choice([2e-5, 3e-5, 5e-5]),
            "batch_size": Choice([8, 16]),
            "epochs": Choice([3, 4]),
            "weight_decay": Choice([0.01, 0.05]),
        },
        max_total_trials=24,
        max_concurrent_trials=4,
        early_termination_policy=BanditPolicy(
            slack_factor=0.2,
            evaluation_interval=1,
            delay_evaluation=2,
        ),
    )

    sweep_job.display_name = "emotion-grid-sweep-wide"
    sweep_job.description = (
        "Wide grid sweep over DeBERTa-v3-base, refining first sweep winner "
        "(lr=2e-5, bs=8, ep=3, wd=0.05, f1=0.5243). "
        "Space: lr in {2e-5, 3e-5, 5e-5} x bs in {8, 16} x ep in {3, 4} x wd in {0.01, 0.05}. "
        "24 trials, 4 concurrent, Bandit early termination (slack=0.2)."
    )

    logger.info(
        "Submitting wide grid sweep -- 24 trials, %d concurrent, Bandit early termination",
        max_concurrent_trials,
    )
    submitted = client.jobs.create_or_update(sweep_job)

    logger.info("Sweep submitted: %s", submitted.name)
    logger.info("View in Azure ML: %s", submitted.studio_url)
    print(f"\nSweep submitted: {submitted.name}")
    print(f"View in Azure ML: {submitted.studio_url}")
    print(
        "\nTo register the winner once it finishes:\n"
        f"  python src/emotion_mlops/azure/register_best.py --sweep-job {submitted.name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a Grid-search hyperparameter sweep to Azure ML"
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
        "--max-concurrent",
        type=int,
        default=4,
        help="Parallel trials (default: 4)",
    )
    args = parser.parse_args()

    run_grid_sweep(
        data_asset_name=args.data_asset,
        data_asset_version=args.data_version,
        max_concurrent_trials=args.max_concurrent,
    )
