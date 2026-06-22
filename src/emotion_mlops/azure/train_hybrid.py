"""
Hybrid-config single training run on Azure ML.

Combines two independent sweep findings:
- Zofia's `teal_ghost_pps4b764m6` random sweep best: lr=5e-5, batch=32, epochs=2
  (with weight_decay hardcoded at 0.01, not searched)
- Ali's `kind_lunch_04qk79cnbm` grid sweep best: weight_decay=0.05
  (independently discovered as the better point on the wd axis)

Neither sweep tested the combination. This script submits a single training job
with that exact untested config, on the same `emotion-raw v1` subset both
previous sweeps used, so the comparison stays apples-to-apples.

If F1 > max(0.5172, 0.5121), it's a genuine "combined" Sprint 3 win.

Usage:
    python src/emotion_mlops/azure/train_hybrid.py
"""

import argparse
import logging
from pathlib import Path

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes

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


def submit_hybrid_training(
    data_asset_name: str = DEFAULT_DATA_ASSET,
    data_asset_version: str = DEFAULT_DATA_VERSION,
    learning_rate: float = 5e-5,
    batch_size: int = 32,
    epochs: int = 2,
    weight_decay: float = 0.05,
    warmup_ratio: float = 0.1,
    display_name: str = "emotion-hybrid-train",
) -> None:
    """
    Submit a single training run with the hybrid config.

    Defaults to lr=5e-5, batch=32, epochs=2, weight_decay=0.05 -- the combination
    that takes the structural winners from each previous sweep.

    Args:
        data_asset_name: Registered data asset name.
        data_asset_version: Data asset version.
        learning_rate: Learning rate.
        batch_size: Batch size.
        epochs: Number of epochs.
        weight_decay: L2 regularisation strength.
        warmup_ratio: LR warmup fraction.
        display_name: Display name for the Azure ML job.
    """
    logger.info("Connecting to Azure ML workspace...")
    client = get_ml_client()

    input_data = Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:{data_asset_name}:{data_asset_version}",
    )

    job = command(
        code=str(_REPO_ROOT / "src"),
        command=(
            "python -m emotion_mlops.data.data_preprocessing "
            "--input ${{inputs.input_data}} "
            "--output data/processed/ && "
            "python -m emotion_mlops.models.model_training "
            "--data data/processed/ "
            f"--learning-rate {learning_rate} "
            f"--batch-size {batch_size} "
            f"--epochs {epochs} "
            f"--weight-decay {weight_decay} "
            f"--warmup-ratio {warmup_ratio} "
            "--max-steps -1 "
            "--run-name hybrid-train "
            "--output ${{outputs.model_output}}"
        ),
        inputs={"input_data": input_data},
        outputs={
            # Declare the trained model as a typed Output so Azure ML persists
            # the folder to workspace storage and we can register it later.
            "model_output": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
        },
        environment=ENV_NAME,
        compute=COMPUTE_NAME,
        resources={"instance_type": INSTANCE_TYPE},
        display_name=display_name,
        description=(
            "Hybrid config: Zofia's best structural params (lr=5e-5, bs=32, ep=2) "
            "+ Ali's best weight_decay (0.05). Same emotion-raw v1 data. "
            "Tests whether combining the two sweep findings beats either alone. "
            "Model artifact is persisted to the workspace via model_output."
        ),
    )

    logger.info(
        "Submitting hybrid training job (lr=%s, bs=%d, ep=%d, wd=%s)...",
        learning_rate,
        batch_size,
        epochs,
        weight_decay,
    )
    submitted = client.jobs.create_or_update(job)
    logger.info("Submitted: %s", submitted.name)
    logger.info("Studio URL: %s", submitted.studio_url)

    print(f"\nHybrid training submitted: {submitted.name}")
    print(f"View in Azure ML:           {submitted.studio_url}")
    print()
    print("Expected run time: ~3-5 minutes on the lambda-2 GPU.")
    print()
    print("To fetch the result once it finishes:")
    print(
        f'  python -c "import mlflow; from emotion_mlops.azure.connect import get_ml_client; '
        f"c=get_ml_client(); ws=c.workspaces.get(c.workspace_name); "
        f"mlflow.set_tracking_uri(ws.mlflow_tracking_uri); "
        f"mc=mlflow.tracking.MlflowClient(); "
        f"print(mc.get_run('{submitted.name}').data.metrics)\""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a single hybrid-config training run to Azure ML"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
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
        "--display-name",
        default="emotion-hybrid-train",
        help="Display name shown in Azure ML Studio Jobs tab",
    )
    args = parser.parse_args()

    submit_hybrid_training(
        data_asset_name=args.data_asset,
        data_asset_version=args.data_version,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        display_name=args.display_name,
    )
