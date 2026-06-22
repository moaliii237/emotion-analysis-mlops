"""
Register the winning model from an Azure ML hyperparameter sweep.

Reads the best child trial from a completed sweep job (selected by Azure ML
automatically based on the sweep's primary_metric) and registers that
trial's model output as a versioned model in the Azure ML workspace.

This fills the missing model-registration step in the MLOps pipeline:
sweeps were running, but nothing was registering their winners, so models
were not version-controlled and could not be referenced by deployment.

Usage:
    # Latest sweep job
    python src/emotion_mlops/azure/register_best.py

    # A specific sweep job by name
    python src/emotion_mlops/azure/register_best.py --sweep-job <sweep_job_name>

    # Override model name / version
    python src/emotion_mlops/azure/register_best.py \\
        --sweep-job <name> --model-name emotion-classifier --version 1
"""

import argparse
import logging
from pathlib import Path

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

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

DEFAULT_MODEL_NAME = "emotion-classifier"


def find_latest_sweep_job(client) -> str:
    """
    Return the name of the most recently submitted sweep job in the workspace.

    Used when the caller does not specify --sweep-job.

    Args:
        client: Azure ML MLClient.

    Returns:
        The name of the latest sweep job.

    Raises:
        RuntimeError: If no sweep jobs are found.
    """
    logger.info("Searching for the latest sweep job in the workspace...")
    jobs = list(client.jobs.list())
    sweep_jobs = [j for j in jobs if getattr(j, "type", "") == "sweep"]
    if not sweep_jobs:
        raise RuntimeError("No sweep jobs found in the workspace.")
    # MLClient returns jobs in reverse-creation order; first match is newest.
    latest = sweep_jobs[0]
    logger.info("Latest sweep job: %s (status=%s)", latest.name, latest.status)
    return latest.name


def get_best_child_run_id(client, sweep_job_name: str) -> str:
    """
    Read the best child trial of a completed sweep.

    Azure ML automatically tracks which child achieved the best primary_metric
    in the sweep's properties under "best_child_run_id".

    Args:
        client: Azure ML MLClient.
        sweep_job_name: Name of the sweep parent job.

    Returns:
        The run id of the best child trial.

    Raises:
        RuntimeError: If the sweep is not finished or has no best child.
    """
    logger.info("Loading sweep job: %s", sweep_job_name)
    sweep = client.jobs.get(sweep_job_name)

    status = getattr(sweep, "status", "Unknown")
    if status not in ("Completed", "Finished"):
        raise RuntimeError(
            f"Sweep '{sweep_job_name}' is not finished (status={status}). "
            "Wait for it to complete before registering a model."
        )

    properties = getattr(sweep, "properties", {}) or {}
    best_child = properties.get("best_child_run_id")
    if not best_child:
        raise RuntimeError(
            f"Sweep '{sweep_job_name}' has no best_child_run_id in its properties. "
            f"Available properties: {list(properties.keys())}"
        )

    logger.info("Best child run: %s", best_child)
    return best_child


def register_model(
    client,
    best_child_run_id: str,
    model_name: str,
    version: str | None = None,
    description: str = "",
) -> Model:
    """
    Register the model produced by a sweep child run as a workspace model.

    The training trial saves its model output to the `model_output` URI
    folder. We point a new versioned Model asset at that path so it shows
    up in Azure ML Studio -> Models and can be deployed to an endpoint.

    Args:
        client: Azure ML MLClient.
        best_child_run_id: Run id of the winning trial.
        model_name: Name for the registered model (e.g. "emotion-classifier").
        version: Optional version string; Azure auto-assigns if not provided.
        description: Optional description shown in the portal.

    Returns:
        The registered Model asset.
    """
    # Use Azure ML's job-output URI scheme so the asset is linked back to the
    # sweep child for full lineage in the portal.
    model_uri = (
        f"azureml://jobs/{best_child_run_id}/outputs/artifacts/paths/sweep_output/"
    )

    model = Model(
        path=model_uri,
        name=model_name,
        version=version,
        description=description or f"Best model from sweep child {best_child_run_id}",
        type=AssetTypes.CUSTOM_MODEL,
    )

    logger.info("Registering model: %s (version=%s)", model_name, version or "auto")
    registered = client.models.create_or_update(model)

    logger.info(
        "Registered: %s:%s -- view in Azure ML Studio -> Models",
        registered.name,
        registered.version,
    )
    return registered


def main() -> None:
    """Find the best sweep child and register its model in Azure ML."""
    parser = argparse.ArgumentParser(
        description="Register the winning model from an Azure ML sweep."
    )
    parser.add_argument(
        "--sweep-job",
        default=None,
        help="Name of the sweep job. Defaults to the latest sweep in the workspace.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Registered model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Model version (default: Azure auto-assigns).",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Optional description shown in the Models tab.",
    )
    args = parser.parse_args()

    client = get_ml_client()

    sweep_name = args.sweep_job or find_latest_sweep_job(client)
    best_child = get_best_child_run_id(client, sweep_name)
    registered = register_model(
        client,
        best_child_run_id=best_child,
        model_name=args.model_name,
        version=args.version,
        description=args.description,
    )

    print("\nModel registered successfully")
    print(f"  Name:    {registered.name}")
    print(f"  Version: {registered.version}")
    print(f"  Sweep:   {sweep_name}")
    print(f"  Child:   {best_child}")


if __name__ == "__main__":
    main()
