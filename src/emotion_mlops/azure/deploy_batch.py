"""
Deploy registered emotion-classifier model as Azure ML Batch Endpoint.

Batch endpoints process large datasets asynchronously, useful for
analysing full TV episode transcripts without waiting for real-time responses.

Workflow:
    1. Create batch endpoint and deployment
    2. Submit batch job on a data asset or local CSV
    3. Download results from blob storage when job completes

Usage:
    # Create batch endpoint and deployment
    python src/emotion_mlops/azure/deploy_batch.py

    # Run batch inference on a data asset
    python src/emotion_mlops/azure/deploy_batch.py \
        --invoke --data-asset emotion-test --data-version 1

    # Run on a local CSV file
    python src/emotion_mlops/azure/deploy_batch.py \
        --invoke --input-file data/processed/test.csv

    # Submit and do not wait for results
    python src/emotion_mlops/azure/deploy_batch.py \
        --invoke --data-asset emotion-test --no-wait
"""

import argparse
import logging
import time
from pathlib import Path

from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.entities import (
    BatchDeployment,
    BatchEndpoint,
    BatchRetrySettings,
    CodeConfiguration,
)

from emotion_mlops.azure.config import (
    get_batch_endpoint_name,
    get_compute_name,
    get_env_name,
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
BATCH_ENDPOINT_NAME = get_batch_endpoint_name()
COMPUTE_NAME = get_compute_name()
ENV_NAME = get_env_name()
MODEL_NAME = "emotion-classifier"
DEPLOYMENT_NAME = "emotion-batch-deployment"
# ------------------------------------------------------------------------------


def get_latest_model_version(client, model_name: str) -> str:
    """
    Get the latest registered version of a model.

    Args:
        client: Azure ML MLClient.
        model_name: Name of the registered model.

    Returns:
        Latest version string.

    Raises:
        ValueError: If no registered model found.
    """
    versions = list(client.models.list(name=model_name))
    if not versions:
        raise ValueError(
            f"No registered model found: {model_name}. "
            "Run register_best.py or pipeline first."
        )
    latest = max(versions, key=lambda m: int(m.version))
    logger.info("Latest model version: %s v%s", model_name, latest.version)
    return latest.version


def create_batch_endpoint(client) -> None:
    """
    Create the batch endpoint if it doesn't exist.

    Args:
        client: Azure ML MLClient.
    """
    endpoint = BatchEndpoint(
        name=BATCH_ENDPOINT_NAME,
        description=(
            "Batch emotion classification, processes full transcript CSVs "
            "asynchronously and stores results in blob storage."
        ),
        tags={"project": "emotion-mlops", "team": "nlp-group-1"},
    )
    logger.info("Creating batch endpoint: %s", BATCH_ENDPOINT_NAME)
    client.batch_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Batch endpoint ready: %s", BATCH_ENDPOINT_NAME)


def create_batch_deployment(client, model_version: str) -> None:
    """
    Create a batch deployment for the registered model.

    Uses score.py scoring script, processes data in mini-batches,
    and writes predictions to predictions.csv in blob storage.

    Args:
        client: Azure ML MLClient.
        model_version: Version of emotion-classifier to deploy.
    """
    deployment = BatchDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=BATCH_ENDPOINT_NAME,
        model=f"azureml:{MODEL_NAME}:{model_version}",
        code_configuration=CodeConfiguration(
            code=str(_REPO_ROOT / "src"),
            scoring_script="emotion_mlops/azure/score.py",
        ),
        environment=f"azureml:{ENV_NAME}",
        compute=COMPUTE_NAME,
        mini_batch_size=10,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.csv",
        retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
        max_concurrency_per_instance=2,
        instance_count=1,
        logging_level="info",
    )
    logger.info(
        "Creating batch deployment: %s (model v%s)",
        DEPLOYMENT_NAME,
        model_version,
    )
    client.batch_deployments.begin_create_or_update(deployment).result()

    # Set as default deployment
    endpoint = client.batch_endpoints.get(BATCH_ENDPOINT_NAME)
    endpoint.defaults.deployment_name = DEPLOYMENT_NAME
    client.batch_endpoints.begin_create_or_update(endpoint).result()

    logger.info("Batch deployment ready: %s", DEPLOYMENT_NAME)


def invoke_batch_job(
    client,
    data_asset_name: str | None = None,
    data_asset_version: str | None = None,
    input_file: str | None = None,
    wait: bool = True,
) -> str:
    """
    Submit a batch inference job and optionally wait for completion.

    Args:
        client: Azure ML MLClient.
        data_asset_name: Name of registered data asset to process.
        data_asset_version: Version of the data asset.
        input_file: Path to local CSV file (alternative to data asset).
        wait: If True, poll until job completes and print results path.

    Returns:
        Job name for tracking in Azure ML portal.

    Raises:
        ValueError: If neither data_asset_name nor input_file provided.
    """
    if data_asset_name:
        input_data = Input(
            type=AssetTypes.URI_FILE,
            path=f"azureml:{data_asset_name}:{data_asset_version or '1'}",
        )
        logger.info(
            "Submitting batch job on data asset: %s v%s",
            data_asset_name,
            data_asset_version,
        )
    elif input_file:
        input_data = Input(type=AssetTypes.URI_FILE, path=input_file)
        logger.info("Submitting batch job on local file: %s", input_file)
    else:
        raise ValueError("Provide either --data-asset or --input-file")

    job = client.batch_endpoints.invoke(
        endpoint_name=BATCH_ENDPOINT_NAME,
        input=input_data,
    )
    logger.info("Batch job submitted: %s", job.name)
    print(f"\nBatch job submitted: {job.name}")

    if wait:
        logger.info("Waiting for batch job to complete...")
        print("Waiting for results (this may take a few minutes)...")
        while True:
            job = client.jobs.get(job.name)
            status = job.status
            print(f"  Status: {status}")
            if status in ("Completed", "Failed", "Canceled"):
                break
            time.sleep(30)

        if status == "Completed":
            print(f"\n[OK] Batch job complete: {job.name}")
            print(
                "   Download results: Azure ML portal -> Jobs -> "
                f"{job.name} -> Outputs + logs -> predictions.csv"
            )
        else:
            print(f"\n[FAIL] Batch job {status}: {job.name}")

    return job.name


def main(
    model_version: str | None,
    invoke: bool,
    data_asset_name: str | None,
    data_asset_version: str | None,
    input_file: str | None,
    wait: bool,
) -> None:
    """
    Main batch endpoint workflow.

    All configuration (endpoint name, compute, environment) is read
    from configs/azure_config.yaml.

    Args:
        model_version: Model version to deploy. None = latest.
        invoke: If True, also submit a batch inference job.
        data_asset_name: Data asset name for batch job.
        data_asset_version: Data asset version for batch job.
        input_file: Local CSV file for batch job.
        wait: If True, wait for batch job to complete.
    """
    client = get_ml_client()

    if model_version is None:
        model_version = get_latest_model_version(client, MODEL_NAME)

    create_batch_endpoint(client)
    create_batch_deployment(client, model_version)

    print(f"\nBatch endpoint ready: {BATCH_ENDPOINT_NAME}")
    print(f"   Model: {MODEL_NAME} v{model_version}")
    print("To run batch inference:")
    print(
        "   python src/emotion_mlops/azure/deploy_batch.py "
        "--invoke --data-asset emotion-test --data-version 1"
    )

    if invoke:
        invoke_batch_job(
            client,
            data_asset_name=data_asset_name,
            data_asset_version=data_asset_version,
            input_file=input_file,
            wait=wait,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy emotion-classifier as Azure ML Batch Endpoint"
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Model version to deploy (default: latest)",
    )
    parser.add_argument(
        "--invoke",
        action="store_true",
        help="Submit a batch inference job after deployment",
    )
    parser.add_argument(
        "--data-asset",
        default=None,
        help="Name of registered data asset to process",
    )
    parser.add_argument(
        "--data-version",
        default="1",
        help="Data asset version (default: 1)",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to local CSV file for batch inference",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for batch job to complete",
    )
    args = parser.parse_args()

    main(
        model_version=args.model_version,
        invoke=args.invoke,
        data_asset_name=args.data_asset,
        data_asset_version=args.data_version,
        input_file=args.input_file,
        wait=not args.no_wait,
    )
