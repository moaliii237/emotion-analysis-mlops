"""
Deploy registered emotion-classifier model as Azure ML Online Endpoint.

Supports blue-green deployment strategy for zero-downtime model updates.
Version control allows selecting specific model versions for deployment.

Model versions and traffic split are configured in configs/azure_config.yaml:

    deployment:
      model_versions:
        blue: 1
        green: 2
      traffic:
        blue: 60
        green: 40

Deployment strategies:
    Blue-Green: Deploy new version as 'green', test, then shift traffic
    Canary:     Split traffic between versions (e.g. 90% blue, 10% green)

Usage:
    # Deploy using versions from azure_config.yaml
    python src/emotion_mlops/azure/deploy_endpoint.py

    # Deploy specific version (overrides config)
    python src/emotion_mlops/azure/deploy_endpoint.py --model-version 2

    # Deploy as 'green' for blue-green strategy
    python src/emotion_mlops/azure/deploy_endpoint.py --deployment-name green

    # Shift all traffic to green
    python src/emotion_mlops/azure/deploy_endpoint.py \
        --set-traffic blue=0 green=100

    # Canary: 90% blue, 10% green
    python src/emotion_mlops/azure/deploy_endpoint.py \
        --set-traffic blue=90 green=10
"""

import argparse
import logging
from pathlib import Path

from azure.ai.ml.entities import (
    CodeConfiguration,
    KubernetesOnlineDeployment,
    KubernetesOnlineEndpoint,
)

from emotion_mlops.azure.config import (
    get_default_deployment,
    get_deployment_instance_type,
    get_endpoint_name,
    get_env_name,
    get_model_versions,
    get_traffic,
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
ENDPOINT_NAME = get_endpoint_name()
DEFAULT_DEPLOYMENT = get_default_deployment()
INSTANCE_TYPE = get_deployment_instance_type()
DEFAULT_TRAFFIC = get_traffic()
MODEL_VERSIONS = get_model_versions()
ENV_NAME = get_env_name()
MODEL_NAME = "emotion-classifier"
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
            f"No registered model found: {model_name}. " "Run register_best.py first."
        )
    latest = max(versions, key=lambda m: int(m.version))
    logger.info(
        "Latest model version: %s v%s (f1=%s)",
        model_name,
        latest.version,
        latest.tags.get("test_f1", "unknown"),
    )
    return latest.version


def create_or_update_endpoint(client) -> None:
    """
    Create the online endpoint if it doesn't exist, or update it.
    Deletes existing Managed endpoint if found (must use Kubernetes type).

    Args:
        client: Azure ML MLClient.
    """
    try:
        existing = client.online_endpoints.get(ENDPOINT_NAME)
        if existing.kind != "K8S":
            logger.info("Deleting existing Managed endpoint: %s", ENDPOINT_NAME)
            client.online_endpoints.begin_delete(ENDPOINT_NAME).result()
            logger.info("Deleted existing Managed endpoint.")
    except Exception:
        pass  # endpoint doesn't exist yet

    endpoint = KubernetesOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Real-time emotion classification endpoint",
        auth_mode="key",
        compute="lambda-2",
        tags={"project": "emotion-mlops", "team": "nlp-group-1"},
    )
    logger.info("Creating/updating endpoint: %s", ENDPOINT_NAME)
    client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Endpoint ready: %s", ENDPOINT_NAME)


def deploy_model(
    client,
    model_version: str,
    deployment_name: str,
    instance_type: str,
) -> None:
    """
    Deploy a registered model version to the endpoint.

    Args:
        client: Azure ML MLClient.
        model_version: Version of emotion-classifier to deploy.
        deployment_name: Name for this deployment (e.g. 'blue', 'green').
        instance_type: Azure compute instance type.
    """
    deployment = KubernetesOnlineDeployment(
        name=deployment_name,
        endpoint_name=ENDPOINT_NAME,
        model=f"azureml:{MODEL_NAME}:{model_version}",
        code_configuration=CodeConfiguration(
            code=str(_REPO_ROOT / "src"),
            scoring_script="emotion_mlops/azure/score.py",
        ),
        environment=f"azureml:{ENV_NAME}",
        instance_type=instance_type,
        instance_count=1,
    )
    logger.info(
        "Deploying %s v%s as '%s'...",
        MODEL_NAME,
        model_version,
        deployment_name,
    )
    client.online_deployments.begin_create_or_update(deployment).result()
    logger.info("Deployment ready: %s", deployment_name)


def set_traffic(client, traffic: dict[str, int]) -> None:
    """
    Set traffic split between deployments.

    Args:
        client: Azure ML MLClient.
        traffic: Dict mapping deployment name to traffic percentage.
                 Must sum to 100. e.g. {"blue": 90, "green": 10}

    Raises:
        ValueError: If traffic percentages do not sum to 100.
    """
    total = sum(traffic.values())
    if total != 100:
        raise ValueError(f"Traffic must sum to 100, got {total}")

    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = traffic
    client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Traffic updated: %s", traffic)


def main(
    model_version: str | None,
    deployment_name: str,
    instance_type: str,
    traffic: dict[str, int] | None,
    set_traffic_only: bool,
) -> None:
    """
    Main deployment workflow.

    All defaults are read from configs/azure_config.yaml.
    CLI args override config values when provided.

    Model version priority:
        1. --model-version CLI arg
        2. model_versions[deployment_name] from azure_config.yaml
        3. latest registered version

    Args:
        model_version: Model version to deploy. None = from config or latest.
        deployment_name: Name for this deployment.
        instance_type: Azure compute instance type.
        traffic: Traffic split dict. None = use config default.
        set_traffic_only: If True, only update traffic without deploying.
    """
    client = get_ml_client()

    if set_traffic_only:
        if traffic is None:
            raise ValueError("--set-traffic required when using set_traffic_only")
        set_traffic(client, traffic)
        return

    # Resolve model version, CLI -> config -> latest
    if model_version is None:
        if deployment_name in MODEL_VERSIONS:
            model_version = str(MODEL_VERSIONS[deployment_name])
            logger.info(
                "Using model version from config: %s -> v%s",
                deployment_name,
                model_version,
            )
        else:
            model_version = get_latest_model_version(client, MODEL_NAME)

    # Create endpoint
    create_or_update_endpoint(client)

    # Deploy model
    deploy_model(client, model_version, deployment_name, instance_type)

    # Set traffic, use config default if not overridden
    if traffic is None:
        traffic = DEFAULT_TRAFFIC
    set_traffic(client, traffic)

    # Print endpoint info
    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    keys = client.online_endpoints.get_keys(ENDPOINT_NAME)
    print("Deployment complete")
    print(f"   Endpoint:   {ENDPOINT_NAME}")
    print(f"   Deployment: {deployment_name}")
    print(f"   Model:      {MODEL_NAME} v{model_version}")
    print(f"   URL:        {endpoint.scoring_uri}")
    print(f"   API key:    {keys.primary_key[:8]}...")
    print(f"   Traffic:    {endpoint.traffic}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy emotion-classifier to Azure ML Online Endpoint"
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Model version to deploy (default: from config or latest)",
    )
    parser.add_argument(
        "--deployment-name",
        default=DEFAULT_DEPLOYMENT,
        choices=["blue", "green"],
        help=f"Deployment name for blue-green strategy (default: {DEFAULT_DEPLOYMENT})",
    )
    parser.add_argument(
        "--instance-type",
        default=INSTANCE_TYPE,
        help=f"Azure compute instance type (default: {INSTANCE_TYPE})",
    )
    parser.add_argument(
        "--set-traffic",
        nargs="+",
        metavar="NAME=PCT",
        help="Set traffic split, e.g. --set-traffic blue=90 green=10",
    )
    args = parser.parse_args()

    # Parse traffic arg
    traffic = None
    set_traffic_only = False
    if args.set_traffic:
        traffic = {}
        for item in args.set_traffic:
            name, pct = item.split("=")
            traffic[name] = int(pct)
        set_traffic_only = True

    main(
        model_version=args.model_version,
        deployment_name=args.deployment_name,
        instance_type=args.instance_type,
        traffic=traffic,
        set_traffic_only=set_traffic_only,
    )
