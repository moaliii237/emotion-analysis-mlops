import json
import logging
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import (
    AzureCliCredential,
    ClientSecretCredential,
    WorkloadIdentityCredential,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
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


def get_ml_client() -> MLClient:
    """
    Connect to Azure ML workspace.

    Credential priority:
        1. WorkloadIdentityCredential (GitHub Actions OIDC)
        2. ClientSecretCredential (Service Principal via config.json)
        3. AzureCliCredential (local dev after az login)
    """
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    federated_token_file = os.environ.get("AZURE_FEDERATED_TOKEN_FILE")

    config_path = Path.cwd() / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        client_id = client_id or cfg.get("client_id")
        client_secret = client_secret or cfg.get("client_secret")
        tenant_id = tenant_id or cfg.get("tenant_id")

    # GitHub Actions OIDC, federated token file is set by azure/login@v2
    if federated_token_file and client_id and tenant_id:
        logger.info("Using WorkloadIdentityCredential (OIDC)")
        credential = WorkloadIdentityCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            token_file_path=federated_token_file,
        )
    elif client_id and client_secret and tenant_id:
        logger.info("Using ClientSecretCredential")
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
    else:
        logger.info("Using AzureCliCredential")
        credential = AzureCliCredential(tenant_id=tenant_id if tenant_id else None)

    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = os.environ.get("AZURE_WORKSPACE_NAME")

    config_path = Path.cwd() / "config.json"
    if not config_path.exists() and not all(
        [subscription_id, resource_group, workspace_name]
    ):
        raise ValueError(
            "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP and AZURE_WORKSPACE_NAME "
            "env vars are required when config.json is not present."
        )

    if config_path.exists():
        client = MLClient.from_config(credential=credential)
    else:
        client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
    logger.info("Connected to workspace: %s", client.workspace_name)
    return client


if __name__ == "__main__":
    client = get_ml_client()
    print(client.workspace_name)
