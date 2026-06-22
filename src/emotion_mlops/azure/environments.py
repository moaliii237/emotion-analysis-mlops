"""
Azure ML environment registration for emotion classification.

Registers a conda-based environment in Azure ML containing
all dependencies for training, evaluation, and inference.

Usage:
    python src/emotion_mlops/azure/environments.py
"""

import logging
from pathlib import Path

from azure.ai.ml.entities import Environment

from emotion_mlops.azure.config import load_azure_config
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


def register_environment() -> None:
    """
    Register the emotion classification environment in Azure ML.

    Reads environment name and version from configs/azure_config.yaml.
    """
    cfg = load_azure_config()
    version = cfg["environment"]["version"]
    name = cfg["environment"]["name"]

    logger.info("Connecting to Azure ML workspace...")
    client = get_ml_client()

    logger.info("Registering environment: %s v%s", name, version)
    env = Environment(
        name=name,
        description="DeBERTa emotion classification environment",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="configs/conda.yaml",
        version=version,
    )
    client.environments.create_or_update(env)
    logger.info("Environment registered successfully: %s v%s", name, version)
    print(f"Registered environment: {name} v{version}")


if __name__ == "__main__":
    register_environment()
