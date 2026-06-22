"""
Azure ML configuration loader.

Loads shared Azure ML settings from configs/azure_config.yaml
so all azure scripts use the same compute, environment and data assets.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_FILE = _REPO_ROOT / "configs" / "azure_config.yaml"


def load_azure_config() -> dict:
    """
    Load Azure ML configuration from configs/azure_config.yaml.

    Returns:
        Dictionary with compute, environment and data settings.

    Raises:
        FileNotFoundError: If azure_config.yaml does not exist.
    """
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(f"Azure config not found: {_CONFIG_FILE}")

    with open(_CONFIG_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("Loaded Azure config from: %s", _CONFIG_FILE)
    return cfg


# Convenient shortcuts
def get_compute_name() -> str:
    """Return the compute target name."""
    return load_azure_config()["compute"]["name"]


def get_env_name() -> str:
    """Return the environment name:version string."""
    cfg = load_azure_config()["environment"]
    return f"{cfg['name']}:{cfg['version']}"


def get_raw_data_asset() -> tuple[str, str]:
    """Return (asset_name, version) for raw data."""
    cfg = load_azure_config()["data"]
    return cfg["raw_asset"], cfg["raw_version"]


def get_instance_type() -> str:
    cfg = load_azure_config()
    return cfg["compute"].get("instance_type", "defaultinstancetype")


def get_endpoint_name() -> str:
    cfg = load_azure_config()
    return cfg["deployment"]["endpoint_name"]


def get_batch_endpoint_name() -> str:
    cfg = load_azure_config()
    return cfg["deployment"]["batch_endpoint_name"]


def get_default_deployment() -> str:
    cfg = load_azure_config()
    return cfg["deployment"]["default_deployment"]


def get_deployment_instance_type() -> str:
    cfg = load_azure_config()
    return cfg["deployment"]["instance_type"]


def get_traffic() -> dict:
    cfg = load_azure_config()
    return cfg["deployment"]["traffic"]


def get_model_versions() -> dict:
    cfg = load_azure_config()
    return cfg["deployment"]["model_versions"]
