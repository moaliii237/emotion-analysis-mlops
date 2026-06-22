"""
Azure ML data asset management for emotion classification.

Uploads local CSV files to Azure ML as versioned data assets.
Run this script after preprocessing to upload train/val/test splits.

Usage:
    python src/emotion_mlops/azure/data_assets.py
"""

import logging
from pathlib import Path

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

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


def upload_data_asset(
    local_path: str,
    name: str,
    version: str = "1",
    description: str = "",
) -> None:
    """
    Upload a local file to Azure ML as a versioned data asset.

    Args:
        local_path: Path to the local CSV file.
        name: Name for the data asset in Azure ML.
        version: Version string for the asset.
        description: Optional description.

    Raises:
        FileNotFoundError: If local_path does not exist.
    """
    if not Path(local_path).exists():
        raise FileNotFoundError(f"Data file not found: {local_path}")

    logger.info("Uploading data asset: %s v%s from %s", name, version, local_path)
    client = get_ml_client()
    data = Data(
        path=local_path,
        type=AssetTypes.URI_FILE,
        name=name,
        version=version,
        description=description,
    )
    client.data.create_or_update(data)
    logger.info("Successfully uploaded: %s v%s", name, version)
    print(f"Uploaded: {name} v{version}")


if __name__ == "__main__":
    # Load asset names and versions from shared config
    cfg = load_azure_config()

    # -- Raw data --------------------------------------------------------------
    # Used as input to the preprocessing pipeline step
    upload_data_asset(
        local_path="data/raw/subset_500.csv",
        name=cfg["data"]["raw_asset"],
        version=cfg["data"]["raw_version"],
        description="Raw GoEmotions dataset, 58k rows, majority vote labels",
    )

    # upload_data_asset(
    #     local_path="data/processed/train.csv",
    #     name=cfg["data"]["train_asset"],
    #     version=cfg["data"]["train_version"],
    #     description="GoEmotions training split (70%), 40k rows",
    # )
    # upload_data_asset(
    #     local_path="data/processed/val.csv",
    #     name=cfg["data"]["val_asset"],
    #     version=cfg["data"]["val_version"],
    #     description="GoEmotions validation split (15%), 8.7k rows",
    # )
    # upload_data_asset(
    #     local_path="data/processed/test.csv",
    #     name=cfg["data"]["test_asset"],
    #     version=cfg["data"]["test_version"],
    #     description="GoEmotions test split (15%), 8.7k rows",
    # )
