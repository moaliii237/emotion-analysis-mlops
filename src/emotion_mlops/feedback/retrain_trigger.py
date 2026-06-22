"""
Retrain trigger for the feedback flywheel, Card #1486 / ILO 9.5D.

Closes the loop:
    feedback DB  ->  corrections CSV  ->  merged training CSV  ->
    new Azure ML data asset version  ->  retraining pipeline

Two modes:
    - dry_run=True  : show what WOULD happen, write CSVs locally, no Azure calls.
                      Use this for testing without burning compute.
    - dry_run=False : actually upload data asset + submit the training pipeline.
                      Returns the Azure job URL.

Usage from CLI:
    # Dry run (no Azure costs, no uploads)
    python -m emotion_mlops.feedback.retrain_trigger --dry-run

    # Real run (uploads + triggers training)
    python -m emotion_mlops.feedback.retrain_trigger --live

Usage from API:
    POST /feedback/retrain?dry_run=true
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from emotion_mlops.feedback.db import export_corrections_for_retraining, get_stats

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
# Candidate local base-data files, tried in order for the dry-run PREVIEW only.
# A live run mounts the base data from the Azure ML data asset instead, so
# these files are NOT required for the retrain to work in CI.
_LOCAL_BASE_CANDIDATES = (
    _REPO_ROOT / "data" / "raw" / "subset_500.csv",
    _REPO_ROOT / "data" / "raw" / "goemotions_collapsed_majority.csv",
    _REPO_ROOT / "data" / "external" / "goemotions_collapsed_majority.csv",
)
_FEEDBACK_OUT_DIR = _REPO_ROOT / "data" / "feedback_exports"
_FEEDBACK_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Azure ML data asset that stores the uploaded feedback corrections. The base
# data asset (emotion-raw) is left untouched; corrections are versioned
# separately so each retrain records exactly which corrections it used.
_CORRECTIONS_ASSET = "emotion-feedback"


@dataclass
class RetrainResult:
    """Structured outcome of a retrain trigger call, used by API + CLI."""

    mode: str
    corrections_count: int
    merged_rows: int
    corrections_csv: str
    merged_csv: str
    new_data_asset: Optional[str] = None
    new_data_version: Optional[str] = None
    job_name: Optional[str] = None
    job_url: Optional[str] = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def export_corrections_csv() -> tuple[Path, int]:
    """
    Dump the corrections-only feedback rows to a timestamped CSV.

    Returns:
        (path_to_csv, number_of_rows)
    """
    corrections = export_corrections_for_retraining()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = _FEEDBACK_OUT_DIR / f"corrections_{ts}.csv"

    if not corrections:
        pd.DataFrame(columns=["Sentence", "Emotion"]).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        logger.warning(
            "[Retrain] No corrections to export, wrote empty CSV: %s", out_path
        )
        return out_path, 0

    df = pd.DataFrame(corrections)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("[Retrain] Exported %d corrections to %s", len(df), out_path)
    return out_path, len(df)


def _local_base_csv() -> Optional[Path]:
    """
    Return the first existing local base-data CSV, or None.

    Used only for the dry-run preview. A live run reads the base data from the
    Azure ML data asset, so returning None here is fine in CI.
    """
    for path in _LOCAL_BASE_CANDIDATES:
        if path.exists():
            return path
    return None


def preview_merge(corrections_csv: Path) -> tuple[Path, int]:
    """
    Locally merge base + corrections so the dry run can report a row count.

    Uses the same merge logic the live job runs in the cloud. If no local base
    CSV is available (e.g. in CI), it falls back to corrections-only and warns
    that the real base merge happens in the cloud.

    Returns:
        (preview_csv_path, total_rows)
    """
    from emotion_mlops.feedback.merge_inputs import merge as merge_csvs

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    preview_path = _FEEDBACK_OUT_DIR / f"merged_preview_{ts}.csv"

    base_csv = _local_base_csv()
    if base_csv is None:
        logger.warning(
            "[Retrain] No local base CSV for preview, showing corrections only. "
            "The live run merges with the Azure data asset in the cloud."
        )
        corrections_df = pd.read_csv(corrections_csv)
        corrections_df.to_csv(preview_path, index=False, encoding="utf-8-sig")
        return preview_path, len(corrections_df)

    n_rows = merge_csvs(str(base_csv), str(corrections_csv), str(preview_path))
    logger.info(
        "[Retrain] Preview merged %s + corrections -> %s (%d rows)",
        base_csv.name,
        preview_path,
        n_rows,
    )
    return preview_path, n_rows


def upload_corrections_asset(corrections_csv: Path) -> tuple[str, str]:
    """
    Upload the feedback corrections as a versioned Azure ML data asset.

    The base training data (emotion-raw) is left untouched; the corrections
    are stored as their own versioned asset so the retraining job can mount
    both and the run records exactly which corrections were used (lineage).

    Returns:
        (asset_name, asset_version)
    """
    from emotion_mlops.azure.data_assets import upload_data_asset

    version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    logger.info(
        "[Retrain] Uploading corrections data asset: %s v%s",
        _CORRECTIONS_ASSET,
        version,
    )
    upload_data_asset(
        local_path=str(corrections_csv),
        name=_CORRECTIONS_ASSET,
        version=version,
        description=(
            "User feedback corrections for the emotion classifier, "
            f"auto-generated at {datetime.utcnow().isoformat()}Z"
        ),
    )
    return _CORRECTIONS_ASSET, version


def submit_training_pipeline(
    base_asset: str,
    base_version: str,
    corrections_asset: str,
    corrections_version: str,
) -> tuple[str, str]:
    """
    Submit the retraining job: merge base + corrections in-cloud, then train.

    Mounts BOTH the base data asset and the corrections asset as inputs. Azure
    ML provides them as local files inside the job, so the first step
    concatenates them into one CSV before the normal preprocess + train. No
    local base data is needed in CI, and the run records both input versions.

    Returns:
        (job_name, studio_url)
    """
    from azure.ai.ml import Input, command
    from azure.ai.ml.constants import AssetTypes

    from emotion_mlops.azure.config import get_compute_name, get_env_name
    from emotion_mlops.azure.connect import get_ml_client

    client = get_ml_client()
    inputs = {
        "base_data": Input(
            type=AssetTypes.URI_FILE,
            path=f"azureml:{base_asset}:{base_version}",
        ),
        "corrections": Input(
            type=AssetTypes.URI_FILE,
            path=f"azureml:{corrections_asset}:{corrections_version}",
        ),
    }

    job = command(
        code=str(_REPO_ROOT / "src"),
        command=(
            "python -m emotion_mlops.feedback.merge_inputs "
            "--base ${{inputs.base_data}} "
            "--corrections ${{inputs.corrections}} "
            "--output data/raw/merged.csv && "
            "python -m emotion_mlops.data.data_preprocessing "
            "--input data/raw/merged.csv "
            "--output data/processed/ && "
            "python -m emotion_mlops.models.model_training "
            "--data data/processed/ "
            "--learning-rate 2e-5 "
            "--batch-size 8 "
            "--epochs 3 "
            "--weight-decay 0.05 "
            "--output models/feedback_retrain/"
        ),
        inputs=inputs,
        environment=get_env_name(),
        compute=get_compute_name(),
        resources={"instance_type": "gpu"},
        display_name=f"feedback-retrain-{corrections_version}",
        description=(
            "Retraining triggered by the user feedback flywheel (ILO 9.5D). "
            f"Merges base data {base_asset}:{base_version} with user "
            f"corrections {corrections_asset}:{corrections_version}, then "
            "preprocesses and retrains."
        ),
    )

    submitted = client.jobs.create_or_update(job)
    logger.info("[Retrain] Pipeline submitted: %s", submitted.name)
    return submitted.name, submitted.studio_url


def run_retrain(dry_run: bool = True) -> RetrainResult:
    """
    Execute the full retrain flow.

    Args:
        dry_run: If True, stop after merging the CSV (no Azure calls).
                 If False, also upload the data asset and submit the job.

    Returns:
        RetrainResult with all the artefacts and Azure job pointers.
    """
    stats = get_stats()
    logger.info(
        "[Retrain] Starting retrain (dry_run=%s). Current feedback stats: %s",
        dry_run,
        stats,
    )

    corrections_csv, n_corrections = export_corrections_csv()
    preview_csv, n_preview = preview_merge(corrections_csv)

    result = RetrainResult(
        mode="dry_run" if dry_run else "live",
        corrections_count=n_corrections,
        merged_rows=n_preview,
        corrections_csv=str(corrections_csv),
        merged_csv=str(preview_csv),
    )

    if dry_run:
        result.note = (
            f"Dry run only. {n_corrections} correction(s) ready; preview merge "
            f"= {n_preview} rows. Run with --live (CLI) or dry_run=false (API) "
            "to upload the corrections asset and submit the Azure ML retrain "
            "(base data is merged from the emotion-raw asset in the cloud)."
        )
        logger.info("[Retrain] Dry run done. %s", result.note)
        return result

    if n_corrections == 0:
        result.note = (
            "No user corrections available, nothing to retrain on. "
            "Collect feedback corrections first."
        )
        logger.warning("[Retrain] %s", result.note)
        return result

    # Live: base data stays in Azure, corrections uploaded as their own asset,
    # and the job merges the two in the cloud before training.
    from emotion_mlops.azure.config import get_raw_data_asset

    base_asset, base_version = get_raw_data_asset()
    corr_asset, corr_version = upload_corrections_asset(corrections_csv)
    result.new_data_asset = corr_asset
    result.new_data_version = corr_version

    job_name, job_url = submit_training_pipeline(
        base_asset, base_version, corr_asset, corr_version
    )
    result.job_name = job_name
    result.job_url = job_url
    result.note = (
        f"Retraining submitted. Base data {base_asset}:{base_version} merged "
        f"in-cloud with {n_corrections} user correction(s) "
        f"({corr_asset}:{corr_version}). Watch the job at: {job_url}"
    )

    logger.info("[Retrain] Live run complete. Job: %s", job_name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trigger retraining using user feedback corrections."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would happen without calling Azure (default).",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Actually upload data asset + submit training job.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dry_run = not args.live
    result = run_retrain(dry_run=dry_run)

    print()
    print("=" * 70)
    print(" FEEDBACK RETRAIN RESULT")
    print("=" * 70)
    for k, v in result.to_dict().items():
        print(f"  {k:22s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
