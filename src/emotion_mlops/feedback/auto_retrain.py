"""
Automatic retraining trigger for the feedback flywheel.

WHAT THIS DOES
    Every time it runs, it checks how many NEW user corrections have been
    collected since the last retrain. When that count reaches a threshold
    (default DEFAULT_THRESHOLD) it runs the full champion/challenger pipeline:

        1. export corrections + merge with the raw training data
        2. upload the merged data as a NEW Azure ML data asset version
        3. submit the training pipeline (preprocess -> train -> evaluate ->
           auto-register if the eval F1 gate passes)
        4. wait for the job to finish
        5. find the newly registered emotion-classifier version
        6. F1 GATE: only promote if the new model beats the live model
           (and clears a safety floor). A few noisy corrections must NOT
           push a worse model live
        7. PROMOTE: deploy the new version to a fresh endpoint slot and shift
           100% of traffic to it (zero-downtime, reuses deploy_endpoint.py)

    A WATERMARK in the retrain_log table records the highest feedback id each
    completed retrain consumed, so the same corrections are never used twice.

TWO MODES
    --dry-run (default): prints exactly what each step WOULD do. Reads the real
        feedback DB so the correction count is real, but makes NO Azure calls
        and spends NO money. This is the mentor-demo mode.
    --live: actually runs the retrain + promotion. Costs ~15 min of Azure GPU
        time and changes the live endpoint. Requires the team's Azure deploy
        modules (deploy_endpoint.py, pipeline.py, the deployment block in
        azure_config.yaml) to be present on the branch.

SAFETY
    Automatic firing from the /feedback endpoint is OPT-IN via the
    AUTO_RETRAIN_ENABLED=1 environment variable, so a demo never accidentally
    launches an expensive Azure job. Without it, feedback is just recorded and
    you trigger retraining yourself (CLI or /feedback/auto-check).

Usage:
    # Safe: show the logic + the real correction count, no Azure calls
    python -m emotion_mlops.feedback.auto_retrain --dry-run

    # Real: retrain + promote if the new model wins (uses Azure)
    python -m emotion_mlops.feedback.auto_retrain --live

    # Lower the threshold for testing
    python -m emotion_mlops.feedback.auto_retrain --dry-run --threshold 5
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from emotion_mlops.feedback import db

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 10
MODEL_NAME = "emotion-classifier"
# Never auto-promote a model below this macro-F1, regardless of the live model.
MIN_PROMOTE_F1 = 0.50


def _now() -> str:
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Result payload
# ---------------------------------------------------------------------------


@dataclass
class AutoRetrainResult:
    """Outcome of one auto-retrain check/run."""

    triggered: bool
    reason: str
    new_corrections: int
    watermark: int
    threshold: int
    mode: str = "dry_run"
    data_version: Optional[str] = None
    job_name: Optional[str] = None
    new_model_version: Optional[str] = None
    new_f1: Optional[float] = None
    live_f1: Optional[float] = None
    promoted: bool = False
    steps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Threshold check (the "enough new corrections?" decision)
# ---------------------------------------------------------------------------


def should_retrain(threshold: int = DEFAULT_THRESHOLD) -> dict:
    """
    Decide whether enough NEW corrections have accumulated to retrain.

    Returns a dict with the counts and a `ready` boolean. `ready` is True only
    when there are >= threshold new corrections AND no retrain is in flight.
    """
    counts = db.count_new_corrections()
    in_flight = db.is_retrain_in_flight()
    ready = counts["new_corrections"] >= threshold and not in_flight
    return {**counts, "threshold": threshold, "in_flight": in_flight, "ready": ready}


# ---------------------------------------------------------------------------
# Azure helpers (all lazy-imported so dry-run needs no Azure deps)
# ---------------------------------------------------------------------------


def _client():
    from emotion_mlops.azure.connect import get_ml_client

    return get_ml_client()


def _latest_version(client) -> Optional[int]:
    versions = list(client.models.list(name=MODEL_NAME))
    return max((int(m.version) for m in versions), default=None)


def _version_f1(client, version) -> Optional[float]:
    try:
        m = client.models.get(name=MODEL_NAME, version=str(version))
        tags = m.tags or {}
        tag = tags.get("test_f1") or tags.get("val_f1") or tags.get("f1")
        return float(tag) if tag else None
    except Exception:
        return None


def _live_model_f1(client) -> Optional[float]:
    """F1 of the model version currently serving the most endpoint traffic."""
    try:
        from emotion_mlops.azure.config import get_endpoint_name

        ep_name = get_endpoint_name()
        ep = client.online_endpoints.get(ep_name)
        if not ep.traffic:
            return None
        live_dep = max(ep.traffic.items(), key=lambda kv: kv[1])[0]
        dep = client.online_deployments.get(name=live_dep, endpoint_name=ep_name)
        version = str(dep.model).rsplit("/", 1)[-1]
        return _version_f1(client, version)
    except Exception as e:
        logger.warning("[AutoRetrain] could not read live model f1: %s", e)
        return None


def _wait_for_job(client, job_name: str, poll: int = 60, timeout: int = 90 * 60) -> str:
    """Poll until the Azure job reaches a terminal state. Returns the status."""
    waited = 0
    while waited < timeout:
        status = client.jobs.get(job_name).status
        logger.info("[AutoRetrain] job %s status=%s (%ds)", job_name, status, waited)
        if status in ("Completed", "Failed", "Canceled"):
            return status
        time.sleep(poll)
        waited += poll
    return "TimedOut"


def _promote(client, version, dry_run: bool, steps: list) -> str:
    """Deploy the new version to a fresh slot and shift 100% of traffic to it."""
    slot = f"green-v{version}"
    steps.append(
        f"PROMOTE: deploy {MODEL_NAME}:v{version} to new slot '{slot}', "
        f"then set_traffic {{{slot}: 100, others: 0}} (zero-downtime)"
    )
    if dry_run:
        return slot
    from emotion_mlops.azure import deploy_endpoint as de
    from emotion_mlops.azure.config import (
        get_deployment_instance_type,
        get_endpoint_name,
    )

    de.create_or_update_endpoint(client)
    de.deploy_model(client, str(version), slot, get_deployment_instance_type())
    ep = client.online_endpoints.get(get_endpoint_name())
    new_traffic = {k: 0 for k in (ep.traffic or {})}
    new_traffic[slot] = 100
    de.set_traffic(client, new_traffic)
    return slot


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_auto_retrain(
    threshold: int = DEFAULT_THRESHOLD,
    dry_run: bool = True,
    min_promote_f1: float = MIN_PROMOTE_F1,
) -> AutoRetrainResult:
    """
    Check the threshold and, if met, run the full retrain -> gate -> promote flow.

    Args:
        threshold: Number of new corrections required to fire.
        dry_run: If True, print the plan and make NO Azure calls.
        min_promote_f1: Safety floor; never promote below this macro-F1.

    Returns:
        AutoRetrainResult describing what happened (or would happen).
    """
    db.init_db()
    chk = should_retrain(threshold)
    res = AutoRetrainResult(
        triggered=False,
        reason="",
        new_corrections=chk["new_corrections"],
        watermark=chk["watermark"],
        threshold=threshold,
        mode="dry_run" if dry_run else "live",
    )

    if chk["in_flight"]:
        res.reason = "a retrain is already in flight; skipping"
        return res
    if not chk["ready"]:
        res.reason = (
            f"only {chk['new_corrections']}/{threshold} new corrections "
            f"since feedback id {chk['watermark']}; not enough yet"
        )
        return res

    res.triggered = True
    res.reason = f"{chk['new_corrections']} new corrections >= {threshold} threshold"
    steps = res.steps

    # Bookkeeping row, only in live mode (a dry-run must not advance the
    # watermark or mark a retrain 'in flight').
    log_id = (
        db.record_retrain_start(chk["from_id"], chk["to_id"], chk["new_corrections"])
        if not dry_run
        else None
    )

    try:
        # STEP 1, upload the corrections as their own versioned data asset.
        # The base data (emotion-raw) is mounted unchanged in the pipeline and
        # merged with these corrections in the cloud, so no local base data is
        # needed in CI.
        steps.append("STEP 1 - export corrections + upload as a versioned data asset")
        if dry_run:
            steps.append(
                "  (dry-run) would write corrections CSV to data/feedback_exports/ "
                "and upload it as the 'emotion-feedback' data asset"
            )
            data_version = f"DRYRUN-upto-{chk['to_id']}"
        else:
            from emotion_mlops.feedback.retrain_trigger import (
                export_corrections_csv,
                upload_corrections_asset,
            )

            corr_csv, _ = export_corrections_csv()
            _, data_version = upload_corrections_asset(corr_csv)
            db.update_retrain_log(log_id, new_data_version=data_version)
        res.data_version = data_version
        steps.append(f"  -> corrections data asset version: {data_version}")

        client = None if dry_run else _client()
        before_version = None if dry_run else _latest_version(client)

        # STEP 2, submit the training pipeline (it auto-registers on its own F1 gate)
        steps.append(
            "STEP 2 - submit Azure ML pipeline (preprocess -> train -> evaluate; "
            "the evaluate step auto-registers emotion-classifier if its F1 gate passes)"
        )
        if dry_run:
            steps.append(
                "  (dry-run) would call build_and_submit_pipeline(data_version=...) on compute lambda-2"
            )
            job_name = "DRYRUN-job"
        else:
            from emotion_mlops.azure.config import get_raw_data_asset
            from emotion_mlops.azure.pipeline import build_and_submit_pipeline

            base_asset, base_version = get_raw_data_asset()
            db.update_retrain_log(log_id, status="training")
            submitted = build_and_submit_pipeline(
                data_asset_name=base_asset,
                data_asset_version=base_version,
                corrections_asset="emotion-feedback",
                corrections_version=data_version,
            )
            job_name = getattr(submitted, "name", str(submitted))
            db.update_retrain_log(log_id, job_name=job_name)
        res.job_name = job_name
        steps.append(f"  -> training job: {job_name}")

        # STEP 3, wait for the job
        steps.append("STEP 3 - wait for the training job to finish")
        if dry_run:
            steps.append(
                "  (dry-run) would poll client.jobs.get(job).status until Completed (~15 min)"
            )
        else:
            status = _wait_for_job(client, job_name)
            if status != "Completed":
                db.update_retrain_log(
                    log_id, status="failed", finished_at=_now(), note=f"job {status}"
                )
                res.reason = f"training job ended {status}"
                return res

        # STEP 4, find the newly registered version
        steps.append("STEP 4 - find the newly registered emotion-classifier version")
        if dry_run:
            res.new_model_version = "N+1"
            res.new_f1 = None
            steps.append(
                "  -> new model: emotion-classifier:vN+1 (f1 from the eval step)"
            )
        else:
            new_version = _latest_version(client)
            if before_version is not None and new_version == before_version:
                db.update_retrain_log(
                    log_id,
                    status="skipped",
                    finished_at=_now(),
                    note="pipeline did not register a new model (eval F1 gate failed)",
                )
                res.reason = (
                    "trained but registered nothing (model was below the eval F1 gate)"
                )
                return res
            new_f1 = _version_f1(client, new_version)
            db.update_retrain_log(
                log_id, new_model_version=str(new_version), new_f1=new_f1
            )
            res.new_model_version = str(new_version)
            res.new_f1 = new_f1
            steps.append(
                f"  -> new model: emotion-classifier:v{new_version} (f1={new_f1})"
            )

        # STEP 5, F1 gate (champion vs challenger)
        steps.append(
            "STEP 5 - F1 gate: promote ONLY if new f1 >= max(live f1, floor). "
            f"floor={min_promote_f1}"
        )
        if dry_run:
            steps.append(
                "  (dry-run) would compare new f1 against the live endpoint model's f1 tag"
            )
        else:
            live_f1 = _live_model_f1(client)
            res.live_f1 = live_f1
            if res.new_f1 is None:
                db.update_retrain_log(
                    log_id,
                    status="registered",
                    finished_at=_now(),
                    live_f1=live_f1,
                    promoted=0,
                    note="new model has no f1 tag; manual review required",
                )
                res.reason = "new model registered but untagged; NOT auto-promoted (needs manual review)"
                return res
            bar = max(min_promote_f1, live_f1 or 0.0)
            if res.new_f1 < bar:
                db.update_retrain_log(
                    log_id,
                    status="registered",
                    finished_at=_now(),
                    live_f1=live_f1,
                    promoted=0,
                    note=f"new f1 {res.new_f1} < bar {bar}; kept live model",
                )
                res.reason = f"new f1 {res.new_f1:.4f} did not beat bar {bar:.4f}; live model kept"
                return res

        # STEP 6, promote to the live endpoint
        steps.append("STEP 6 - promote the new model to the live endpoint")
        slot = _promote(client, res.new_model_version, dry_run, steps)
        if not dry_run:
            db.update_retrain_log(
                log_id,
                status="promoted",
                finished_at=_now(),
                live_f1=res.live_f1,
                promoted=1,
                note=f"promoted v{res.new_model_version} to slot {slot}",
            )
        res.promoted = True
        res.reason = (
            "dry-run complete: all gates pass -> would promote the new model to the live endpoint"
            if dry_run
            else f"promoted {MODEL_NAME}:v{res.new_model_version} to the live endpoint (slot {slot})"
        )
        return res

    except Exception as e:
        logger.error("[AutoRetrain] failed: %s", e)
        if not dry_run and log_id is not None:
            db.update_retrain_log(
                log_id, status="failed", finished_at=_now(), note=str(e)
            )
        res.reason = f"error: {e}"
        return res


# ---------------------------------------------------------------------------
# Hook called by the /feedback endpoint (opt-in via env var)
# ---------------------------------------------------------------------------


def maybe_trigger_after_feedback(
    threshold: int = DEFAULT_THRESHOLD,
) -> Optional[AutoRetrainResult]:
    """
    Called after each feedback insert. Fires a LIVE auto-retrain only when:
      - AUTO_RETRAIN_ENABLED=1 is set in the environment, AND
      - the threshold of new corrections has been reached.

    Returns the result if it ran, else None. Designed to be invoked from a
    FastAPI BackgroundTask so it never blocks the /feedback response.
    """
    if os.getenv("AUTO_RETRAIN_ENABLED") != "1":
        return None
    chk = should_retrain(threshold)
    if not chk["ready"]:
        return None
    logger.info(
        "[AutoRetrain] threshold reached (%d), firing live retrain",
        chk["new_corrections"],
    )
    return run_auto_retrain(threshold=threshold, dry_run=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_result(res: AutoRetrainResult) -> None:
    print()
    print("=" * 72)
    print(f"  AUTO-RETRAIN  [{res.mode}]")
    print("=" * 72)
    print(f"  new corrections since last retrain : {res.new_corrections}")
    print(f"  threshold                          : {res.threshold}")
    print(f"  watermark (last consumed id)       : {res.watermark}")
    print(f"  triggered                          : {res.triggered}")
    print(f"  reason                             : {res.reason}")
    if res.steps:
        print("  ---- plan ----")
        for s in res.steps:
            print(f"  {s}")
    if res.new_model_version:
        print(f"  new model version                  : {res.new_model_version}")
        print(f"  new f1 / live f1                   : {res.new_f1} / {res.live_f1}")
        print(f"  promoted                           : {res.promoted}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-retrain after N feedback corrections."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show the plan + real correction count, no Azure calls (default).",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Actually retrain + promote if the new model wins (uses Azure).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Corrections required to fire (default: {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    res = run_auto_retrain(threshold=args.threshold, dry_run=not args.live)
    _print_result(res)


if __name__ == "__main__":
    main()
