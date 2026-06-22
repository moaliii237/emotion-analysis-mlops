# Feedback Loop and Continuous Retraining

This is the data flywheel part (ILO 9.5D). The idea is simple: users rate the
predictions, their corrections get stored, and once there are enough new
corrections the model retrains on the full dataset plus those corrections. It
only goes live if the new model is actually better. I wanted this to run on its
own without me babysitting it.

## Big picture

```text
user rates a prediction (correct or wrong + the right label)
        |  POST /feedback
        v
 feedback table (PostgreSQL, SQLite fallback)
        |
        v  at least 10 new corrections since last retrain? (watermark)
        v yes
 export corrections, upload as Azure ML data asset (emotion-feedback:<ts>)
        |
        v  Azure ML job
 merge emotion-raw (full base) + emotion-feedback (merged in the cloud)
        |
        v
 preprocess, train, evaluate (macro F1)
        |
        v  champion / challenger gate
 new_f1 >= max(0.50, live_f1) ?
   no  -> keep current model
   yes -> register + promote to live endpoint (blue/green)
```

## 1. Collect feedback

`POST /feedback` records every rating in the `feedback` table. When the user
disagrees, they also send the correct label (`user_label`).

| Column | Meaning |
|---|---|
| `sentence` | the input text |
| `predicted_label` / `predicted_conf` | what the model said |
| `user_label` | the human correction (when `is_correct = 0`) |
| `is_correct` | 1 means agreed, 0 means corrected |
| `model_version` | which model produced it |

The DB layer (`feedback/db.py`) uses PostgreSQL in production and falls back to
SQLite for local dev. The `DATABASE_URL` env var decides which one is used.

## 2. Decide when to retrain

`should_retrain(threshold=10)` counts only the new corrections since the last
successful retrain. It does this with a watermark (`corrections_to_id` in the
`retrain_log` table) so the same corrections are never used twice. There is also
an in-flight lock (`is_retrain_in_flight`) so two retrains can not overlap.

## 3. Two ways it fires

| Trigger | How | When |
|---|---|---|
| Automatic | `maybe_trigger_after_feedback()` runs after each `/feedback` | Opt-in via `AUTO_RETRAIN_ENABLED=1` and threshold reached |
| Scheduled / manual | `.github/workflows/retrain.yml` | Daily cron at 06:00 UTC + the Run workflow button |

The automatic path is opt-in on purpose. I did not want a demo to accidentally
launch an expensive Azure job, so it stays off unless the env var is set.

## 4. Train on base + feedback (cloud merge)

The base dataset only lives in Azure (`emotion-raw`), so the merge happens in the
cloud and CI never needs a local copy of the data:

1. Export corrections, upload them as a versioned `emotion-feedback` data asset.
2. Submit an Azure ML job that mounts both `emotion-raw` (full base) and
   `emotion-feedback` (the corrections).
3. `merge_inputs.py` joins them inside the job (a correction overrides the base
   label for the same sentence).
4. preprocess, train, evaluate.

The cloud merge is the part I had to fix. My first version trained on the
corrections only, and the val_f1 came out as 0 because a few dozen sentences is
not enough to learn anything. Once I merged them with the full base dataset I got
a real model again.

## 5. Champion / challenger (only promote if better)

After evaluation the new model is the challenger. It is only promoted when:

```text
new_f1 >= max(MIN_PROMOTE_F1, live_f1)        # floor = 0.50
```

- Passes: register, deploy to a fresh slot, shift 100% of traffic (blue/green,
  no downtime).
- Fails: keep the current model and log the reason. This way a few noisy
  corrections can never push a worse model live, which was my main worry.

## Two code paths

| Path | Module | What it does |
|---|---|---|
| Train-only | `feedback/retrain_trigger.py` | merge, preprocess, train (used by `retrain.yml`) |
| Full flywheel | `feedback/auto_retrain.py` | merge, train, evaluate, F1 gate, promote |

## Key files

| File | Role |
|---|---|
| `feedback/db.py` | feedback storage and watermark bookkeeping |
| `feedback/retrain_trigger.py` | export corrections, upload asset, submit retrain |
| `feedback/merge_inputs.py` | in-cloud merge of base + corrections |
| `feedback/auto_retrain.py` | threshold check, F1 gate, promotion |
| `azure/pipeline.py` | the merge, preprocess, train, evaluate pipeline |
| `.github/workflows/retrain.yml` | scheduled + manual trigger |

## Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `DEFAULT_THRESHOLD` | `10` | new corrections needed to fire |
| `MIN_PROMOTE_F1` | `0.50` | safety floor, never promote below it |
| `AUTO_RETRAIN_ENABLED` | unset | set to `1` to enable auto-fire from `/feedback` |

## Feedback API

| Endpoint | Purpose |
|---|---|
| `POST /feedback` | record a rating or correction |
| `GET /feedback/stats` | agreement rate, per-model counts |
| `GET /feedback/export` | export corrections as CSV |
| `GET /feedback/auto-check` | flywheel progress (corrections vs threshold) |
| `POST /feedback/retrain?dry_run=true` | trigger a retrain (dry-run by default) |

## Run it yourself

```bash
# Safe dry-run: shows the real correction count, no Azure calls, no cost
python -m emotion_mlops.feedback.auto_retrain --dry-run

# Live: retrain and promote if the challenger wins
python -m emotion_mlops.feedback.auto_retrain --live

# Or from GitHub: Actions, Retrain Pipeline, Run workflow
```

## Safety guarantees

- Opt-in auto-fire (`AUTO_RETRAIN_ENABLED`).
- Watermark, so corrections are never reused.
- In-flight lock, so retrains do not overlap.
- F1 gate, so a worse model can never go live.
