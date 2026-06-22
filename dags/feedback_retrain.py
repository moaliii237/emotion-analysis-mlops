import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = "BredaUniversityADSAI/2025-26d-fai2-adsai-group-nlp-group-1"
API = f"https://api.github.com/repos/{REPO}/actions/workflows"

# -- DAG 4: Feedback Retrain ---------------------------------------------------

with DAG(
    dag_id="emotion_feedback_retrain",
    schedule="0 6 * * *",  # daily 06:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "retraining", "feedback"],
    description="Daily feedback flywheel, triggers retraining when user corrections >= 10",
) as dag_feedback:

    trigger_feedback_retrain = BashOperator(
        task_id="trigger_feedback_retrain",
        bash_command=f"""
            curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
              -H "Authorization: token {GITHUB_TOKEN}" \\
              -H "Accept: application/vnd.github.v3+json" \\
              {API}/retrain.yml/dispatches \\
              -d '{{"ref":"main","inputs":{{"dry_run":"false","model_version":"v3"}}}}' | \\
            grep -q "204" && \\
            echo "Feedback retrain triggered successfully" || \\
            echo "Failed to trigger, check GITHUB_TOKEN"
        """,
    )