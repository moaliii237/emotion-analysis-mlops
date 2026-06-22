import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = "BredaUniversityADSAI/2025-26d-fai2-adsai-group-nlp-group-1"
API = f"https://api.github.com/repos/{REPO}/actions/workflows"

# -- DAG 3: Full Retrain -------------------------------------------------------

with DAG(
    dag_id="emotion_full_retrain",
    schedule="0 0 * * 0",  # every Sunday 00:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "retraining", "full"],
    description="Full weekly retraining: data preprocessing + model training + registration + deployment",
) as dag_full_retrain:

    trigger_preprocess = BashOperator(
        task_id="trigger_preprocessing",
        bash_command=f"""
            curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
              -H "Authorization: token {GITHUB_TOKEN}" \\
              -H "Accept: application/vnd.github.v3+json" \\
              {API}/data_preprocessing.yml/dispatches \\
              -d '{{"ref":"main"}}' | grep -q "204" && \\
            echo "Preprocessing triggered" || echo "Failed"
            sleep 300
        """,
    )

    trigger_train = BashOperator(
        task_id="trigger_training",
        bash_command=f"""
            curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
              -H "Authorization: token {GITHUB_TOKEN}" \\
              -H "Accept: application/vnd.github.v3+json" \\
              {API}/model_training.yml/dispatches \\
              -d '{{"ref":"main","inputs":{{"epochs":"3","batch_size":"16","deployment_name":"green"}}}}' | \\
            grep -q "204" && \\
            echo "Training triggered" || echo "Failed"
        """,
    )

    trigger_preprocess >> trigger_train
