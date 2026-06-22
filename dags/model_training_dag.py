import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = "BredaUniversityADSAI/2025-26d-fai2-adsai-group-nlp-group-1"
API = f"https://api.github.com/repos/{REPO}/actions/workflows"

# -- DAG 2: Model Training -----------------------------------------------------

with DAG(
    dag_id="emotion_model_training",
    schedule="0 8 * * 1",  # every Monday 08:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["training", "azure-ml"],
    description="Trigger Azure ML training pipeline, preprocess -> train -> register -> deploy green",
) as dag_training:

    trigger_training = BashOperator(
        task_id="trigger_model_training",
        bash_command=f"""
            curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
              -H "Authorization: token {GITHUB_TOKEN}" \\
              -H "Accept: application/vnd.github.v3+json" \\
              {API}/model_training.yml/dispatches \\
              -d '{{"ref":"main","inputs":{{"epochs":"3","batch_size":"16","learning_rate":"2e-5","deployment_name":"green"}}}}' | \\
            grep -q "204" && \\
            echo "Model training triggered successfully" || \\
            echo "Failed to trigger, check GITHUB_TOKEN"
        """,
    )

