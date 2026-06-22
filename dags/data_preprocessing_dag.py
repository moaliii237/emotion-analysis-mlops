import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = "BredaUniversityADSAI/2025-26d-fai2-adsai-group-nlp-group-1"
API = f"https://api.github.com/repos/{REPO}/actions/workflows"

# -- DAG 1: Data Preprocessing ------------------------------------------------

with DAG(
    dag_id="emotion_data_preprocessing",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data", "preprocessing"],
    description="Preprocess GoEmotions data and upload to Azure ML as versioned data asset",
) as dag_preprocessing:

    trigger_preprocessing = BashOperator(
        task_id="trigger_data_preprocessing",
        bash_command=f"""
            curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
              -H "Authorization: token {GITHUB_TOKEN}" \\
              -H "Accept: application/vnd.github.v3+json" \\
              {API}/data_preprocessing.yml/dispatches \\
              -d '{{"ref":"main"}}' | grep -q "204" && \\
            echo "Data preprocessing triggered successfully" || \\
            echo "Failed to trigger, check GITHUB_TOKEN"
        """,
    )