Setup
=====

Prerequisites
-------------

- Python 3.11
- Docker Desktop
- Azure CLI (``az login`` with BUas credentials)
- Git

Installation
------------

.. code-block:: bash

   git clone https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-nlp-group-1.git
   cd 2025-26d-fai2-adsai-group-nlp-group-1

   python -m venv .venv311

   # Windows
   .\.venv311\Scripts\Activate.ps1

   # macOS/Linux
   source .venv311/bin/activate

   # 1. Install PyTorch (CPU)
   pip install torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cpu \
       --only-binary=:all:

   # 2. Install production dependencies
   pip install -r requirements.txt

   # 3. Install dev dependencies
   pip install -r requirements-dev.txt

   # 4. Install Whisper
   pip install openai-whisper

   # 5. Install the package in editable mode
   pip install -e .

Install from wheel
------------------

.. code-block:: bash

   pip install dist/emotion_mlops-0.1.0-py3-none-any.whl

Environment Variables
---------------------

Create a ``.env`` file (never commit) or set in your shell:

.. code-block:: bash

   export ML_ENDPOINT_URL="https://adsai2.ngrok.dev/port-31785/api/v1/endpoint/emotion-classifier-endpoint/score"
   export ML_ENDPOINT_KEY="your-key-here"
   export DATABASE_URL="postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope"

Windows PowerShell:

.. code-block:: powershell

   $env:ML_ENDPOINT_URL="https://adsai2.ngrok.dev/port-31785/api/v1/endpoint/emotion-classifier-endpoint/score"
   $env:ML_ENDPOINT_KEY="your-key-here"
   $env:DATABASE_URL="postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope"

Azure Config
------------

Create ``config.json`` at the repo root (do not commit):

.. code-block:: json

   {
     "subscription_id": "<AZURE_SUBSCRIPTION_ID>",
     "resource_group": "buas-y2",
     "workspace_name": "team-NLP1-2026"
   }

CLI
---

Three entry points are available after ``pip install -e .``:

.. code-block:: bash

   # Run the full pipeline on an audio file
   emotion-pipeline \
       --config configs/config.yaml \
       --audio data/audio.wav \
       --out results.csv

   # Run on an existing transcript CSV
   emotion-pipeline \
       --config configs/config.yaml \
       --transcript_csv data/data_test/transcript.csv \
       --out results.csv

Without installing the package:

.. code-block:: powershell

   $env:PYTHONPATH="src"
   python -m emotion_mlops.cli.infer `
       --config configs/config.yaml `
       --transcript_csv data/data_test/transcript.csv `
       --out results.csv

Docker
------

Local development (all services):

.. code-block:: bash

   docker-compose up -d

Services:

+-------------+---------------------------+---------------------------+
| Service     | URL                       | Description               |
+=============+===========================+===========================+
| backend     | http://localhost:2002     | FastAPI API + Frontend    |
| postgres    | localhost:5432            | PostgreSQL database       |
| prometheus  | http://localhost:2003     | Metrics scraper           |
| grafana     | http://localhost:2004     | Metrics dashboard         |
| airflow     | http://localhost:2005     | Pipeline orchestration    |
| mlflow      | http://localhost:2006     | Experiment tracking       |
+-------------+---------------------------+---------------------------+

Airflow
-------

Run locally:

.. code-block:: powershell

   docker run -p 2005:8080 `
     -v ${PWD}/dags:/opt/airflow/dags `
     -e AIRFLOW__CORE__EXECUTOR=SequentialExecutor `
     -e AIRFLOW__CORE__LOAD_EXAMPLES=False `
     -e _AIRFLOW_WWW_USER_CREATE=true `
     -e _AIRFLOW_WWW_USER_USERNAME=admin `
     -e _AIRFLOW_WWW_USER_PASSWORD=admin `
     apache/airflow:2.9.0 standalone

.. note::

   Airflow standalone generates a random password on first run if the user already exists.
   Check the container logs for the actual password::

       standalone | Login with username: admin  password: <generated>

   Or reset manually:

   .. code-block:: bash

      docker exec -it <container_id> airflow users reset-password -u admin -p admin

DAGs:

+------------------------------+----------------+------------------------------------------+
| DAG                          | Schedule       | Description                              |
+==============================+================+==========================================+
| emotion_data_preprocessing   | manual         | Preprocess GoEmotions CSV                |
| emotion_model_training       | Monday 08:00   | Trigger Azure ML training pipeline       |
| emotion_full_retrain         | Sunday 00:00   | Full retraining pipeline                 |
| emotion_feedback_retrain     | Daily 06:00    | Retrain when corrections >= 10           |
+------------------------------+----------------+------------------------------------------+
