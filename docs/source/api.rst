API
===

The FastAPI backend exposes the pipeline over HTTP.

Start the Backend Locally
--------------------------

.. code-block:: powershell

   $env:PYTHONPATH="src"
   $env:ML_ENDPOINT_URL="https://adsai2.ngrok.dev/port-31785/api/v1/endpoint/emotion-classifier-endpoint/score"
   $env:ML_ENDPOINT_KEY="your-key-here"
   $env:DATABASE_URL="postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope"
   python -m uvicorn emotion_mlops.api.app:app --port 8000 --reload

Open http://localhost:8000/docs for the Swagger UI.

Endpoints
---------

+--------+---------------------------+--------------------------------------------------+
| Method | Path                      | Description                                      |
+========+===========================+==================================================+
| GET    | /                         | Frontend UI (static HTML)                        |
+--------+---------------------------+--------------------------------------------------+
| GET    | /api                      | API info and available endpoints                 |
+--------+---------------------------+--------------------------------------------------+
| GET    | /health                   | Model load status and version                    |
+--------+---------------------------+--------------------------------------------------+
| GET    | /endpoint/info            | Azure ML endpoint traffic split and URI          |
+--------+---------------------------+--------------------------------------------------+
| GET    | /predictions              | Recent predictions from database                 |
+--------+---------------------------+--------------------------------------------------+
| POST   | /predict/text             | Predict emotion from a single sentence           |
+--------+---------------------------+--------------------------------------------------+
| POST   | /predict/text/full        | Predict emotion with all label probabilities     |
+--------+---------------------------+--------------------------------------------------+
| POST   | /predict/cloud            | Predict via Azure ML Online Endpoint             |
+--------+---------------------------+--------------------------------------------------+
| POST   | /predict/pipeline         | Full pipeline: audio/CSV -> emotions             |
+--------+---------------------------+--------------------------------------------------+
| POST   | /predict/youtube          | Full pipeline from YouTube URL                   |
+--------+---------------------------+--------------------------------------------------+
| POST   | /feedback                 | Record user feedback on a prediction             |
+--------+---------------------------+--------------------------------------------------+
| GET    | /feedback/stats           | Aggregated feedback statistics                   |
+--------+---------------------------+--------------------------------------------------+
| GET    | /feedback/export          | Export user corrections for retraining           |
+--------+---------------------------+--------------------------------------------------+
| GET    | /feedback/auto-check      | Flywheel progress (corrections vs threshold)     |
+--------+---------------------------+--------------------------------------------------+
| POST   | /feedback/retrain         | Trigger retraining from feedback                 |
+--------+---------------------------+--------------------------------------------------+
| POST   | /summarize                | Summarize transcript text                        |
+--------+---------------------------+--------------------------------------------------+
| POST   | /ask                      | Q&A about transcript using Groq LLM              |
+--------+---------------------------+--------------------------------------------------+

Examples
--------

.. code-block:: python

   import requests, json

   # Predict via cloud endpoint
   r = requests.post('http://localhost:8000/predict/text/full',
       json={'text': 'I am so happy today!'})
   print(json.dumps(r.json(), indent=2))

   # Submit feedback
   r = requests.post('http://localhost:8000/feedback', json={
       'sentence': 'I am so happy today!',
       'predicted_label': 'happiness',
       'predicted_conf': 0.98,
       'is_correct': True,
       'model_version': 'best_model_v3',
       'source': 'text'
   })

   # Check flywheel progress
   r = requests.get('http://localhost:8000/feedback/auto-check')
   print(r.json())
   # {"new_corrections": 13, "threshold": 10, "ready": true}

API Module Reference
---------------------

.. automodule:: emotion_mlops.api.app
   :members:
   :undoc-members:
   :show-inheritance:
