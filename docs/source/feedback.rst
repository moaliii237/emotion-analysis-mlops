Feedback & Continuous Retraining
=================================

The application implements a full data flywheel for continuous model improvement.

How It Works
------------

.. code-block:: text

   user clicks thumbs-down and selects correct label
           |
           v
   POST /feedback -> PostgreSQL feedback table
           |
           v
   GitHub Actions: retrain.yml (daily 06:00 UTC)
           |
           v
   check_corrections_threshold (>= 10 new corrections?)
           | yes
           v
   export corrections CSV + merge with raw data
           |
           v
   Azure ML: upload new versioned data asset (emotion-raw vYYYYMMDD)
           |
           v
   Azure ML: submit training pipeline (preprocess -> train -> evaluate)
           |
           v
   register best model -> deploy as green endpoint (blue/green strategy)

Submit Feedback via API
-----------------------

.. code-block:: python

   import requests

   # User confirms prediction is correct
   requests.post('http://localhost:8000/feedback', json={
       'sentence': 'I hate this show!',
       'predicted_label': 'anger',
       'predicted_conf': 0.91,
       'is_correct': True,
       'model_version': 'best_model_v3',
       'source': 'text'
   })

   # User corrects a wrong prediction
   requests.post('http://localhost:8000/feedback', json={
       'sentence': 'Oh wow, I did not see that coming.',
       'predicted_label': 'happiness',
       'predicted_conf': 0.73,
       'is_correct': False,
       'user_label': 'surprise',
       'model_version': 'best_model_v3',
       'source': 'text'
   })

Automated Retraining
---------------------

**1. Daily schedule (GitHub Actions)**

Every day at 06:00 UTC, the ``retrain.yml`` workflow checks whether the feedback
threshold has been reached (>= 10 new user corrections since the last retrain).
If yes, it automatically exports corrections, merges with raw training data,
uploads a new versioned data asset to Azure ML, and submits a training pipeline job.

**2. Manual trigger**

Via GitHub Actions::

   GitHub → Actions → Retrain Pipeline → Run workflow

Via API (dry run):

.. code-block:: python

   requests.post('http://<SERVER_IP>:2002/feedback/retrain?dry_run=true')

Monitor Flywheel Progress
--------------------------

.. code-block:: python

   r = requests.get('http://<SERVER_IP>:2002/feedback/auto-check')
   # {"new_corrections": 13, "threshold": 10, "ready": true, "history": [...]}

Or in the UI: **Features > Monitor** > Retrain Flywheel section.

Feedback Module Reference
--------------------------

.. automodule:: emotion_mlops.feedback.auto_retrain
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: emotion_mlops.feedback.retrain_trigger
   :members:
   :undoc-members:
   :show-inheritance:
