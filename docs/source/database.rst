Database
========

All predictions and feedback are persisted in PostgreSQL automatically.

Predictions Table
-----------------

Every call to ``/predict/cloud`` and ``/predict/text`` logs a row:

+-------------+-----------+-----------------------------------+
| Column      | Type      | Description                       |
+=============+===========+===================================+
| id          | SERIAL    | Auto-incremented primary key      |
+-------------+-----------+-----------------------------------+
| timestamp   | TIMESTAMP | UTC time of prediction            |
+-------------+-----------+-----------------------------------+
| text        | TEXT      | Input sentence                    |
+-------------+-----------+-----------------------------------+
| emotion     | VARCHAR   | Predicted emotion label           |
+-------------+-----------+-----------------------------------+
| confidence  | FLOAT     | Model confidence (0-1)            |
+-------------+-----------+-----------------------------------+
| source      | VARCHAR   | ``local`` or ``azure-ml-endpoint``|
+-------------+-----------+-----------------------------------+

Feedback Table
--------------

Every call to ``/feedback`` logs a row:

+-----------------+-----------+---------------------------------------+
| Column          | Type      | Description                           |
+=================+===========+=======================================+
| id              | SERIAL    | Auto-incremented primary key          |
+-----------------+-----------+---------------------------------------+
| timestamp       | TIMESTAMP | UTC time of feedback                  |
+-----------------+-----------+---------------------------------------+
| sentence        | TEXT      | Original input sentence               |
+-----------------+-----------+---------------------------------------+
| predicted_label | VARCHAR   | What the model predicted              |
+-----------------+-----------+---------------------------------------+
| predicted_conf  | FLOAT     | Model confidence                      |
+-----------------+-----------+---------------------------------------+
| user_label      | VARCHAR   | User's corrected label                |
+-----------------+-----------+---------------------------------------+
| is_correct      | INTEGER   | 1 if user agreed, 0 if not            |
+-----------------+-----------+---------------------------------------+
| model_version   | VARCHAR   | Which model produced the prediction   |
+-----------------+-----------+---------------------------------------+
| source          | VARCHAR   | ``audio``, ``transcript``, or ``text``|
+-----------------+-----------+---------------------------------------+

Local Development Database
---------------------------

.. code-block:: bash

   docker-compose up -d postgres

Connection string::

   DATABASE_URL=postgresql://nlp1_user:nlp1_password@localhost:5432/emotionscope

Azure Managed Postgres
-----------------------

Shared BUas server: ``pg-students-y2d-2026.postgres.database.azure.com``

The ``DATABASE_URL`` is stored as a secret in Azure Container Apps and as an
environment variable in Portainer, never in code or git.

.. code-block:: powershell

   az containerapp secret set `
     --name nlp1-backend `
     --resource-group buas-y2 `
     --secrets db-url="YOUR_CONNECTION_STRING"

   az containerapp update `
     --name nlp1-backend `
     --resource-group buas-y2 `
     --set-env-vars DATABASE_URL=secretref:db-url

Database Module Reference
--------------------------

.. automodule:: emotion_mlops.api.database
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: emotion_mlops.feedback.db
   :members:
   :undoc-members:
   :show-inheritance:
