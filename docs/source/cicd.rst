CI/CD
=====

Branching Strategy
------------------

We follow a four-branch GitFlow strategy:

- ``feature/*``, ``fix/*``, ``chore/*``, development branches (enforced by ``validate-branch.yml``)
- ``develop``, integration branch where features are merged and tested before release
- ``main``, production-ready code, triggers all deployments on merge
- ``deploy``, GitOps release branch, written by CI only, polled by Portainer

The typical flow is: ``feature/x`` -> PR to ``develop`` -> PR to ``main`` -> automated deployment.

Workflows
---------

+-----------------------------+-------------------------------+------------------------------------------+
| Workflow                    | Trigger                       | What it does                             |
+=============================+===============================+==========================================+
| ``validate-branch.yml``     | push / PR                     | Enforces branch naming convention        |
+-----------------------------+-------------------------------+------------------------------------------+
| ``lint.yml``                | push / PR to main or develop  | Runs ``black``, ``isort``, ``flake8``    |
+-----------------------------+-------------------------------+------------------------------------------+
| ``test.yml``                | push / PR to main or develop  | Runs ``pytest`` with coverage            |
+-----------------------------+-------------------------------+------------------------------------------+
| ``deploy.yml``              | push to ``main``              | Builds images -> GHCR -> Portainer        |
+-----------------------------+-------------------------------+------------------------------------------+
| ``deploy_container_apps.yml``| push to ``main``             | Builds images -> Azure Container Apps    |
+-----------------------------+-------------------------------+------------------------------------------+
| ``deploy_model.yml``        | push to ``main`` / manual     | Deploys model to Azure ML endpoint       |
+-----------------------------+-------------------------------+------------------------------------------+
| ``retrain.yml``             | daily 06:00 UTC + manual      | Checks threshold, triggers retraining    |
+-----------------------------+-------------------------------+------------------------------------------+
| ``docs.yml``                | push to ``main``              | Builds Sphinx docs -> GitHub Pages        |
+-----------------------------+-------------------------------+------------------------------------------+

Required GitHub Secrets
-----------------------

+------------------------+-------------------------------------------+
| Secret                 | Description                               |
+========================+===========================================+
| ``DATABASE_URL``       | PostgreSQL connection string              |
+------------------------+-------------------------------------------+
| ``ML_ENDPOINT_KEY``    | Azure ML endpoint key                     |
+------------------------+-------------------------------------------+
| ``GHCR_USERNAME``      | GitHub Container Registry username        |
+------------------------+-------------------------------------------+
| ``PORTAINER_WEBHOOK_URL`` | Portainer stack webhook URL            |
+------------------------+-------------------------------------------+

Required GitHub Variables
--------------------------

+-------------------------+------------------------------------------+
| Variable                | Description                              |
+=========================+==========================================+
| ``AZURE_CLIENT_ID``     | Azure service principal client ID (OIDC) |
+-------------------------+------------------------------------------+
| ``AZURE_TENANT_ID``     | Azure tenant ID (OIDC)                   |
+-------------------------+------------------------------------------+
| ``AZURE_SUBSCRIPTION_ID``| Azure subscription ID (OIDC)            |
+-------------------------+------------------------------------------+

Full CI/CD Flow
---------------

.. code-block:: text

   push to main
       v
   lint.yml        -> black + isort + flake8
   test.yml        -> pytest --cov
       v
   deploy.yml
       +-- build nlp-backend -> push to GHCR
       +-- publish deploy branch -> Portainer polls every 5 min
       +-- webhook -> Portainer instant redeploy

   deploy_container_apps.yml
       +-- build + push -> GHCR
       +-- Azure login (OIDC, no stored secret)
       +-- az containerapp update -> nlp1-backend

   deploy_model.yml (only if azure/ files changed)
       +-- deploy_endpoint.py -> emotion-classifier-endpoint

   retrain.yml (daily 06:00 UTC)
       +-- check feedback threshold (>= 10 corrections?)
       +-- upload new data asset to Azure ML
       +-- submit training pipeline job
