Deployment
==========

On-Premise (Portainer)
-----------------------

The application runs on the BUas Lambda server (``<SERVER_IP>``) via Portainer.

Services:

+-----------------------------+-------------------------------------+---------------+
| Service                     | URL                                 | Credentials   |
+=============================+=====================================+===============+
| Frontend + Backend API      | http://<SERVER_IP>:2002             |               |
+-----------------------------+-------------------------------------+---------------+
| API Docs (Swagger)          | http://<SERVER_IP>:2002/docs        |               |
+-----------------------------+-------------------------------------+---------------+
| Prometheus                  | http://<SERVER_IP>:2003             |               |
+-----------------------------+-------------------------------------+---------------+
| Grafana                     | http://<SERVER_IP>:2004             | admin / admin |
+-----------------------------+-------------------------------------+---------------+
| Airflow                     | http://<SERVER_IP>:2005             | admin / admin |
+-----------------------------+-------------------------------------+---------------+
| Portainer                   | http://<SERVER_IP>:3085             |               |
+-----------------------------+-------------------------------------+---------------+

Automated deployment via GitOps polling. Portainer polls the ``deploy`` branch every 5 minutes.

Cloud (Azure Container Apps)
-----------------------------

+------------------+------------------------------------------------------------------------+
| Service          | URL                                                                    |
+==================+========================================================================+
| Backend + UI     | https://nlp1-backend.politeriver-95d366e5.northeurope.azurecontainerapps.io |
+------------------+------------------------------------------------------------------------+
| API Docs         | https://nlp1-backend.politeriver-95d366e5.northeurope.azurecontainerapps.io/docs |
+------------------+------------------------------------------------------------------------+

.. note::

   Azure Container Apps uses the NGROK tunnel to reach the Kubernetes endpoint:
   ``https://adsai2.ngrok.dev/port-31785/api/v1/endpoint/emotion-classifier-endpoint/score``

Azure ML Endpoint
-----------------

| Resource       | Value                                      |
|----------------|--------------------------------------------|
| Endpoint name  | ``emotion-classifier-endpoint``            |
| Compute        | Kubernetes lambda-2                        |
| Traffic        | blue: 70%, green: 30%                      |
| Auth           | Bearer token (key-based)                   |

Blue-Green Deployment
---------------------

.. code-block:: powershell

   # Deploy new version as green
   python src/emotion_mlops/azure/deploy_endpoint.py --model-version 2 --deployment-name green

   # Canary: 10% traffic to green
   python src/emotion_mlops/azure/deploy_endpoint.py --set-traffic blue=90 green=10

   # Promote: full traffic to green
   python src/emotion_mlops/azure/deploy_endpoint.py --set-traffic blue=0 green=100

   # Rollback: back to blue
   python src/emotion_mlops/azure/deploy_endpoint.py --set-traffic blue=100 green=0

Azure Module Reference
-----------------------

.. automodule:: emotion_mlops.azure.deploy_endpoint
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: emotion_mlops.azure.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: emotion_mlops.azure.register_best
   :members:
   :undoc-members:
   :show-inheritance:
