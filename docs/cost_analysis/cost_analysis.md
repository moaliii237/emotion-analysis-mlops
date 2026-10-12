# Cost Analysis

## 1. Introduction

This document evaluates the estimated operational costs of deploying the EmotionScope platform in a production environment.

The objective of this analysis is to assess the financial feasibility of the proposed architecture and demonstrate how cloud resources have been selected and managed in a cost-effective manner. The platform combines machine learning inference, automated deployment, monitoring, storage, and containerized application hosting to provide a scalable emotion classification solution.

Rather than focusing solely on the current educational deployment, this analysis estimates the costs that would be incurred by a client deploying the complete system independently in a commercial environment.

---

## 2. System Architecture Overview

The proposed production architecture consists of the following components:

### Application Layer

* Frontend application
* FastAPI backend
* Azure Container Apps

### Machine Learning Layer

* Azure Machine Learning Workspace
* Model training pipeline
* Model registry
* Managed inference endpoint

### Data Layer

* PostgreSQL database
* Azure Blob Storage

### Monitoring and Operations

* Azure Log Analytics
* Prometheus monitoring
* GitHub Actions CI/CD
* GitHub Container Registry (GHCR)

This architecture separates application hosting, machine learning workloads, persistent storage, and monitoring services, creating a scalable and maintainable deployment structure.

---

## 3. Resource Analysis

### 3.1 Azure Container Apps

**Purpose**

Azure Container Apps host the backend application responsible for:

* Processing user requests
* Communicating with machine learning services
* Managing API interactions
* Returning emotion classification predictions

**Configuration**

Current deployment:

* 0.5 vCPU
* 1 GiB RAM
* North Europe region

**Reason for Selection**

Azure Container Apps provide a serverless deployment model that eliminates the need for maintaining virtual machines while still supporting automatic scaling and containerized deployments.

---

### 3.2 Azure Machine Learning Workspace

**Purpose**

Azure Machine Learning is used for:

* Dataset management
* Experiment tracking
* Model training
* Model registration
* Pipeline orchestration
* Endpoint management

**Reason for Selection**

Azure ML centralizes the machine learning lifecycle and provides enterprise-grade MLOps capabilities that support reproducibility, deployment automation, and model governance.

---
### 3.3 GPU-Enabled Azure Machine Learning Inference Infrastructure

**Purpose**

The EmotionScope platform uses a dedicated GPU-backed Azure Machine Learning endpoint to perform real-time emotion classification. Transformer-based natural language processing models are computationally intensive and benefit significantly from hardware acceleration during inference.

**Configuration**

Evidence collected from the deployed Azure Machine Learning endpoint confirms that the production deployment uses:

* Dedicated GPU instance type
* Minimum instance count: 1
* Maximum instance count: 1
* Azure Machine Learning Managed Online Endpoint
* Autoscaling configuration based on utilization thresholds

In addition, the Azure Kubernetes compute environment provides GPU-enabled compute resources with:

* 4 requested CPU cores
* 32 GiB requested memory
* Up to 32 CPU cores
* Up to 256 GiB memory
* 1 dedicated GPU

**Reason for Selection**

GPU acceleration substantially improves inference performance for transformer-based models and reduces prediction latency compared with CPU-only deployments. Although GPU resources increase infrastructure costs, they enable the platform to deliver real-time predictions while maintaining acceptable response times for production workloads.


### 3.4 Managed Inference Endpoint

**Purpose**

The inference endpoint serves the trained emotion classification model and processes prediction requests in real time.

**Reason for Selection**

Managed endpoints simplify deployment, scaling, security, and monitoring while providing high availability for production workloads.

---

### 3.5 PostgreSQL Database

**Purpose**

The PostgreSQL database stores:

* User submissions
* Prediction results
* Feedback data
* System metadata

**Reason for Selection**

PostgreSQL provides a reliable and scalable relational database solution suitable for transactional application workloads.

---

### 3.6 Azure Blob Storage

**Purpose**

Blob Storage is used for:

* Training datasets
* Model artifacts
* Pipeline outputs
* Evaluation reports

**Reason for Selection**

Blob Storage offers low-cost scalable storage that integrates directly with Azure Machine Learning.

---

### 3.7 Azure Log Analytics

**Purpose**

Azure Log Analytics collects:

* Application logs
* Deployment logs
* Monitoring metrics
* Diagnostic information

**Reason for Selection**

Centralized monitoring improves observability and simplifies troubleshooting.

---

### 3.8 GitHub Actions and GitHub Container Registry

**Purpose**

GitHub Actions automates:

* Testing
* Linting
* Build pipelines
* Deployments

GitHub Container Registry stores Docker images used during deployment.

**Reason for Selection**

The combination of GitHub Actions and GHCR provides an efficient CI/CD workflow while minimizing infrastructure complexity.

---

## 4. Estimated Monthly Cost Breakdown

The following estimates are based on public Azure pricing information for the North Europe region and represent a realistic small-scale production deployment of the EmotionScope platform.

| Resource                         | Configuration                          | Estimated Monthly Cost |
| -------------------------------- | -------------------------------------- | ---------------------- |
| Azure Container App              | 0.5 vCPU, 1 GiB RAM                    | €5-15                  |
| Azure Machine Learning Workspace | Metadata, experiments, registry        | €0-5                   |
| Azure ML Managed Endpoint (CPU)  | CPU-based inference deployment         |€20-80                  | 
| Azure ML Managed Endpoint (GPU)  | GPU-enabled deployment                 | €150-800+              |
| PostgreSQL Database              | Small managed production instance      | €15-50                 |
| Azure Blob Storage               | Datasets, model artifacts, logs        | €1-5                   |
| Azure Log Analytics              | Application monitoring and diagnostics | €1-10                  |
| GitHub Actions                   | CI/CD pipeline execution               | €0-5                   |
| GitHub Container Registry        | Docker image storage                   | €0-5                   |
| Prometheus Monitoring            | Self-hosted monitoring                 | €0-10                  |

The deployed architecture uses a dedicated GPU-backed Azure Machine Learning endpoint for model inference. As a result, GPU compute represents the largest recurring infrastructure cost within the platform. While Azure Container Apps, databases, monitoring services, and storage contribute to operational expenditure, the machine learning endpoint is the primary cost driver due to the dedicated GPU resources required for real-time transformer model inference.

### Estimated Total Monthly Cost

**Baseline CPU-Based Production Deployment:**
€45-185 per month

The CPU-based configuration represents the most cost-effective deployment option and is sufficient for moderate workloads.

### Optional GPU-Accelerated Deployment

GPU-enabled inference can increase operational costs by approximately €150-800+ per month depending on workload requirements, scaling policies, and hardware allocation. GPU deployments significantly increase infrastructure expenditure but may be justified for larger transformer models, higher throughput requirements, or latency-sensitive production environments.

### Estimated Annual Cost

**Baseline CPU-Based Production Deployment:**
€540-2,220 per year

**Optional GPU-Accelerated Deployment:**
€1,800-9,600+ per year

These estimates assume moderate usage levels, a single inference endpoint, and a small production database. Costs can vary depending on traffic volume, endpoint scaling, monitoring retention periods, and storage consumption.

Actual operational costs depend primarily on the number of prediction requests processed by the platform and the computational resources allocated to the machine learning inference service. As user demand increases, additional resources may be provisioned to maintain performance, resulting in higher infrastructure costs.
---

## 5. Performance and Latency Analysis

In addition to financial cost, system responsiveness is an important consideration when evaluating cloud infrastructure. For an emotion classification platform, users expect predictions to be returned in near real time.

### What is Latency?

Latency refers to the time required for a request to travel through the system, be processed, and return a prediction to the user.

For the EmotionScope platform, a prediction request follows the path:

Client Request
-> Azure Container App
-> FastAPI Backend
-> Azure ML Managed Endpoint
-> Prediction Returned

The total response time depends on network communication, backend processing, model inference time, and database interactions.

### Expected Latency Components

| Component | Estimated Latency |
|------------|------------|
| Network Transfer | 20-100 ms |
| FastAPI Processing | 5-20 ms |
| Azure ML Endpoint Inference | 50-300 ms |
| Database Operations | 5-30 ms |
| Total Expected Response Time | 80-450 ms |

### Observed Deployment Performance

In addition to theoretical latency estimates, Azure monitoring data was collected from the deployed backend service. The monitoring dashboard reported an average response time of approximately 1.36 seconds during testing.

While this value is higher than the estimated latency range for a production-optimized deployment, it reflects the performance of the current educational deployment configuration and demonstrates that the system is capable of processing requests reliably using a cost-efficient infrastructure setup. These measurements provide real operational evidence that can be used to evaluate future scaling decisions and determine whether additional compute resources are justified.

The observed resource utilization must be interpreted in the context of the deployed architecture. Azure monitoring showed that backend CPU utilization remained extremely low, averaging approximately 0.29%. This does not indicate that the machine learning workload itself is lightweight. Rather, the computationally intensive model inference operations are executed on a dedicated GPU-backed Azure Machine Learning endpoint. Consequently, the backend primarily functions as an orchestration layer that forwards requests to the GPU inference service and returns prediction results to the client. The low CPU utilization therefore reflects an efficient separation of responsibilities between the application layer and the machine learning infrastructure.



### Cost-Performance Trade-Off

Infrastructure cost and latency are closely related.

The relationship between cost and performance becomes particularly significant when GPU resources are introduced. GPU-enabled deployments can substantially reduce model inference times and training durations but incur considerably higher operational costs than CPU-based configurations. Consequently, GPU resources should only be provisioned when workload requirements justify the additional expenditure.

Lower-cost deployments typically use smaller compute resources and fewer replicas, which may increase response times under heavy load. Larger deployments provide lower latency and greater throughput but incur higher operational costs.

| Deployment Size | Estimated Monthly Cost | Expected Latency |
|-----------------|------------------------|------------------|
| Small Deployment (CPU) | €45-185 | 200-450 ms |
| Medium Deployment (Mixed CPU/GPU) | €150-800 | 100-250 ms |
| Enterprise Deployment (GPU Accelerated) | €800+ | <100 ms |


### Scalability Impact

As the number of requests increases, additional resources may be provisioned automatically through Azure scaling mechanisms.

Potential scaling actions include:

- Additional backend replicas
- Increased endpoint compute capacity
- Larger database instances
- Expanded monitoring resources

These measures help maintain acceptable response times while supporting larger workloads.

### Interpretation

The expected latency values indicate that the proposed architecture is suitable for real-time emotion classification applications. Even under a small deployment configuration, users can generally expect predictions to be returned in less than half a second.

This demonstrates that the selected cloud infrastructure provides a reasonable balance between operational cost and system responsiveness.

## 6. Cost Optimisation Decisions

Several architectural decisions were made specifically to reduce operational expenditure while maintaining scalability and maintainability.

## Monitoring-Based Cost Validation

To validate the proposed infrastructure configuration, Azure monitoring metrics were analysed following deployment.

CPU utilization remained very low, averaging approximately 0.29%, indicating that the backend service was not CPU constrained. This suggests that the selected compute allocation is sufficient for the current workload and that additional CPU resources would provide limited performance benefits relative to their cost.

Memory utilization remained significantly higher at approximately 90.8%, indicating that memory represents the primary resource constraint within the current deployment. Future scaling decisions should therefore prioritise memory allocation before increasing CPU capacity.

Response-time monitoring showed average response times of approximately 1.36 seconds. While acceptable for the current workload, these metrics provide a basis for evaluating future scaling decisions and determining whether additional compute resources are justified.

By analysing actual infrastructure usage rather than relying solely on theoretical estimates, the deployment demonstrates a data-driven approach to cost optimisation and capacity planning.

### GPU Deployment Evidence

Azure Machine Learning endpoint configuration confirms that the deployed emotion-classifier-endpoint uses a dedicated GPU instance type with a minimum replica count of one. This provides direct evidence that GPU resources are incorporated into the production architecture and validates their inclusion within the financial evaluation. The GPU endpoint enables hardware-accelerated inference while supporting autoscaling policies and managed deployment through Azure Machine Learning services.


### Azure Container Apps Instead of Virtual Machines

The backend is deployed using Azure Container Apps rather than dedicated virtual machines.

Benefits:

* Reduced idle infrastructure costs
* Automatic scaling capabilities
* Simplified deployment management
* No operating system maintenance requirements

### Lightweight Backend Configuration

The current deployment uses:

* 0.5 vCPU
* 1 GiB RAM

This allocation is sufficient for handling API requests and communication with the machine learning endpoint while minimizing infrastructure expenditure.

### Dedicated GPU Inference Infrastructure

The deployed system uses a dedicated GPU-backed Azure Machine Learning endpoint for production inference. The GPU endpoint is responsible for executing transformer-based emotion classification models and represents the primary computational resource within the architecture.

Although GPU resources increase operational expenditure compared with CPU-only deployments, they provide substantial improvements in inference speed and throughput. This trade-off was considered acceptable because real-time prediction performance is a core requirement of the platform.

Monitoring data demonstrates that the application backend remains lightly used while the machine learning workload is isolated to the dedicated inference infrastructure. This separation improves scalability and allows future resource allocation decisions to be made independently for the application and machine learning layers.



### Managed Machine Learning Infrastructure

Azure Machine Learning provides:

* Experiment tracking
* Model versioning
* Model registration
* Managed endpoint deployment

Using managed services reduces operational complexity and eliminates the need to maintain custom model-serving infrastructure.

### Blob Storage for Artifact Management

Datasets, trained models, evaluation outputs, and deployment artifacts are stored in Azure Blob Storage.

Benefits:

* Low storage costs
* High durability
* Native Azure ML integration

### Automated CI/CD Pipeline

GitHub Actions automates:

* Code quality checks
* Testing
* Container image creation
* Deployment

This reduces manual deployment effort and lowers operational overhead.

### Modular Architecture

The platform separates:

* Application hosting
* Model serving
* Data storage
* Monitoring

This allows each component to scale independently, ensuring infrastructure costs remain proportional to demand.

---

## 7. Scalability Considerations

The architecture is designed to support gradual scaling as user demand increases.

### Small Deployment

Suitable for:

* Demonstrations
* Research projects
* Internal tools
* Small user groups

Estimated cost:

€45-185/month

### Medium Deployment

Additional resources:

* Multiple backend replicas
* Larger database instance
* Increased endpoint capacity
* Longer monitoring retention

Estimated cost:

€150-800/month

### Enterprise Deployment

Additional resources:

* Multiple inference replicas
* High-availability deployment
* Advanced monitoring
* Increased storage capacity
* Higher request throughput

Estimated cost:

€800+/month

Because Azure services scale independently, infrastructure expenses can grow incrementally rather than requiring significant upfront investment.

---

## 8. Deployment Status

The EmotionScope platform has been successfully deployed using a containerized cloud architecture.

The backend service is hosted on Azure Container Apps and is configured to communicate with a machine learning inference endpoint through environment variables. Automated deployment is managed through GitHub Actions, while monitoring and logging are integrated through Azure services.

The deployed architecture demonstrates a complete MLOps workflow consisting of:

* Containerized application hosting
* Automated CI/CD
* Model serving
* Monitoring and observability
* Scalable cloud infrastructure

The cost estimates presented in this document reflect a realistic production deployment scenario for an independent client environment.

---

## 9. Conclusion

The financial evaluation demonstrates that the proposed architecture is capable of supporting both cost-efficient and high-performance deployment scenarios. Monitoring evidence indicates that the current deployment is not CPU constrained and that infrastructure resources have been allocated conservatively, supporting the objective of minimizing operational expenditure while maintaining acceptable performance.

The architecture provides a clear scaling path through the use of managed Azure services, containerized deployment, automated monitoring, and optional GPU acceleration. This enables operational costs to remain proportional to demand while allowing the platform to expand as user workloads increase.

Overall, the analysis demonstrates that the EmotionScope platform is financially feasible, scalable, and capable of supporting production deployment without requiring excessive infrastructure investment.
