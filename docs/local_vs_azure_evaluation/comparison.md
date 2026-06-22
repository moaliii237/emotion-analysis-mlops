# Model Comparison, Local vs Azure ML

## Parameters

Both runs used **identical parameters** to ensure a fair comparison.

**Model:** `distilbert-base-uncased`  
**Dataset:** GoEmotions (58k rows, 7 emotion labels)  
**Split:** 70/15/15, train: 40,607 / val: 8,702 / test: 8,702

| Parameter | Value |
|---|---|
| epochs | 1 |
| batch_size | 4 |
| max_steps | 100 |
| learning_rate | 2e-5 |
| max_length | 128 |
| gradient_accumulation_steps | 2 |
| seed | 42 |

---

## Local Run

**Command:**
```powershell
python -m emotion_mlops.models.model_training `
    --data data/processed/ `
    --output models/comparison_run/ `
    --epochs 1 `
    --batch-size 4 `
    --max-steps 100 `
    --learning-rate 2e-5 `
    --run-name "comparison-baseline"
```

**Hardware:** Local CPU (Windows 10)

**Results:**

| Metric | Value |
|---|---|
| F1 (macro) | 0.0957 |
| Accuracy | 0.4538 |

**Classification report:**

```
              precision    recall  f1-score   support

       anger       0.00      0.00      0.00      1371
     disgust       0.00      0.00      0.00        93
        fear       0.00      0.00      0.00       119
   happiness       0.45      1.00      0.62      3896
     neutral       0.63      0.02      0.05      2415
     sadness       0.00      0.00      0.00       424
     surprise       0.00      0.00      0.00       384

    accuracy                           0.45      8702
   macro avg       0.15      0.15      0.10      8702
weighted avg       0.38      0.45      0.29      8702
```

---

## Azure ML Run

The same pipeline with identical parameters was submitted to Azure ML using `pipeline.py` and ran on the `lambda-2` Kubernetes compute target.
Link to the training, evaluation job: 
https://ml.azure.com/runs/lime_guitar_01gh35pt9v?wsid=%2Fsubscriptions%2F<AZURE_SUBSCRIPTION_ID>%2FresourceGroups%2F<AZURE_RESOURCE_GROUP>%2Fproviders%2FMicrosoft.MachineLearningServices%2Fworkspaces%2F<AZURE_WORKSPACE_NAME>&tid=<AZURE_TENANT_ID>&nsq=#

Evaluation results: 
https://ml.azure.com/data/azureml_093257aa-4c80-4c71-baa6-7f5eb5760c83_output_data_eval_output/1/details?wsid=/subscriptions/<AZURE_SUBSCRIPTION_ID>/resourceGroups/<AZURE_RESOURCE_GROUP>/providers/Microsoft.MachineLearningServices/workspaces/<AZURE_WORKSPACE_NAME>&tid=<AZURE_TENANT_ID>#exploreTab

**Results:**

| Metric | Value |
|---|---|
| F1 (macro) | 0.0889 |
| Accuracy | 0.4481 |

**Classification report:**

```
              precision    recall  f1-score   support

       anger       0.00      0.00      0.00      1371
     disgust       0.00      0.00      0.00        93
        fear       0.00      0.00      0.00       119
   happiness       0.45      1.00      0.62      3896
     neutral       0.67      0.00      0.00      2415
     sadness       0.00      0.00      0.00       424
     surprise       0.00      0.00      0.00       384

    accuracy                           0.45      8702
   macro avg       0.16      0.14      0.09      8702
weighted avg       0.39      0.45      0.28      8702
```

---

## Comparison

| Metric | Local | Azure ML | Difference |
|---|---|---|---|
| F1 (macro) | 0.0957 | 0.0889 | -0.0068 |
| Accuracy | 0.4538 | 0.4481 | -0.0057 |

## Analysis

Both runs produce comparable results, the difference in F1 (0.007) and accuracy (0.006) is within expected variance for a model trained with only 100 steps. Neither run has fully converged, which explains the low F1, the model predicts almost exclusively `happiness` (the dominant class) after only 100 training steps.

Key observations:
- Both environments reproduce the same behaviour: model collapses to majority class (`happiness`) at 100 steps
- The small difference between local and cloud results is expected due to floating-point non-determinism across different hardware
- Full training (3 epochs, no step limit) would significantly improve F1 across all emotion classes

This confirms the pipeline is reproducible across local and cloud environments. 
