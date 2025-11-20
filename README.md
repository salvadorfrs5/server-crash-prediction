# Preventing Server Crashes Using Tree-Based Models

**Team:** Leilany Rojas, Ammar Salama, Salvador Frias, Mashel Khan

## Topic

Preventing server crashes using tree-based models.

## Research Question

How can AI models predict spikes in server load to prevent slowdowns or crashes, while accounting for variations in workload and server types?

## Sources of Bias

- Server logs often come from a limited set of machines, meaning patterns from less common servers may be underrepresented.
- There are various types of servers and machines, so data that doesn’t consider all of them may be biased.
- Different workloads can affect CPU, memory, and network usage differently, which may not be equally captured in datasets.
- Anomalies such as maintenance events or unexpected traffic may not appear frequently, leading to uneven model performance.

## Summary

Modern web services and APIs rely on stable servers to deliver content reliably. When servers become overloaded (CPU, memory, or network spikes), applications can slow down or crash, impacting user experience and business operations. Predicting these overloads before they happen is important for maintaining uptime and reliability.

Large-scale server outages, like the recent AWS outage, can cause major disruptions for thousands of businesses and users. AWS hosts websites and services for other companies, so when its services crashed due to overload and DNS issues, it caused numerous apps and websites to go down. These failures show the importance of being able to predict and prevent server overloads.

## Machine Learning Algorithm

We will use tree-based models. These models:

- Make decisions by splitting data into steps, similar to a flowchart.
- Handle mixed data easily (e.g., CPU or memory metrics).
- Work well with imbalanced data.

They will help predict server slowdowns more efficiently.

## Datasets

- https://github.com/google/cluster-data
- https://github.com/Azure/AzurePublicDataset
- https://github.com/alibaba/clusterdata/tree/master
- https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.htm
- https://www.kaggle.com/datasets/zoya77/cloud-workload-dataset-for-scheduling-analysis/data **(Currently Using)**
- https://www.kaggle.com/datasets/gagansomashekar/microservices-bottleneck-detection-dataset

## Project Structure

```
server-crash-prediction/
├── data/
│   └── cloud_workload_dataset.csv    # Cloud workload dataset (5000 records)
├── notebooks/
│   └── 01_exploration.ipynb          # Data exploration and model training
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- VS Code with Python and Jupyter extensions (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/server-crash-prediction.git
cd server-crash-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
   - Open `notebooks/01_exploration.ipynb` in VS Code or JupyterLab
   - Select your Python kernel
   - Run cells sequentially

## Implementation

### Current Dataset

We are using the **Cloud Workload Dataset** which contains 5000 job records with the following features:

- **Job Metrics:** CPU Utilization, Memory Consumption, Task Execution Time
- **System Metrics:** System Throughput, Task Waiting Time, Network Bandwidth Utilization
- **Workload Characteristics:** Data Source, Number of Active Users
- **Configuration:** Job Priority, Scheduler Type, Resource Allocation Type
- **Target Variable:** Error Rate (converted to binary classification: high error vs normal)

### Model Approach

1. **Target Variable:** We classify jobs with error rates above the 75th percentile (3.8%) as "high error" situations
2. **Feature Engineering:** One-hot encoding for categorical variables (Data Source, Job Priority, Scheduler Type, Resource Allocation Type)
3. **Model:** Random Forest Classifier with balanced class weights
4. **Evaluation:** Classification metrics, confusion matrix, and feature importance analysis

### Notebook Overview

The `01_exploration.ipynb` notebook includes:

1. **Data Loading:** Load and inspect the cloud workload dataset
2. **Feature Engineering:** Create binary target variable and encode categorical features
3. **Data Preprocessing:** Train-test split and feature standardization
4. **Model Training:** Random Forest classifier with 100 estimators
5. **Evaluation:** Classification report, confusion matrix, and feature importance visualization

## Sources

1. **BBC News – AWS Outage (2024)** – A major DNS error at AWS disrupted over 1,000 services globally, illustrating the risks of centralization and the urgent need for predictive systems to prevent cascading infrastructure failures.
2. **Saxena et al. (2023)** – Advanced ML models like quantum neural networks and ensembles significantly outperformed traditional approaches in predicting cluster overloads using Google workload traces.
3. **Alzahrani et al. (2022)** – Their workload-agnostic ML system achieved high accuracy in predicting job failures tied to overloads across multiple real-world server datasets.
4. **Zhu et al. – CloudProphet (2023)** – CloudProphet used neural networks to predict VM overloads and performance drops with over 90% accuracy across varying server setups.
5. **Lei et al. (2025)** – ML models trained on data center telemetry achieved <10% error in predicting performance and showed strong fairness across server types.
6. **AWS Predictive Scaling (2024)** – Amazon’s predictive scaling service uses ML to proactively forecast and prevent CPU/memory overloads in ECS deployments.
7. **Wen et al. – TempoScale (2024)** – TempoScale combined deep learning with workload pattern decomposition to improve overload prediction accuracy and responsiveness by over 30%.
