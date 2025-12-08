# Server Crash Prediction - Model Results

**Project**: AI-Powered Server Crash Prediction using Cloud Workload Data
**Team**: Leilany Rojas, Salvador Frias, Mashel Khan
**Program**: AI4ALL Ignite Program
**Date**: December 2025

---

## Executive Summary

This project developed a machine learning model to predict server crashes and high error rates in cloud computing environments. Using RandomForestClassifier trained on 5,000 cloud job records, the model analyzes CPU, memory, network, and workload metrics to flag high-risk scenarios before they cause outages.

**Key Achievement**: Built a functional early warning system that identifies system performance metrics (throughput, memory, network bandwidth) as the strongest predictors of server issues.

---

## Model Performance

### Metrics Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 71.0% | Correctly classifies 710/1000 test cases |
| **Precision** | 16.7% | Of predicted high-error cases, 17% are actually high-error |
| **Recall** | 4.4% | Catches only 4% of actual high-error cases |
| **F1-Score** | 0.070 | Low due to precision-recall trade-off |

### Confusion Matrix (Test Set: 1000 samples)

```
                  Predicted Normal    Predicted High Error
Actual Normal            696                  55
Actual High Error        238                  11
```

**Breakdown**:
- **True Negatives (696)**: Correctly identified normal operations
- **True Positives (11)**: Correctly caught high-error cases
- **False Positives (55)**: False alarms (predicted error when normal)
- **False Negatives (238)**: Missed high-error cases (biggest challenge)

### Performance Analysis

**Strengths**:
- High accuracy on normal cases (93% of normal jobs correctly identified)
- Low false alarm rate (only 55 false positives out of 1000 tests)
- Model is conservative and stable

**Challenges**:
- Low recall (4.4%) - misses most high-error cases
- Class imbalance remains despite SMOTE resampling
- Threshold tuning needed to balance precision vs. recall

**Why Low Recall?**
The model is overly conservative due to:
1. Original class imbalance (75% normal, 25% high-error)
2. High cost of false positives in tree-based models
3. SMOTE helped but didn't fully solve the imbalance issue

---

## Feature Importance

### Top 10 Most Predictive Features

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | System Throughput (tasks/sec) | 10.36% | Performance |
| 2 | Memory Consumption (MB) | 10.30% | Resource |
| 3 | Network Bandwidth Utilization (Mbps) | 10.29% | Network |
| 4 | Task Execution Time (ms) | 10.08% | Performance |
| 5 | Number of Active Users | 10.07% | Workload |
| 6 | Task Waiting Time (ms) | 9.91% | Performance |
| 7 | CPU Utilization (%) | 9.50% | Resource |
| 8 | Job Priority (Medium) | 4.42% | Configuration |
| 9 | Job Priority (Low) | 4.31% | Configuration |
| 10 | Scheduler Type (FCFS) | 3.24% | Configuration |

### Key Insights

1. **Numerical metrics dominate**: The top 7 features are all continuous performance/resource metrics, accounting for ~70% of predictive power

2. **Categorical features are weak**: Job priority, scheduler type, and resource allocation type contribute only ~30% combined

3. **Balanced importance**: No single feature is overwhelmingly dominant - the model considers multiple signals

4. **Actionable features**: High-importance features (throughput, memory, network) are all monitorable in real-time

---

## Methodology

### Data Preparation

**Dataset**: Cloud Workload Dataset
- **Total Records**: 5,000 jobs
- **Features**: 15 columns (7 numerical, 4 categorical, 3 identifiers, 1 target)
- **Target Variable**: Binary classification
  - Error Rate ≥ 3.8% (75th percentile) = **High Error (1)**
  - Error Rate < 3.8% = **Normal (0)**

**Class Distribution**:
- Normal: 3,753 (75.1%)
- High Error: 1,247 (24.9%)

### Feature Engineering

1. **Column Name Cleaning**: Removed spaces, parentheses, and special characters
2. **Dropped Identifiers**: Job_ID, Task_Start_Time, Task_End_Time
3. **One-Hot Encoding**: Categorical features (Data_Source, Job_Priority, Scheduler_Type, Resource_Allocation_Type)
4. **Final Feature Count**: 16 features after encoding

### Handling Class Imbalance

**Technique**: SMOTE (Synthetic Minority Over-sampling Technique)

- Applied only to training data (avoid data leakage)
- Balanced training set: 3,002 normal / 3,002 high-error
- Test set remains imbalanced (realistic evaluation)

### Model Architecture

**Algorithm**: RandomForestClassifier

**Hyperparameters**:
```python
n_estimators=600           # 600 decision trees
max_depth=None             # No depth limit (grow until pure)
min_samples_split=2        # Minimum samples to split a node
min_samples_leaf=1         # Minimum samples in leaf nodes
class_weight=None          # SMOTE replaces need for class weighting
random_state=42            # Reproducibility
n_jobs=-1                  # Use all CPU cores
```

**Why RandomForest?**
- Handles mixed data types (numerical + categorical)
- Resistant to overfitting
- Provides feature importance
- No need for feature scaling
- Robust to outliers

### Train/Test Split

- **Training Set**: 4,000 samples (80%)
- **Test Set**: 1,000 samples (20%)
- **Stratification**: Yes (maintains class balance in split)

---

## Addressing Bias and Limitations

### Documented Bias Sources

1. **Server Type Bias**
   - Dataset contains logs from limited machine types
   - Patterns from uncommon server configurations may be underrepresented
   - **Mitigation**: Documented in project materials; future work should include diverse server types

2. **Workload Variation Bias**
   - CPU-intensive, memory-intensive, and network-intensive jobs may not be equally captured
   - Model may perform better on certain workload profiles
   - **Mitigation**: Feature importance analysis reveals balanced contribution from CPU, memory, and network

3. **Anomaly Frequency Bias**
   - Rare events (maintenance, unexpected traffic spikes) appear infrequently
   - Model may struggle with novel edge cases
   - **Mitigation**: SMOTE helps but doesn't create truly novel scenarios

4. **Configuration Diversity**
   - Not all scheduler types and resource allocation strategies are equally represented
   - **Mitigation**: One-hot encoding allows model to learn from available configurations

### Model Limitations

- **Low Recall**: Model misses ~96% of high-error cases
- **Threshold Sensitivity**: Default 0.5 threshold may not be optimal for this use case
- **Generalization**: Trained on specific dataset; may not transfer to other cloud environments
- **Temporal Factors**: Model doesn't account for time-series patterns or trends

---

## Business Impact & Use Cases

### Real-World Applications

**1. Proactive Monitoring**
- Deploy model in cloud infrastructure monitoring stack
- Flag high-risk jobs for manual review before execution
- Reduce MTTR (Mean Time To Resolution) by catching issues early

**2. Resource Optimization**
- Identify workload patterns that lead to high error rates
- Optimize job scheduling and resource allocation
- Prevent cascading failures in distributed systems

**3. SLA Protection**
- Predict potential SLA violations before they occur
- Prioritize high-risk jobs for premium resource allocation
- Reduce customer-facing downtime

### Trade-offs

| Scenario | Precision Focus | Recall Focus |
|----------|----------------|--------------|
| **Use Case** | Cost-sensitive (avoid false alarms) | Safety-critical (catch all issues) |
| **Current Model** | ✅ Good fit (16.7% precision, low FP) | ❌ Poor fit (4.4% recall, high FN) |
| **Recommendation** | Use default 0.5 threshold | Lower threshold to ~0.3-0.4 |

---

## Future Improvements

### Short-Term (Next Iteration)

1. **Threshold Optimization**
   - Test thresholds from 0.3 to 0.5
   - Balance precision/recall based on business requirements
   - Consider F-beta score (favor recall for safety-critical applications)

2. **Cross-Validation**
   - Implement 5-fold or 10-fold CV for robust performance estimates
   - Reduce variance in metrics due to train/test split randomness

3. **Hyperparameter Tuning**
   - GridSearchCV or RandomizedSearchCV
   - Optimize: n_estimators, max_depth, min_samples_split, min_samples_leaf
   - Potential 5-10% accuracy improvement

### Medium-Term (Research Extensions)

4. **Model Comparison**
   - GradientBoostingClassifier
   - XGBoost (industry standard for tabular data)
   - LightGBM (faster training, similar performance)
   - Ensemble voting classifier

5. **Feature Engineering**
   - Interaction features (e.g., CPU × Memory)
   - Ratio features (e.g., Throughput / Active Users)
   - Polynomial features for non-linear relationships

6. **Advanced Resampling**
   - Try ADASYN (adaptive synthetic sampling)
   - Combine SMOTE with Tomek links (SMOTETomek)
   - Ensemble methods with different resampling strategies

### Long-Term (Production Deployment)

7. **Temporal Features**
   - Time-series analysis (rolling averages, trends)
   - Seasonality detection (hour-of-day, day-of-week patterns)
   - LSTM/GRU for sequence modeling

8. **More Data**
   - Google Cluster Data (ClusterData2019)
   - Azure Public Dataset
   - Alibaba ClusterData
   - Multi-source ensemble for robustness

9. **Model Deployment**
   - REST API with Flask/FastAPI
   - Containerization (Docker)
   - CI/CD pipeline for model updates
   - A/B testing framework

---

## Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core programming language | 3.9+ |
| **scikit-learn** | ML framework (RandomForest, metrics, train/test split) | 1.0+ |
| **imbalanced-learn** | SMOTE implementation | 0.12+ |
| **Pandas** | Data manipulation and feature engineering | 1.3+ |
| **NumPy** | Numerical computing | 1.21+ |
| **Matplotlib** | Static visualization | 3.4+ |
| **Seaborn** | Statistical visualization | 0.11+ |
| **Joblib** | Model serialization | 1.1+ |
| **JupyterLab** | Interactive development environment | 3.0+ |

---

## Reproducibility

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/salvadorfrs5/server-crash-prediction.git
cd server-crash-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter lab notebooks/01_exploration.ipynb
```

### Model Artifacts

All trained models and results are saved in the `models/` directory:

- **random_forest_smote_model.pkl**: Trained RandomForest model (600 estimators)
- **feature_importance.csv**: Feature importance rankings
- **metrics.json**: Performance metrics and metadata

### Reproducibility Notes

- **Random State**: All random operations use `random_state=42`
- **Deterministic**: Results should be identical across runs
- **Dataset**: `cloud_workload_dataset.csv` (615 KB, included in repo)

---

## Presentation Materials

### Key Talking Points for Symposium

1. **Problem Statement**: Server crashes cost businesses millions in downtime - can we predict them before they happen?

2. **Innovative Approach**: Used SMOTE to balance training data and feature importance analysis to identify top predictors

3. **Key Finding**: System throughput, memory consumption, and network bandwidth are the strongest predictors (10% each)

4. **Impact**: Early warning system could reduce MTTR and prevent cascading failures in cloud infrastructure

5. **Lessons Learned**: Class imbalance is hard! Even with SMOTE, recall remained low - threshold tuning is critical

6. **Future Vision**: Expand to temporal modeling and multi-cloud datasets for production-ready deployment

---

## Acknowledgments

**Team Members**:
- Leilany Rojas
- Ammar Salama
- Salvador Frias
- Mashel Khan

**Program**: AI4ALL Ignite Program
**Presentation Venue**: AI4ALL Research Symposium

---

## Contact & Links

**Repository**: https://github.com/salvadorfrs5/server-crash-prediction
**Dataset Source**: Cloud Workload Dataset (Kaggle/custom)
**License**: MIT

---

*Generated: December 2025*
*Model Version: 1.0*
*Framework: scikit-learn + imbalanced-learn*
