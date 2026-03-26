# Predicting Heart Disease Using Clinical Data

**Author:** Viswanath Nuggu  
**Date:** March 2026
**Dataset:** [Kaggle Playground Series S6E2 — Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2/overview)

---

## Executive Summary

Heart disease is the leading cause of death worldwide. This project builds a machine learning model that can predict whether a patient has heart disease — using only information already collected during a routine medical check-up, such as blood pressure, cholesterol, and EKG results.

Our best model (XGBoost) correctly identifies heart disease patients with a **95.6% discriminating ability** and **88.9% overall accuracy** across 630,000 patient records. The five most important warning signs identified were: an abnormal Thallium stress test, multiple blocked arteries, asymptomatic chest pain, a low maximum heart rate, and elevated ST depression on an EKG.

This analysis shows that early, automated screening is not just possible — it is highly accurate using data clinicians already have.

---

## Why Does This Matter?

Every year, approximately **17.9 million people die from heart disease** globally (WHO). Many of these deaths are preventable — but only if the disease is caught early enough.

The problem today is that most patients are only diagnosed *after* something goes wrong: a heart attack, a hospital emergency, or a serious symptom. By that point, the damage is done and treatment is expensive.

This project addresses a simple but powerful question:

> *"Given what we already know about a patient from their routine check-up, should we be concerned about their heart?"*

A reliable answer to this question allows doctors to **intervene early**, before a crisis occurs — saving lives and reducing healthcare costs.

---

## Research Question

Can a machine learning model accurately predict the presence or absence of heart disease in a patient based on their clinical measurements and diagnostic test results, and which factors are most strongly associated with that risk?

---

## Data Source

- **Source:** [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2/data)
- **Size:** 630,000 training records | 270,000 test records
- **Target:** Heart Disease — Presence or Absence
- **Features (13):** Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, Fasting Blood Sugar, EKG Results, Max Heart Rate, Exercise Angina, ST Depression, ST Slope, Number of Vessels (Fluoroscopy), Thallium Stress Test

The dataset was synthetically generated from real patient records as part of a Kaggle machine learning competition.

---

## Methodology

### Step 1 — Data Cleaning & Exploration *([heart_disease_eda_baseline.ipynb](./heart_disease_eda_baseline.ipynb))*

- Confirmed zero missing values and zero duplicate records across all 630,000 rows
- Verified the dataset is nearly balanced (55% No Disease / 45% Disease) — no special handling required
- Performed outlier analysis using the IQR method — extreme values were retained as clinically valid

### Step 2 — Feature Engineering

Created additional variables to capture clinical knowledge:

- **High Cholesterol flag** — above 240 mg/dL (borderline-high threshold per AHA guidelines)
- **High Blood Pressure flag** — at or above 140 mmHg (Stage 2 hypertension)
- **Low Max Heart Rate flag** — below 120 bpm (reduced cardiac response)
- **Composite Risk Score** — counts how many of 6 risk factors a patient has (0 = lowest risk → 6 = highest risk)

### Step 3 — Baseline Model *([heart_disease_eda_baseline.ipynb](./heart_disease_eda_baseline.ipynb))*

Trained a Logistic Regression model as the interpretable starting point.

### Step 4 — Advanced Models + Tuning *([heart_disease_final_report.ipynb](./heart_disease_final_report.ipynb))*

Trained three additional models and tuned the best one:

- **Decision Tree** — simple, rule-based interpretation
- **XGBoost** — powerful gradient boosting, tuned via RandomizedSearchCV (20 combinations, 3-fold cross-validation)
- **LightGBM** — optimized for large datasets

### Step 5 — Cross-Validation

All models were evaluated using **5-fold stratified cross-validation** to confirm results are stable and not due to a lucky data split.

---

## Results

### Key EDA Findings

- The dataset is complete with **zero missing values** and **zero duplicate records** across all 630,000 rows
- Classes are nearly balanced — 55.2% Absence / 44.8% Presence — no resampling required
- `Max HR`, `Thallium`, `Chest Pain Type`, `ST Depression`, and `Number of Vessels Fluro` are the strongest predictors of heart disease
- Patients with lower max heart rate and asymptomatic chest pain (Type 4) show the highest disease rates
- The engineered **Risk Score** validates the additive nature of clinical risk factors — ranging from 9.5% disease rate (0 flags) to 97.4% (6 flags)
- Outliers were detected in BP, ST Depression, and Max HR but retained as clinically valid extreme values

### Baseline Model Performance — Logistic Regression


| Metric      | Score      |
| ----------- | ---------- |
| Accuracy    | 88.5%      |
| Precision   | 88.2%      |
| Recall      | 85.9%      |
| F1 Score    | 87.0%      |
| **ROC-AUC** | **0.9516** |


The baseline exceeded the target ROC-AUC of 0.90, confirming the features are highly informative even for a linear classifier.

---

### Final Model Performance Summary


| Model               | Accuracy  | ROC-AUC    | Cross-Val AUC       |
| ------------------- | --------- | ---------- | ------------------- |
| Logistic Regression | 88.5%     | 0.9516     | 0.9504 ± 0.0008     |
| Decision Tree       | 87.3%     | 0.9399     | 0.9389 ± 0.0013     |
| **XGBoost (Tuned)** | **88.9%** | **0.9561** | **0.9547 ± 0.0007** |
| LightGBM            | 88.9%     | 0.9560     | 0.9546 ± 0.0008     |


**Best model: XGBoost (Tuned)** with ROC-AUC = 0.9561.

> In plain terms: if you randomly selected one patient with heart disease and one without, our model would correctly identify the sick patient as higher risk **95.6% of the time**. A coin flip would only get it right 50% of the time.

### The 5 Biggest Warning Signs

Based on SHAP analysis (a technique that explains *why* the model makes each prediction):


| Rank | Warning Sign                         | Plain English Meaning                                                              |
| ---- | ------------------------------------ | ---------------------------------------------------------------------------------- |
| 1    | **Thallium Reversible Defect**       | Part of the heart is not getting enough blood during exercise                      |
| 2    | **Multiple Blocked Vessels**         | More clogged arteries = dramatically higher risk                                   |
| 3    | **Asymptomatic Chest Pain (Type 4)** | No obvious pain does not mean no problem — silent disease is the highest-risk type |
| 4    | **Low Maximum Heart Rate**           | A struggling heart cannot speed up the way a healthy one does during exercise      |
| 5    | **High ST Depression**               | An EKG sign that the heart muscle is not receiving enough oxygen                   |


---

## Key Findings

1. **Early detection is possible with existing data.** No new tests are needed — every feature used is routinely collected during standard check-ups.
2. **Asymptomatic patients are the highest-risk group.** Patients reporting no chest pain (Type 4) had the highest heart disease rates. This challenges the assumption that "no symptoms = low risk."
3. **The Risk Score is a powerful quick-screen tool.** Patients with 0 risk factors had only a 9.5% disease rate. Patients with all 6 risk factors had a 97.4% rate. This score can be calculated in seconds from existing data.
4. **All models are stable and reliable.** Cross-validation standard deviations of ≤ 0.0013 confirm results hold consistently across different patient subsets.
5. **Even a simple model performs very well.** Logistic Regression (AUC 0.9516) is close to the best model (XGBoost 0.9561), suggesting the risk patterns in this data are strong and consistent.

---

## Actionable Recommendations

**For Clinicians:**

- Patients with a Thallium reversible defect combined with multiple vessel blockages should be flagged for immediate cardiology referral
- Treat asymptomatic chest pain as a serious warning sign, not reassurance
- Use the composite Risk Score as a quick triage filter during routine appointments — patients scoring 4 or above have greater than 85% chance of heart disease

**For Healthcare Systems:**

- This model can be integrated into existing Electronic Health Record (EHR) systems with minimal engineering effort
- Prioritizing high-risk patients for follow-up can significantly reduce emergency hospitalizations
- The model is fully explainable — physicians can see *why* a patient is flagged, not just *that* they are flagged

---

## Next Steps

- **Threshold tuning** — Adjust the model's decision point to minimize missed heart disease cases, since missing a sick patient is more costly than a false alarm
- **Ensemble modeling** — Combine XGBoost and LightGBM predictions for further gains
- **Fairness audit** — Ensure the model performs equally well across gender and age subgroups
- **Clinical validation** — Test predictions against real patient outcomes in a hospital setting
- **Deployment** — Build a simple web tool where a clinician inputs patient values and receives an instant risk score

---

## Project Notebooks


| Notebook                                                               | Description                                                                                                                                    |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [heart_disease_eda_baseline.ipynb](./heart_disease_eda_baseline.ipynb) | Data loading, cleaning, EDA, visualizations, feature engineering, and Logistic Regression baseline model                                       |
| [heart_disease_final_report.ipynb](./heart_disease_final_report.ipynb) | Advanced models (Decision Tree, XGBoost, LightGBM), hyperparameter tuning, cross-validation, SHAP explainability, and business recommendations |


---

## Contact

**Viswanath Nuggu**  
Dataset: [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2/overview)
