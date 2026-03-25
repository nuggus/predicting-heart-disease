# Predicting Heart Disease Using Clinical and Diagnostic Data

**Author:** Viswanath Nuggu
**Date:** March 2026

---

#### Executive Summary

This project builds a machine learning model to predict whether a patient has heart disease based on 13 clinical features including age, blood pressure, cholesterol, EKG results, and diagnostic test outcomes. A Logistic Regression baseline model trained on 630,000 patient records achieved a **ROC-AUC of 0.9516** and **88.5% accuracy**, exceeding the initial target. The top predictors identified were Thallium stress test results, chest pain type, and maximum heart rate — all of which are routinely collected during standard clinical check-ups.

---

#### Rationale

Heart disease is the leading cause of death globally, responsible for approximately 17.9 million deaths per year (WHO). Most patients are only diagnosed after a serious cardiac event such as a heart attack — by which point the opportunity for low-cost preventive intervention has often passed. A reliable predictive model that uses data already collected during routine check-ups can shift healthcare from reactive to proactive, enabling clinicians to flag high-risk patients early and intervene before the situation becomes critical. This directly translates to saved lives, reduced hospitalizations, and lower healthcare costs.

---

#### Research Question

Can a machine learning model accurately predict the presence or absence of heart disease in a patient based on their clinical measurements and diagnostic test results, and which factors are most strongly associated with that risk?

---

#### Data Sources

- **Dataset:** [Kaggle Playground Series S6E2 — Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2/overview)
- **Data files:** [https://www.kaggle.com/competitions/playground-series-s6e2/data](https://www.kaggle.com/competitions/playground-series-s6e2/data)
- **Training records:** 630,000 rows × 14 features + 1 target
- **Test records:** 270,000 rows
- **Target variable:** Heart Disease (Presence / Absence)
- **Features:** Age, Sex, Chest Pain Type, BP, Cholesterol, FBS over 120, EKG Results, Max HR, Exercise Angina, ST Depression, Slope of ST, Number of Vessels Fluro, Thallium

The dataset is synthetically generated from real patient records as part of the Kaggle Playground Series, designed for ML practice and experimentation.

---

#### Methodology

1. **Exploratory Data Analysis (EDA)**
   - Assessed data quality: no missing values found across all 630,000 records
   - Analyzed class balance: 55.2% Absence / 44.8% Presence — nearly balanced, no resampling needed
   - Examined distributions of numeric and categorical features split by target class
   - Generated correlation heatmap to identify multicollinearity and feature-target relationships

2. **Feature Engineering**
   - `High_Cholesterol`: Binary flag for cholesterol > 240 (borderline high threshold)
   - `High_BP`: Binary flag for blood pressure ≥ 140 (stage 2 hypertension threshold)
   - `Low_MaxHR`: Binary flag for max heart rate < 120 (poor cardiac response indicator)
   - `Risk_score`: Composite score summing 6 binary risk flags — showed near-monotonic relationship with disease rate (0 flags → 9.5% disease rate; 6 flags → 97.4%)
   - `Age_group`: Age binned into 5 groups for categorical analysis

3. **Baseline Model: Logistic Regression**
   - 80/20 stratified train/validation split
   - StandardScaler normalization applied
   - Evaluated on accuracy, precision, recall, F1-score, and ROC-AUC

---

#### Results

**Key EDA Findings:**
- The dataset is complete with zero missing values
- `Max HR`, `Thallium`, `Chest Pain Type`, `ST Depression`, and `Number of Vessels Fluro` are the strongest predictors of heart disease
- Patients with lower max heart rate and asymptomatic chest pain (type 4) show the highest disease rates
- The engineered Risk Score validates the additive nature of clinical risk factors

**Baseline Model Performance (Logistic Regression):**

| Metric | Score |
|--------|-------|
| Accuracy | 88.5% |
| Precision | 88.2% |
| Recall | 85.9% |
| F1 Score | 87.0% |
| **ROC-AUC** | **0.9516** |

The baseline Logistic Regression model significantly exceeded the target ROC-AUC of 0.90, demonstrating that the features are highly informative even for a linear classifier.

**Top 3 Most Predictive Features:**
1. Thallium stress test result (reversible defect = high risk)
2. Chest Pain Type (asymptomatic type 4 = high risk)
3. Max Heart Rate (lower HR = higher risk)

---

#### Next Steps

- Train ensemble models (XGBoost, LightGBM) and compare against Logistic Regression baseline
- Perform hyperparameter tuning via cross-validation to optimize gradient boosting models
- Generate SHAP (SHapley Additive exPlanations) values for deeper feature interpretability
- Prepare final results in a format accessible to non-technical audiences (Module 24)
- Investigate whether additional feature interactions further improve model performance

---

#### Outline of Project

- [heart_disease_eda_baseline.ipynb](./heart_disease_eda_baseline.ipynb) — Full EDA, feature engineering, visualizations, and Logistic Regression baseline model

---

##### Contact and Further Information

**Viswanath Nuggu**
Email: viswanath_nuggu@intuit.com
Dataset: [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2/overview)
