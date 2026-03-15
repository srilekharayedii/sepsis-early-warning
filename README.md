# Sepsis Early Warning System
> Predicting sepsis onset 6 hours before clinical diagnosis.

## Problem
Sepsis kills 270,000 Americans annually. 80% preventable with early detection.
Most hospitals use 1990s rule-based alerts with 30% false positive rate.

## Solution
ML system on 40,000 real ICU records detecting sepsis risk 6 hours early
by analyzing trends in vital signs and lab values.

## Tech Stack
Python · pandas · XGBoost · scikit-learn · MLflow · SHAP · FastAPI · Docker · AWS


# Results

| Model | AUC-ROC | Recall | Precision | Threshold |
|-------|---------|--------|-----------|-----------|
| Logistic Regression (baseline) | 0.697 | 0.581 | 0.037 | 0.5 |
| XGBoost (5K patients) | 0.733 | 0.677 | 0.038 | 0.1 |
| XGBoost (40K patients) | 0.758 | 0.769 | 0.032 | 0.1 |

### In hospital terms (per 100 sepsis patients)
- Baseline SIRS criteria: ~60% recall, ~30% false positive rate
- Our model: catches 77 patients, misses 23
