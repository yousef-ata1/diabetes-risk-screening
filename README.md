# 🩺 Diabetes Risk Screening Tool

A short health questionnaire that tells people whether they should get a diabetes blood test, built on ~250K CDC survey responses. **A screening aid, not a medical diagnosis.**

**🔗 Live app:** `[paste your Streamlit / Hugging Face link here]`

![App screenshot](screenshot.png)  <!-- add a screenshot or GIF of your app -->

## Problem
Many people have diabetes or prediabetes without knowing it. The goal is a cheap, quick questionnaire that flags who should see a doctor, with **recall** as the priority (missing a case is worse than an extra check-up).

## Results (test set, never used for tuning)
| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Baseline (majority class) | [X] | [X] | [X] | [X] |
| Best model, all 21 questions | [X] | [X] | [X] | [X] |
| **Final model, 7 questions** | [X] | [X] | [X] | [X] |

> Replace every `[X]` with the numbers printed by the last cells of the notebook.

## Approach
- Dropped duplicate rows, split train/test **before** fitting anything, everything inside a scikit-learn `Pipeline`.
- Compared logistic regression, random forest and gradient boosting with stratified CV; tuned the best one with randomized search (training data only).
- Chose the decision threshold from out-of-fold training predictions to reach ~80% recall.
- Ranked questions with permutation importance (validation split from train) and cut the form to 7 questions.

## Run it
```bash
pip install -r requirements.txt            # pin scikit-learn to the version used in the notebook
# 1) download the data (not included): https://archive.ics.uci.edu/dataset/891
#    file: diabetes_binary_health_indicators_BRFSS2015.csv  -> put it next to the notebook
# 2) run diabetes_risk_screening.ipynb (Restart & Run All) -> creates model.joblib
streamlit run app.py
```

## Deploy
Push `app.py`, `model.joblib`, `requirements.txt` to GitHub, then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud) (main file: `app.py`).

## Limitations
Self-reported US data from 2015; not validated for other populations; many false alarms at high recall; shows association, not causation.

## Data
CDC Diabetes Health Indicators (BRFSS 2015), UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/891
