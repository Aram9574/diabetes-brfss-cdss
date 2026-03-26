# Diabetes Risk Stratification — CDC BRFSS 2015

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Data-CDC%20BRFSS%202015-lightgrey)](https://www.cdc.gov/brfss/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Multiclass clinical ML pipeline for diabetes risk stratification using population-level behavioral and socioeconomic data.**  
> 253,680 adults · CDC BRFSS 2015 · No diabetes / Prediabetes / Diabetes

---

## Clinical Context

The CDC Behavioral Risk Factor Surveillance System (BRFSS) is the largest continuously conducted health survey in the world, collecting data on health behaviors, chronic conditions, and preventive health practices from U.S. adults annually.

This project builds a multiclass classification pipeline on the 2015 BRFSS data to stratify adults into three categories: no diabetes, prediabetes, and diabetes — using only variables available in a routine health survey context: BMI, blood pressure, physical activity, smoking status, general health self-assessment, and socioeconomic indicators.

**Clinical framing:** Can population-level behavioral and socioeconomic data, without any laboratory values, meaningfully stratify diabetes risk across three clinical categories? This project answers that question — including its uncomfortable answer for the prediabetes class.

---

## Key Finding: The Prediabetes Detection Problem

A central result of this project is that **prediabetes is nearly undetectable with survey-based variables alone**, regardless of model choice or class balancing strategy.

This is not a modeling failure. It is a data limitation with direct clinical implications:

- Prediabetes is defined biochemically (HbA1c 5.7–6.4% or FPG 100–125 mg/dL), not by behavioral or symptomatic markers
- The BRFSS contains no laboratory values — only self-reported behaviors and conditions
- Individuals with prediabetes are largely asymptomatic and behaviorally indistinguishable from normoglycemic adults in survey data
- This finding reinforces why clinical guidelines recommend biochemical screening, not risk-score triage, for prediabetes detection

This limitation is documented explicitly in the analysis and surfaced in the app UI.

---

## What This Project Demonstrates

| Layer | Content |
|---|---|
| Clinical reasoning | Multiclass framing with explicit justification; honest treatment of model limitations as clinical findings |
| ML pipeline | SMOTE for severe class imbalance (46:1), stratified CV, GridSearchCV, multiclass metrics |
| Explainability | SHAP multiclass — per-class feature contributions, global importance |
| Equity analysis | Stratified performance by Sex, Age group, Income, and Education level |
| Deployment | Streamlit app with survey-style input, three-class risk output, SHAP explanation |

---

## Dataset

**Source:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
**Origin:** CDC BRFSS 2015 — 253,680 survey responses, U.S. general adult population

**Target variable — `Diabetes_012`:**

| Class | Label | N | % |
|---|---|---|---|
| 0 | No diabetes | 213,703 | 84.2% |
| 1 | Prediabetes | 4,631 | 1.8% |
| 2 | Diabetes | 35,346 | 13.9% |

**Features (21 variables):**

| Feature | Type | Clinical meaning |
|---|---|---|
| HighBP | Binary | Self-reported hypertension diagnosis |
| HighChol | Binary | Self-reported high cholesterol diagnosis |
| CholCheck | Binary | Cholesterol check in past 5 years |
| BMI | Continuous | Body Mass Index |
| Smoker | Binary | ≥100 cigarettes lifetime |
| Stroke | Binary | History of stroke |
| HeartDiseaseorAttack | Binary | History of CHD or MI |
| PhysActivity | Binary | Physical activity in past 30 days |
| Fruits | Binary | Fruit consumption ≥1/day |
| Veggies | Binary | Vegetable consumption ≥1/day |
| HvyAlcoholConsump | Binary | Heavy alcohol use |
| AnyHealthcare | Binary | Any health coverage |
| NoDocbcCost | Binary | Couldn't see doctor due to cost |
| GenHlth | Ordinal (1–5) | Self-rated general health |
| MentHlth | 0–30 days | Days of poor mental health |
| PhysHlth | 0–30 days | Days of poor physical health |
| DiffWalk | Binary | Difficulty walking/climbing stairs |
| Sex | Binary | 0=Female, 1=Male |
| Age | Ordinal (1–13) | Age categories (18–24 to 80+) |
| Education | Ordinal (1–6) | Education level |
| Income | Ordinal (1–8) | Household income level |

### Known Limitations

- **No laboratory values.** HbA1c, fasting glucose, and insulin are absent. This fundamentally limits prediabetes detection.
- **Self-reported data.** Diagnosis of diabetes and prediabetes is self-reported, not clinically verified.
- **U.S. population, 2015.** May not generalize to other healthcare systems, populations, or time periods.
- **Cross-sectional design.** No causal inference possible; incident vs. prevalent cases not distinguished.
- **Severe class imbalance.** Prediabetes class (1.8%) requires explicit handling and careful metric interpretation.

---

## Model Performance

| Model | Balanced Accuracy | Macro F1 | Diabetes Recall | Prediabetes Recall |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

> Full results in `notebooks/03_evaluation.ipynb`

**Primary metric: Balanced Accuracy** (macro-averaged recall across classes). Accuracy is misleading with 84/2/14% class distribution.

---

## Project Structure

```
diabetes-brfss-cdss/
├── data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
├── notebooks/
│   ├── 01_eda.ipynb              # EDA, class imbalance, feature analysis
│   ├── 02_preprocessing.ipynb   # SMOTE, scaling, train/test split
│   └── 03_evaluation.ipynb      # Model comparison, SHAP, equity analysis
├── src/
│   ├── model_xgb.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/Aram9574/diabetes-brfss-cdss.git
cd diabetes-brfss-cdss
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Live Demo

[Launch App](https://your-streamlit-url.streamlit.app) ← *updated after deployment*

---

## Clinical Disclaimer

This tool is for educational and research purposes only. It does not constitute medical advice and is not validated for clinical use. Prediabetes and diabetes diagnosis requires biochemical confirmation per clinical guidelines (ADA 2024, WHO).

---

## Author

**Alejandro Zakzuk** — Physician · AI Applied to Health (CEMP) · Digital Health (Universidad Europea de Madrid)

[LinkedIn](https://linkedin.com/in/alejandrozakzuk-ia-salud-digital) · [Website](https://alejandrozakzuk.com) · [GitHub](https://github.com/Aram9574)
