# ModelWatch — Project 03 Scope Freeze

---

## Project Name

**ModelWatch**
*Production ML Model Monitoring and Drift Detection*

Tagline: *The system that catches your model dying before your users do.*

---

## Problem Statement

A model gets trained. It gets deployed. Everyone moves on.

Six months later the model is still running but the world has
changed. Interest rates shifted. Consumer behaviour changed.
A new competitor entered the market. The feature distributions
that the model was trained on look nothing like what it is seeing
in production. Nobody notices because the model is still returning
predictions — just wrong ones.

This is model drift. It is one of the most expensive silent
failures in production ML.

**Real companies that got this wrong:**

Zillow lost $569 million and shut down its iBuying division in 2021.
Their home price prediction model drifted as COVID changed housing
markets. The model kept predicting with high confidence. The
predictions were catastrophically wrong. No monitoring caught it.

Uber has published that their fraud models degrade within weeks of
deployment without active monitoring. They now run continuous
drift checks on every model in production.

Amazon pulled their ML hiring algorithm after it drifted to
discriminate against women — undetected for years. The model
was performing fine on its original metric. It had silently
learned a different, discriminatory decision boundary.

**The scale of the problem:**

The MLOps monitoring market (Evidently AI, Arize, Fiddler,
WhyLabs) is valued at $4.5 billion and growing 35% annually.
Every company with a model in production needs this.
Most are solving it with cron jobs and spreadsheets.

**Three specific failures ModelWatch prevents:**

Problem 1 — Distribution drift goes undetected
The incoming data starts looking different from training data.
PSI and KS tests catch this. Most teams have no automated check.

Problem 2 — Performance degradation is invisible
AUC drops from 0.82 to 0.71 over four months. Nobody notices
because the model is still returning predictions confidently.
ModelWatch tracks performance across time windows and alerts.

Problem 3 — Nobody knows which feature drifted or why
When drift is detected, finding the root cause requires manual
analysis. SHAP drift comparison and Isolation Forest anomaly
detection pinpoint exactly which features changed and which
records look anomalous.

---

## Dataset

**UCI Default of Credit Card Clients**

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository |
| Name | Default of Credit Card Clients |
| Records | 30,000 real credit card clients |
| Features | Payment history, bill amounts, demographics |
| Target | Default next month (binary) |
| URL | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| Format | Excel (.xls) |
| Size | 2.7 MB |
| Cost | Free — no account needed |

**Why this dataset:**
Payment behaviour and bill amounts are exactly the kind of
features that drift when economic conditions change — making
drift simulation realistic and defensible in interviews.
Small enough to train fast. Rich enough to show all drift types.

**How drift is injected across 4 windows:**
```
Window 1 (baseline):   Original distribution — model trained here
Window 2 (mild):       5-10% shift in payment features
Window 3 (moderate):   15-25% shift — alerts trigger
Window 4 (severe):     40%+ shift — performance degrades visibly
```

This is exactly how Evidently AI and WhyLabs demo their products
to enterprise customers.

---

## MVP — 14 Features

| # | Feature | Category |
|---|---|---|
| 1 | Data pipeline — UCI load + 4-window drift injection | Data |
| 2 | LightGBM production model (baseline) | Classical ML |
| 3 | PSI + KS + Jensen-Shannon drift detection | Statistics |
| 4 | Autoencoder unsupervised drift detection | Neural Network |
| 5 | Isolation Forest per-record anomaly detection | Unsupervised ML |
| 6 | Evidently AI drift reports | Industry Tool |
| 7 | Model performance tracking across windows | ML Monitoring |
| 8 | SHAP drift analysis — feature importance shift | Explainability |
| 9 | Alerting system — PSI threshold rules engine | Monitoring |
| 10 | Gemini 2.5 Flash plain English drift report | GenAI |
| 11 | LSTM drift forecasting — predict when retrain needed | Deep Learning |
| 12 | Champion-Challenger retraining loop | Full MLOps Loop |
| 13 | Streamlit dashboard — 6 pages | Deployment |
| 14 | dbt metric registry for monitoring KPIs | Data Governance |

---

## Drift Detection Methods — Complete

| Method | Detects | Threshold |
|---|---|---|
| PSI (Population Stability Index) | Distribution shift in inputs | < 0.10 green, 0.10-0.25 amber, > 0.25 red |
| KS Test (Kolmogorov-Smirnov) | Statistical distribution change | p-value < 0.05 = drift |
| Jensen-Shannon Divergence | Probability distance | > 0.1 = moderate drift |
| Chi-Square Test | Categorical feature drift | p-value < 0.05 = drift |
| Autoencoder Reconstruction Error | Non-linear unsupervised drift | > 2 std from baseline mean |
| Isolation Forest Anomaly Score | Per-record anomalies | score < -0.1 = anomaly |
| SHAP Importance Drift | Model behaviour change | rank correlation < 0.8 |
| AUC / F1 Degradation | Outcome performance drift | > 5% drop = alert |
| Evidently AI DataDriftReport | Comprehensive drift suite | Built-in thresholds |

---

## Models Used

| Model | Type | Purpose |
|---|---|---|
| LightGBM | Gradient Boosting | Production credit default model |
| Autoencoder | Neural Network (TF/Keras) | Unsupervised drift detection via reconstruction error |
| LSTM | Recurrent Neural Network | Time-series forecasting of PSI scores |
| Isolation Forest | Unsupervised ML | Per-record anomaly scoring |
| Logistic Regression | Classical ML | Champion-Challenger comparison baseline |

**No fine-tuning needed. No pretrained models.**
All models trained from scratch on the UCI dataset and drift windows.

---

## Stack

```
Python 3.11
lightgbm==4.3.0           production model
scikit-learn==1.4.0       isolation forest + preprocessing
tensorflow==2.15.0        autoencoder + LSTM
shap==0.44.0              feature importance drift
evidently==0.4.16         industry drift reports
scipy==1.12.0             KS test, chi-square, JS divergence
pandas==2.2.0             data processing
numpy==1.26.0             numerical computing
streamlit==1.32.0         dashboard
plotly==5.19.0            charts
google-generativeai==0.5.0  Gemini 2.5 Flash
python-dotenv==1.0.0      environment variables
dbt-core==1.7.0           metric layer
dbt-sqlite==1.7.0         SQLite backend for dbt
pytest==8.0.0             test suite
joblib==1.3.2             model serialisation
GitHub Actions            CI/CD
Hugging Face Spaces       free deployment
```

---

## Dashboard — 6 Pages

| Page | What it shows |
|---|---|
| 1 — Overview | Model health score, drift status across all windows, alert summary, PSI heatmap |
| 2 — Data Drift | PSI + KS per feature, distribution overlays A vs B, Evidently HTML report |
| 3 — Model Performance | AUC/F1/calibration across windows, degradation curves, performance vs drift scatter |
| 4 — Deep Drift | Autoencoder reconstruction error, Isolation Forest anomaly scores, LSTM forecast |
| 5 — SHAP Drift | Feature importance ranking shift, top movers, importance correlation across windows |
| 6 — Drift Report | Gemini 2.5 Flash plain English report, Champion-Challenger comparison, retraining recommendation |

---

## The Key Output

```
ModelWatch Health Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Window 1 (baseline)   AUC: 0.82   PSI: 0.00   Status: HEALTHY ✓
Window 2 (mild)       AUC: 0.81   PSI: 0.08   Status: HEALTHY ✓
Window 3 (moderate)   AUC: 0.77   PSI: 0.18   Status: MONITOR ⚠
Window 4 (severe)     AUC: 0.71   PSI: 0.31   Status: RETRAIN ✗

Most drifted features:
  PAY_0        PSI=0.41  ← payment status — highest predictor
  BILL_AMT1    PSI=0.28
  LIMIT_BAL    PSI=0.19

Autoencoder reconstruction error:
  Window 1: 0.012  (baseline)
  Window 4: 0.089  (+641% — severe anomaly)

Isolation Forest anomaly rate:
  Window 1: 2.1%
  Window 4: 18.3%  ← 8.7x increase

LSTM forecast:
  Window 5 predicted PSI: 0.38 — RETRAIN RECOMMENDED

Champion vs Challenger:
  Champion (original):   AUC 0.71 on Window 4
  Challenger (retrained): AUC 0.80 on Window 4
  Decision: PROMOTE CHALLENGER

Gemini recommendation:
  "The model has degraded 13.4% from baseline AUC.
   PAY_0 distribution has shifted significantly,
   suggesting a change in customer payment behaviour.
   Retrain immediately using the last 60 days of data."
```

---

## Folder Structure

```
ModelWatch/
│
├── data/
│   ├── loader.py            UCI Excel → SQLite + 4 window split
│   └── drift_injector.py    Synthetic drift injection per window
│
├── database/
│   ├── schema.sql
│   └── db.py
│
├── models/
│   ├── __init__.py
│   ├── lgbm_model.py        LightGBM production model
│   ├── autoencoder.py       TF/Keras autoencoder drift detector
│   ├── lstm_forecast.py     LSTM PSI time series forecasting
│   ├── isolation_forest.py  Per-record anomaly detection
│   └── champion_challenger.py  Retraining + comparison
│
├── monitoring/
│   ├── __init__.py
│   ├── psi.py               Population Stability Index
│   ├── statistical_tests.py KS test, chi-square, JS divergence
│   ├── performance_tracker.py AUC/F1 across windows
│   ├── shap_drift.py        Feature importance shift
│   ├── evidently_reports.py Evidently AI integration
│   └── alerting.py          Rules engine + alert generation
│
├── reports/
│   ├── __init__.py
│   └── drift_report.py      Gemini 2.5 Flash report generator
│
├── dbt_project/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── models/
│       ├── staging/
│       │   └── stg_predictions.sql
│       ├── marts/
│       │   └── drift_metrics.sql
│       └── metrics/
│           └── metric_definitions.yml
│
├── dashboard/
│   └── app.py               Streamlit 6-page dashboard
│
├── tests/
│   ├── conftest.py
│   ├── test_drift_detection.py
│   ├── test_models.py
│   └── test_monitoring.py
│
├── docs/
│   ├── problem_statement.md
│   ├── scope.md
│   ├── algorithms.md
│   ├── results.md
│   └── future_work.md
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── README.md
├── requirements.txt
├── app.py
├── verify.py
├── ci_setup.py
├── .env.example
└── .gitignore
```

---

## Build Order — 11 Prompts

| Prompt | Builds |
|---|---|
| P1 | Scaffold + .gitignore |
| P2 | Data pipeline — UCI load + drift injection |
| P3 | Database schema + helpers |
| P4 | LightGBM + Isolation Forest + Autoencoder |
| P5 | PSI + KS + JS + Evidently drift detection |
| P6 | Performance tracking + SHAP drift |
| P7 | LSTM forecast + Champion-Challenger |
| P8 | Alerting + Gemini drift report |
| P9 | Streamlit dashboard — 6 pages |
| P10 | pytest test suite |
| P11 | CI + README + docs + final push |

---

## Done Means

- UCI data loaded and split into 4 realistic drift windows
- LightGBM trained on baseline window
- PSI + KS + JS drift scores for all features across all windows
- Autoencoder reconstruction error spiking on drifted windows
- Isolation Forest flagging anomalous records in Window 4
- Evidently AI HTML report generated
- LSTM forecasting when next retrain will be needed
- Champion-Challenger comparison showing retrained model wins
- Gemini drift report with plain English recommendation
- 6-page Streamlit dashboard live on Hugging Face
- Tests passing
- CI green
- GitHub clean

---

## Interview Talking Point

> "Zillow lost $569 million because their model drifted and
> nobody caught it in time. ModelWatch is a three-layer detection
> system. The statistical layer uses PSI and KS tests — the
> industry standard for credit and finance models. The ML layer
> uses Isolation Forest to flag which specific records look
> anomalous. The neural network layer uses a TensorFlow autoencoder
> — trained on baseline data, reconstruction error spikes when
> production data looks nothing like training. I added an LSTM
> that forecasts PSI as a time series so you get a warning before
> drift crosses the threshold, not after. Evidently AI generates
> professional HTML reports. Gemini 2.5 Flash writes the plain
> English recommendation. The Champion-Challenger loop closes the
> full MLOps cycle — detect drift, retrain, compare, promote.
> The MLOps monitoring market is worth $4.5 billion. Most companies
> are still solving this with cron jobs and spreadsheets."
