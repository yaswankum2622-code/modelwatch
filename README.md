---
title: ModelWatch
emoji: "🔭"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
python_version: "3.10"
fullWidth: true
header: mini
short_description: Wide MLOps dashboard for drift and model health.
tags:
  - streamlit
  - mlops
  - drift-detection
  - explainability
  - monitoring
  - lightgbm
  - tensorflow
pinned: false
---

# ModelWatch

ModelWatch is a production-style monitoring surface for a credit-risk model trained on the UCI Credit Card Default dataset.

The dashboard tracks:

- data drift with PSI, KS, JS divergence, chi-square tests, and Evidently reports
- model degradation with AUC, F1, precision, recall, and KS tracking
- deep drift signals with Isolation Forest, autoencoder reconstruction error, and LSTM PSI forecasting
- behaviour drift with SHAP rank-correlation analysis
- alerting, challenger retraining, and Gemini-generated monitoring reports

The Space runs as a Dockerized Streamlit app on Hugging Face Spaces.
