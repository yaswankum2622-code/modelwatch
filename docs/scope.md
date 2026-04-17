# ModelWatch Scope

ModelWatch v1.0 is scoped as a focused production-monitoring project with **14 MVP features**:

1. Data pipeline: UCI Excel load plus four monitoring windows with injected drift
2. LightGBM production baseline model
3. PSI, KS, and Jensen-Shannon drift detection
4. Autoencoder unsupervised drift detection
5. Isolation Forest anomaly detection
6. Evidently AI drift reports
7. Performance tracking across windows
8. SHAP importance drift analysis
9. Rules-based alerting engine
10. Gemini 2.5 Flash executive drift report
11. LSTM drift forecasting
12. Champion-challenger retraining loop
13. Streamlit dashboard for operational monitoring
14. dbt metric registry for monitoring KPIs

Included in scope:

- One real dataset: UCI Default of Credit Card Clients
- One baseline training window and three progressively drifted monitoring windows
- Statistical, unsupervised, explainability, and reporting layers
- Reproducible model artifacts and pytest coverage
- GitHub Actions CI and Hugging Face deployment support

Explicitly out of scope for v1.0:

- Real-time Kafka or streaming ingestion
- Multi-model registry and multi-tenant serving
- Human-in-the-loop alert workflow tools
- Online feature store integration
- Automated scheduled retraining in production

The goal is not to build a full commercial observability platform. The goal is to demonstrate the complete monitoring loop clearly: detect shift, quantify impact, explain why it happened, forecast whether it will worsen, and compare retrained alternatives before promotion.
