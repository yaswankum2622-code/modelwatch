# Future Work

ModelWatch v1.0 proves the end-to-end monitoring loop on batch windows. The next iteration should move closer to production reality:

- **Real-time streaming ingestion** with Kafka or Redpanda so drift signals update continuously instead of per batch window
- **Scheduled retraining orchestration** so alert thresholds can automatically open a retraining job instead of only recommending one
- **Prediction logging and delayed-label reconciliation** for live AUC and calibration tracking
- **Alert routing** to Slack, email, or incident tooling with alert suppression and deduplication
- **Metric registry expansion** for fairness, calibration, and business-cost metrics
- **Model registry integration** so champion and challenger promotion is versioned formally
- **Human review workflow** for high-risk alert windows before model promotion

The core architecture is already in place. Future work is mainly about turning the current batch observability demo into a continuously running platform.
