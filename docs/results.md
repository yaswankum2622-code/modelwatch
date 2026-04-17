# ModelWatch Results

## Window-by-window monitoring results

The current run produces the following operating picture:

| Window | AUC | F1 | Max PSI | Anomaly rate | AE drift ratio | SHAP rank corr |
|---|---:|---:|---:|---:|---:|---:|
| W1 baseline | 0.9258 | 0.7382 | 0.0000 | 5.00% | 1.000x | 1.0000 |
| W2 mild drift | 0.7521 | 0.5025 | 0.0492 | 5.99% | 1.034x | 0.9417 |
| W3 moderate drift | 0.7334 | 0.4728 | 0.2834 | 9.25% | 1.204x | 0.7984 |
| W4 severe drift | 0.6553 | 0.4104 | 0.9189 | 16.53% | 1.291x | 0.6166 |

The key result is the severe production window. **Window 4 AUC falls to 0.6553** from a baseline of 0.9258, a degradation of roughly **29.2%**. At the same time PSI becomes extreme on repayment-status variables, anomaly density rises, and SHAP correlation drops, so every detector points in the same direction.

## Most drifted Window 4 features

| Feature | PSI | Status |
|---|---:|---|
| PAY_0 | 0.9189 | RED |
| PAY_6 | 0.4693 | RED |
| PAY_4 | 0.4265 | RED |
| PAY_3 | 0.4026 | RED |
| PAY_2 | 0.3962 | RED |

## Champion-challenger comparison

| Model | Evaluation window | AUC | F1 | Decision impact |
|---|---:|---:|---:|---|
| Champion | W4 holdout | 0.6317 | 0.3998 | Original production model |
| Challenger | W4 holdout | 0.7459 | 0.4908 | Retrained on recent windows |

| Summary metric | Value |
|---|---:|
| AUC improvement | +0.1143 |
| Final decision | PROMOTE CHALLENGER |

The retrained challenger materially outperforms the champion, which means ModelWatch does not stop at detection. It shows that retraining is operationally worth doing.

## Forecast result

- Predicted Window 5 mean PSI: `0.5294`
- Predicted Window 5 max PSI: `1.5242`
- Recommendation: `RETRAIN URGENTLY`
