# ModelWatch Algorithms

## Statistical drift layer

ModelWatch uses three classical drift detectors on top of the same baseline window:

- **PSI (Population Stability Index)** for feature-level distribution shift
- **KS test** for continuous distribution change
- **Jensen-Shannon divergence** for probability-distance comparison
- **Chi-square** for categorical feature drift

The PSI thresholds match standard credit-risk monitoring practice:

- `PSI < 0.10`: healthy
- `0.10 <= PSI < 0.25`: monitor closely
- `PSI >= 0.25`: alert / retraining candidate

## Production model

The production scoring model is a **LightGBM classifier** trained on Window 1. It is used as the champion baseline because gradient-boosted trees are common in tabular credit decisioning and work well with mixed numeric and categorical-style encoded variables.

## Autoencoder architecture

The unsupervised neural detector is a fully connected **autoencoder** built in TensorFlow / Keras. It scales the baseline features, compresses them through a bottleneck representation, and reconstructs the original input:

`23 features -> 11 hidden units -> bottleneck -> 11 hidden units -> 23 outputs`

The dashboard monitors mean reconstruction error and the ratio versus the baseline mean. As drift intensifies, reconstruction error rises because the current data no longer resembles the baseline manifold the autoencoder learned.

## Isolation Forest

Isolation Forest is used for per-record anomaly scoring. It does not need labels, which makes it useful when recent ground truth is delayed. In ModelWatch it acts as a second unsupervised signal alongside the autoencoder.

## SHAP behaviour drift

Data drift is not enough on its own. ModelWatch also computes mean absolute SHAP importance by feature and compares rankings across windows using **Spearman rank correlation**. Lower correlation means the model is making decisions for different reasons than it did at baseline.

## LSTM forecast

The forecasting layer is a compact **LSTM** sequence model trained on the PSI trajectory of the most drifted features. Its job is not long-horizon forecasting accuracy. Its job is to answer the operational question: if current drift momentum continues, how bad is the next monitoring window likely to be?

## Champion-challenger loop

When performance degradation becomes material, ModelWatch retrains a challenger on recent windows and evaluates it against the original champion on a holdout slice. This closes the MLOps loop by turning monitoring into an evidence-based retraining decision rather than a manual guess.
