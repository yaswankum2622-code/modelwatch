"""
ModelWatch | models/lstm_forecast.py | LSTM forecast of when retraining needed
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent.parent))

SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)


def build_psi_series() -> np.ndarray:
    """
    Build PSI time series from all 4 windows.
    Returns array of shape (4, n_features).
    """
    from monitoring.psi import compute_psi_all_features

    psi_frames = {w: compute_psi_all_features(w).set_index("feature") for w in [1, 2, 3, 4]}
    feature_order = (
        pd.concat(
            [psi_frames[w]["psi"].rename(f"window_{w}") for w in [1, 2, 3, 4]],
            axis=1,
        )
        .fillna(0.0)
        .max(axis=1)
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    series = []
    for w in [1, 2, 3, 4]:
        values = psi_frames[w].reindex(feature_order)["psi"].fillna(0.0).values
        series.append(values)

    return np.array(series)


def build_lstm_model(input_shape: tuple) -> keras.Model:
    """
    Lightweight LSTM for PSI forecasting.
    Input: (timesteps, features)
    Output: PSI values for next timestep
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(
            32,
            return_sequences=False
        ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(input_shape[1], activation="linear")
    ], name="psi_forecaster")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    return model


def train_lstm_forecaster():
    """
    Train LSTM on PSI time series.
    With only 4 windows we use sliding window augmentation.
    Returns model and predicted Window 5 PSI.
    """
    psi_series = build_psi_series()
    n_features = psi_series.shape[1]

    augmented = []
    targets = []

    np.random.seed(42)
    tf.random.set_seed(42)
    for _ in range(200):
        noise = np.random.normal(0, 0.01, psi_series.shape)
        aug = np.clip(psi_series + noise, 0.0, None)
        augmented.append(aug[:3])
        targets.append(aug[3])

    X_train = np.array(augmented)
    y_train = np.array(targets)

    model = build_lstm_forecaster_model(
        input_shape=(3, n_features)
    )

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True
            )
        ]
    )

    X_pred = psi_series[1:].reshape(1, 3, n_features)
    predicted_psi = np.clip(model.predict(X_pred, verbose=0)[0], 0.0, None)

    predicted_mean = float(np.mean(predicted_psi))
    predicted_max = float(np.max(predicted_psi))

    recommendation = (
        "RETRAIN URGENTLY" if predicted_max > 0.50 else
        "RETRAIN SOON" if predicted_max > 0.25 else
        "MONITOR" if predicted_max > 0.10 else
        "HEALTHY"
    )

    result = {
        "predicted_window": 5,
        "predicted_mean_psi": round(predicted_mean, 4),
        "predicted_max_psi": round(predicted_max, 4),
        "recommendation": recommendation,
        "feature_predictions": predicted_psi.tolist(),
    }

    model.save(SAVED_DIR / "lstm_forecaster.keras")
    joblib.dump(result, SAVED_DIR / "lstm_forecast_result.joblib")

    return model, result


def build_lstm_forecaster_model(input_shape):
    """Alias for build_lstm_model."""
    return build_lstm_model(input_shape)


def get_forecast() -> dict:
    """Load saved forecast or recompute."""
    forecast_path = SAVED_DIR / "lstm_forecast_result.joblib"
    if forecast_path.exists():
        return joblib.load(forecast_path)
    _, result = train_lstm_forecaster()
    return result


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - LSTM PSI Forecaster")
    print("--------------------------------------")

    model, result = train_lstm_forecaster()

    print(f"\nWindow 5 PSI Forecast:")
    print(f"  Predicted mean PSI: {result['predicted_mean_psi']:.4f}")
    print(f"  Predicted max PSI:  {result['predicted_max_psi']:.4f}")
    print(f"  Recommendation:     {result['recommendation']}")

    assert result["predicted_max_psi"] > 0, "Prediction should be positive"
    assert result["recommendation"] in [
        "RETRAIN URGENTLY", "RETRAIN SOON", "MONITOR", "HEALTHY"
    ]
    print("\nALL LSTM ASSERTIONS PASSED")
