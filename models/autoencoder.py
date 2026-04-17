"""
ModelWatch | models/autoencoder.py | TF/Keras autoencoder drift detector
"""

import sys
import os
import numpy as np
import pandas as pd
import sqlite3
import joblib
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "modelwatch.db"
SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)


def load_window_features(window_id: int) -> np.ndarray:
    """Load feature matrix for a window."""
    feature_cols = joblib.load(SAVED_DIR / "feature_cols.joblib")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM credit_records WHERE window_id = ?",
        conn,
        params=(window_id,),
    )
    conn.close()
    available = [c for c in feature_cols if c in df.columns]
    return df[available].fillna(0).values


def build_autoencoder(input_dim: int) -> keras.Model:
    """
    Build autoencoder architecture.
    Encoder compresses to bottleneck dimension.
    Decoder reconstructs original features.
    High reconstruction error = data looks different from training.
    """
    bottleneck = max(4, input_dim // 4)

    inputs = keras.Input(shape=(input_dim,))

    x = keras.layers.Dense(input_dim // 2, activation="relu")(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(bottleneck, activation="relu", name="bottleneck")(x)

    x = keras.layers.Dense(input_dim // 2, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


def train_autoencoder():
    """
    Train autoencoder on Window 1 baseline features.
    Saves model and scaler. Returns baseline threshold.
    """
    print("Training autoencoder on Window 1...")
    X_baseline = load_window_features(1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_baseline)

    input_dim = X_scaled.shape[1]
    model = build_autoencoder(input_dim)

    print(f"Architecture: {input_dim} -> {input_dim//2} "
          f"-> {max(4, input_dim//4)} -> {input_dim//2} -> {input_dim}")

    history = model.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
            )
        ],
    )

    reconstructions = model.predict(X_scaled, verbose=0)
    baseline_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
    baseline_mean = float(np.mean(baseline_errors))
    baseline_std = float(np.std(baseline_errors))
    alert_threshold = baseline_mean + 2 * baseline_std

    print(f"Baseline mean error: {baseline_mean:.6f}")
    print(f"Alert threshold:     {alert_threshold:.6f}")

    model.save(SAVED_DIR / "autoencoder.keras")
    joblib.dump(scaler, SAVED_DIR / "ae_scaler.joblib")
    joblib.dump(
        {
            "mean": baseline_mean,
            "std": baseline_std,
            "threshold": alert_threshold,
        },
        SAVED_DIR / "ae_baseline.joblib",
    )

    return model, scaler, alert_threshold


def score_window(window_id: int) -> dict:
    """
    Compute reconstruction error for all records in a window.
    High error = records look different from training baseline.
    """
    model = keras.models.load_model(SAVED_DIR / "autoencoder.keras")
    scaler = joblib.load(SAVED_DIR / "ae_scaler.joblib")
    baseline = joblib.load(SAVED_DIR / "ae_baseline.joblib")

    X = load_window_features(window_id)
    X_scaled = scaler.transform(X)

    reconstructions = model.predict(X_scaled, verbose=0)
    errors = np.mean(np.square(X_scaled - reconstructions), axis=1)

    mean_error = float(np.mean(errors))
    pct_above = float((errors > baseline["threshold"]).mean() * 100)
    drift_ratio = mean_error / baseline["mean"]

    status = (
        "SEVERE" if drift_ratio > 3.0 else
        "MODERATE" if drift_ratio > 1.5 else
        "MILD" if drift_ratio > 1.1 else
        "HEALTHY"
    )

    return {
        "window_id": window_id,
        "mean_error": round(mean_error, 6),
        "baseline_mean": round(baseline["mean"], 6),
        "drift_ratio": round(drift_ratio, 3),
        "pct_above_threshold": round(pct_above, 2),
        "status": status,
        "errors": errors.tolist(),
    }


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch — Autoencoder Training")
    print("--------------------------------------")

    tf.random.set_seed(42)
    np.random.seed(42)
    train_autoencoder()

    print("\nReconstruction errors across windows:")
    for w in [1, 2, 3, 4]:
        result = score_window(w)
        print(f"  Window {w}: mean_error={result['mean_error']:.6f}  "
              f"drift_ratio={result['drift_ratio']:.3f}  "
              f"status={result['status']}")

    w1 = score_window(1)["mean_error"]
    w4 = score_window(4)["mean_error"]
    assert w4 > w1, (
        f"Window 4 error {w4:.6f} should be > Window 1 {w1:.6f}"
    )
    print("\nALL AUTOENCODER ASSERTIONS PASSED")
