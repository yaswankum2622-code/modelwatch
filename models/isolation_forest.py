"""
ModelWatch | models/isolation_forest.py | Isolation Forest anomaly detection
"""

import sys
import joblib
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from sklearn.ensemble import IsolationForest
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


def train_isolation_forest():
    """
    Train Isolation Forest on Window 1 baseline.
    Contamination = 0.05 (expect ~5% anomalies in clean data).
    """
    print("Training Isolation Forest on Window 1...")
    X_baseline = load_window_features(1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_baseline)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)

    joblib.dump(iso_forest, SAVED_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, SAVED_DIR / "iso_scaler.joblib")

    scores = iso_forest.decision_function(X_scaled)
    labels = iso_forest.predict(X_scaled)
    anomaly_rate = (labels == -1).mean() * 100

    print(f"Baseline anomaly rate: {anomaly_rate:.1f}%")
    print(f"Mean anomaly score:    {scores.mean():.4f}")

    return iso_forest, scaler


def score_window(window_id: int) -> dict:
    """
    Score all records in a window.
    Returns anomaly rate, mean score, and per-record scores.
    """
    iso_forest = joblib.load(SAVED_DIR / "isolation_forest.joblib")
    scaler = joblib.load(SAVED_DIR / "iso_scaler.joblib")

    X = load_window_features(window_id)
    X_scaled = scaler.transform(X)

    scores = iso_forest.decision_function(X_scaled)
    labels = iso_forest.predict(X_scaled)

    anomaly_rate = float((labels == -1).mean() * 100)
    mean_score = float(scores.mean())

    return {
        "window_id": window_id,
        "anomaly_rate": round(anomaly_rate, 2),
        "mean_score": round(mean_score, 4),
        "n_anomalies": int((labels == -1).sum()),
        "n_total": len(labels),
        "scores": scores.tolist(),
        "labels": labels.tolist(),
    }


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch — Isolation Forest")
    print("--------------------------------------")

    train_isolation_forest()

    print("\nAnomaly rates across windows:")
    for w in [1, 2, 3, 4]:
        result = score_window(w)
        print(f"  Window {w}: {result['anomaly_rate']:.1f}% anomalies  "
              f"mean_score={result['mean_score']:.4f}")

    w1 = score_window(1)["anomaly_rate"]
    w4 = score_window(4)["anomaly_rate"]
    assert w4 > w1, (
        f"Window 4 anomaly rate {w4} should be > Window 1 {w1}"
    )
    print("\nALL ISOLATION FOREST ASSERTIONS PASSED")
