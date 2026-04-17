"""
ModelWatch | models/lgbm_model.py | LightGBM credit default production model
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, brier_score_loss, classification_report
)
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "modelwatch.db"
SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COL = "DEFAULT"


def load_window(window_id: int = 1) -> pd.DataFrame:
    """Load a specific window from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT *, default_label AS "DEFAULT"
        FROM credit_records
        WHERE window_id = ?
        """,
        conn,
        params=(window_id,),
    )
    conn.close()
    return df


def prepare_features(df: pd.DataFrame):
    """Return X, y arrays from DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df[TARGET_COL].values.astype(int)
    return X, y, available


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Full accuracy metric set."""
    ks, _ = ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0])
    return {
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ks_stat": round(float(ks), 4),
    }


def train_lgbm():
    """Train LightGBM on Window 1. Save model and scaler."""
    print("Loading Window 1 (baseline)...")
    df = load_window(1)
    X, y, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train):,} samples...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    joblib.dump(model, SAVED_DIR / "lgbm_baseline.joblib")
    joblib.dump(feature_cols, SAVED_DIR / "feature_cols.joblib")
    np.save(SAVED_DIR / "X_test_baseline.npy", X_test)
    np.save(SAVED_DIR / "y_test_baseline.npy", y_test)

    print("\nLightGBM Baseline Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<12} {v}")

    return model, metrics, feature_cols, X_test, y_test


def evaluate_on_window(window_id: int) -> dict:
    """
    Evaluate trained model on any window.
    Returns metrics dict for that window.
    """
    model = joblib.load(SAVED_DIR / "lgbm_baseline.joblib")
    feature_cols = joblib.load(SAVED_DIR / "feature_cols.joblib")

    df = load_window(window_id)
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    y = df[TARGET_COL].values.astype(int)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, y_pred, y_prob)
    metrics["window_id"] = window_id

    return metrics


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch — LightGBM Training")
    print("--------------------------------------")
    model, metrics, feature_cols, X_test, y_test = train_lgbm()

    print("\nEvaluating on all windows...")
    results = []
    for w in [1, 2, 3, 4]:
        m = evaluate_on_window(w)
        results.append(m)
        print(f"  Window {w}: AUC={m['auc_roc']}  "
              f"F1={m['f1']}  KS={m['ks_stat']}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(SAVED_DIR / "performance_by_window.csv", index=False)
    print("\nSaved: models/saved/performance_by_window.csv")

    assert metrics["auc_roc"] >= 0.65, "AUC too low"
    assert os.path.exists(SAVED_DIR / "lgbm_baseline.joblib")
    print("\nALL LGBM ASSERTIONS PASSED")
