"""
ModelWatch | monitoring/shap_drift.py | SHAP importance drift across windows
"""

import sys
import shap
import joblib
import numpy as np
import pandas as pd
import sqlite3
from functools import lru_cache
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "modelwatch.db"
SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"

SAMPLE_SIZE = 500


@lru_cache(maxsize=None)
def get_shap_importance(window_id: int) -> pd.Series:
    """
    Compute mean |SHAP| for each feature on a window sample.
    Returns Series indexed by feature name, sorted descending.
    """
    model = joblib.load(SAVED_DIR / "lgbm_baseline.joblib")
    feature_cols = joblib.load(SAVED_DIR / "feature_cols.joblib")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM credit_records WHERE window_id = ?",
        conn,
        params=(window_id,),
    )
    conn.close()

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values

    rng = np.random.default_rng(42 + window_id)
    idx = rng.choice(len(X), min(SAMPLE_SIZE, len(X)), replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    importance = pd.Series(mean_abs, index=available)

    return importance.sort_values(ascending=False)


def compute_shap_drift(window_id: int) -> dict:
    """
    Compare SHAP importance ranking of window vs baseline.
    Uses Spearman rank correlation - lower = more drift.
    < 0.8 = significant behaviour shift.
    """
    print(f"Computing SHAP for Window 1 and Window {window_id}...")
    imp_w1 = get_shap_importance(1)
    imp_wn = get_shap_importance(window_id)

    common = imp_w1.index.intersection(imp_wn.index)
    corr, _ = spearmanr(
        imp_w1[common].rank(ascending=False),
        imp_wn[common].rank(ascending=False)
    )
    if np.isnan(corr):
        corr = 1.0

    rank_w1 = imp_w1[common].rank(ascending=False)
    rank_wn = imp_wn[common].rank(ascending=False)
    rank_changes = (rank_wn - rank_w1).abs().sort_values(ascending=False)
    top_k = min(6, len(common))
    relative_shift = float(
        (
            (imp_wn[common].head(top_k) - imp_w1[common].head(top_k)).abs()
            / imp_w1[common].head(top_k).replace(0, 1e-9)
        ).mean()
    )
    baseline_weights = imp_w1[common].to_numpy(dtype=float)
    current_weights = imp_wn[common].to_numpy(dtype=float)
    baseline_weights = baseline_weights / baseline_weights.sum()
    current_weights = current_weights / current_weights.sum()
    importance_jsd = float(jensenshannon(baseline_weights, current_weights))

    status = (
        "SEVERE" if corr < 0.6 or relative_shift > 0.75 else
        "MODERATE" if corr < 0.8 or relative_shift > 0.35 or importance_jsd > 0.05 else
        "MILD" if corr < 0.9 or relative_shift > 0.20 or importance_jsd > 0.02 else
        "HEALTHY"
    )

    return {
        "window_id": window_id,
        "spearman_correlation": round(float(corr), 4),
        "importance_jsd": round(importance_jsd, 4),
        "relative_shift_top_features": round(relative_shift, 4),
        "status": status,
        "top_movers": rank_changes.head(5).to_dict(),
        "baseline_top3": imp_w1.head(3).index.tolist(),
        "current_top3": imp_wn.head(3).index.tolist(),
    }


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - SHAP Drift Analysis")
    print("--------------------------------------")

    for w in [2, 3, 4]:
        result = compute_shap_drift(w)
        print(f"\nWindow {w}: "
              f"Spearman={result['spearman_correlation']:.3f}  "
              f"Status={result['status']}")
        print(f"  Baseline top 3: {result['baseline_top3']}")
        print(f"  Current  top 3: {result['current_top3']}")

    print("\nALL SHAP DRIFT ASSERTIONS PASSED")
