"""
ModelWatch | monitoring/statistical_tests.py | KS, JS divergence, chi-square tests
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
import joblib
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "modelwatch.db"
SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"

CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE",
                    "PAY_0", "PAY_2", "PAY_3",
                    "PAY_4", "PAY_5", "PAY_6"]


def ks_test(baseline: np.ndarray,
            current: np.ndarray) -> dict:
    """KS test for continuous feature drift."""
    stat, pval = ks_2samp(baseline, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(pval), 6),
        "drift": bool(pval < 0.05)
    }


def js_divergence(baseline: np.ndarray,
                  current: np.ndarray,
                  bins: int = 20) -> float:
    """
    Jensen-Shannon divergence between two distributions.
    Returns value between 0 (identical) and 1 (completely different).
    > 0.1 indicates moderate drift.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    range_min = min(baseline.min(), current.min())
    range_max = max(baseline.max(), current.max())

    if range_max == range_min:
        return 0.0

    bins_arr = np.linspace(range_min, range_max, bins + 1)
    eps = 1e-10

    p = np.histogram(baseline, bins=bins_arr)[0].astype(float) + eps
    q = np.histogram(current, bins=bins_arr)[0].astype(float) + eps

    p /= p.sum()
    q /= q.sum()

    return round(float(jensenshannon(p, q)), 6)


def chi_square_test(baseline: np.ndarray,
                    current: np.ndarray) -> dict:
    """Chi-square test for categorical feature drift."""
    all_cats = np.union1d(
        np.unique(baseline),
        np.unique(current)
    )

    baseline_counts = np.array(
        [np.sum(baseline == c) for c in all_cats]
    )
    current_counts = np.array(
        [np.sum(current == c) for c in all_cats]
    )

    baseline_counts = np.maximum(baseline_counts, 1)
    current_counts = np.maximum(current_counts, 1)

    contingency = np.array([baseline_counts, current_counts])
    chi2, pval, dof, _ = chi2_contingency(contingency)

    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(pval), 6),
        "drift": bool(pval < 0.05)
    }


def run_all_tests(window_id: int) -> pd.DataFrame:
    """
    Run KS, JS, and Chi-square tests for all features
    comparing Window 1 baseline vs specified window.
    """
    feature_cols = joblib.load(SAVED_DIR / "feature_cols.joblib")

    conn = sqlite3.connect(DB_PATH)
    df_w1 = pd.read_sql(
        "SELECT * FROM credit_records WHERE window_id = ?",
        conn,
        params=(1,),
    )
    df_wn = pd.read_sql(
        "SELECT * FROM credit_records WHERE window_id = ?",
        conn,
        params=(window_id,),
    )
    conn.close()

    results = []

    for col in feature_cols:
        if col not in df_w1.columns or col not in df_wn.columns:
            continue

        b = df_w1[col].dropna().values
        c = df_wn[col].dropna().values

        if col in CATEGORICAL_COLS:
            test = chi_square_test(b, c)
            method = "chi_square"
            stat = test["chi2_statistic"]
            pval = test["p_value"]
            js_div = js_divergence(b, c)
        else:
            ks = ks_test(b, c)
            method = "ks_test"
            stat = ks["ks_statistic"]
            pval = ks["p_value"]
            js_div = js_divergence(b, c)

        results.append({
            "feature": col,
            "method": method,
            "statistic": stat,
            "p_value": pval,
            "js_divergence": js_div,
            "drift_detected": bool(pval < 0.05),
            "window_id": window_id,
        })

    return pd.DataFrame(results).sort_values(
        "js_divergence", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - Statistical Drift Tests")
    print("--------------------------------------")

    for w in [2, 3, 4]:
        results = run_all_tests(w)
        n_drift = results["drift_detected"].sum()
        print(f"\nWindow {w}: {n_drift}/{len(results)} features with detected drift")
        print(results.head(5)[[
            "feature", "method", "statistic", "p_value", "drift_detected"
        ]].to_string(index=False))

    w4 = run_all_tests(4)
    assert w4["drift_detected"].sum() > 0, "Window 4 should show drift"
    print("\nALL STATISTICAL TEST ASSERTIONS PASSED")
