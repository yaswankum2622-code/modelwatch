"""
ModelWatch | monitoring/psi.py | Population Stability Index drift detection
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "modelwatch.db"
SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"

PSI_GREEN = 0.10
PSI_AMBER = 0.25


def compute_psi_single(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10
) -> float:
    """
    Compute PSI for a single feature.
    PSI = sum((current_pct - baseline_pct) * ln(current_pct / baseline_pct))

    PSI < 0.10  -> no significant change
    PSI 0.10-0.25 -> moderate change - monitor
    PSI > 0.25  -> significant change - consider retraining
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    breakpoints = np.nanpercentile(
        baseline,
        np.linspace(0, 100, bins + 1)
    )
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts = np.histogram(current, bins=breakpoints)[0]

    eps = 1e-4
    bucket_count = len(breakpoints) - 1
    baseline_pct = (baseline_counts + eps) / (len(baseline) + eps * bucket_count)
    current_pct = (current_counts + eps) / (len(current) + eps * bucket_count)

    psi = np.sum(
        (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    )

    return round(float(psi), 6)


def compute_psi_all_features(
    window_id: int,
    feature_cols: list = None
) -> pd.DataFrame:
    """
    Compute PSI for every feature comparing Window 1 (baseline)
    against the specified window.
    Returns DataFrame sorted by PSI descending.
    """
    if feature_cols is None:
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
        psi = compute_psi_single(
            df_w1[col].dropna().values,
            df_wn[col].dropna().values
        )
        status = (
            "RED" if psi >= PSI_AMBER else
            "AMBER" if psi >= PSI_GREEN else
            "GREEN"
        )
        results.append({
            "feature": col,
            "psi": psi,
            "status": status,
            "window_id": window_id,
        })

    return pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)


def get_overall_psi_status(psi_df: pd.DataFrame) -> str:
    """Overall drift status based on worst feature PSI."""
    max_psi = psi_df["psi"].max()
    if max_psi >= PSI_AMBER:
        return "RED"
    if max_psi >= PSI_GREEN:
        return "AMBER"
    return "GREEN"


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - PSI Drift Detection")
    print("--------------------------------------")

    for w in [2, 3, 4]:
        psi_df = compute_psi_all_features(w)
        status = get_overall_psi_status(psi_df)
        top3 = psi_df.head(3)
        print(f"\nWindow {w} vs Baseline - Overall: {status}")
        for _, row in top3.iterrows():
            print(f"  {row['feature']:<15} PSI={row['psi']:.4f}  {row['status']}")

    w2 = compute_psi_all_features(2)["psi"].mean()
    w4 = compute_psi_all_features(4)["psi"].mean()
    assert w4 > w2, "Window 4 PSI should be higher than Window 2"
    print("\nALL PSI ASSERTIONS PASSED")
