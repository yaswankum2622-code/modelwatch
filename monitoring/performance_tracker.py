"""
ModelWatch | monitoring/performance_tracker.py | Track model performance across windows
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"


def get_performance_all_windows() -> pd.DataFrame:
    """
    Load pre-computed performance CSV or recompute.
    Returns DataFrame with metrics for all 4 windows.
    """
    perf_path = SAVED_DIR / "performance_by_window.csv"

    if perf_path.exists():
        return pd.read_csv(perf_path)

    from models.lgbm_model import evaluate_on_window

    results = []
    for w in [1, 2, 3, 4]:
        m = evaluate_on_window(w)
        results.append(m)

    df = pd.DataFrame(results)
    df.to_csv(perf_path, index=False)
    return df


def compute_degradation(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage degradation from baseline for each metric.
    """
    baseline = perf_df[perf_df["window_id"] == 1].iloc[0]
    result = perf_df.copy()

    for metric in ["auc_roc", "f1", "precision", "recall"]:
        if metric in result.columns:
            base_val = baseline[metric]
            result[f"{metric}_degradation_pct"] = (
                (result[metric] - base_val) / base_val * 100
            ).round(2)

    return result


def get_health_status(perf_df: pd.DataFrame,
                      window_id: int) -> str:
    """
    Return health status for a window based on AUC degradation.
    GREEN:  < 3% degradation
    AMBER:  3-8% degradation
    RED:    > 8% degradation
    """
    degraded = compute_degradation(perf_df)
    row = degraded[degraded["window_id"] == window_id]

    if row.empty:
        return "UNKNOWN"

    deg = float(row["auc_roc_degradation_pct"].iloc[0])

    if deg < -8:
        return "RED"
    if deg < -3:
        return "AMBER"
    return "GREEN"


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - Performance Tracker")
    print("--------------------------------------")

    perf = get_performance_all_windows()
    degraded = compute_degradation(perf)

    for _, row in degraded.iterrows():
        status = get_health_status(perf, int(row["window_id"]))
        deg = row.get("auc_roc_degradation_pct", 0)
        print(f"  Window {int(row['window_id'])}: "
              f"AUC={row['auc_roc']:.4f}  "
              f"Delta={deg:+.1f}%  {status}")

    assert get_health_status(perf, 4) in ["RED", "AMBER"], (
        "Window 4 should show degradation"
    )
    print("\nALL PERFORMANCE TRACKER ASSERTIONS PASSED")
