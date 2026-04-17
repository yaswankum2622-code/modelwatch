"""
ModelWatch | monitoring/evidently_reports.py | Evidently AI drift reports
"""

import sys
import os
import pandas as pd
import sqlite3
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "modelwatch.db"
SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"
REPORT_DIR = Path(__file__).parent.parent / "reports" / "generated"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_window_df(window_id: int) -> pd.DataFrame:
    """Load window data from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM credit_records WHERE window_id = ?",
        conn,
        params=(window_id,),
    )
    conn.close()
    return df


def generate_drift_report(
    window_id: int,
    save_html: bool = True
) -> dict:
    """
    Generate Evidently AI data drift report.
    Compares Window 1 (baseline) vs specified window.
    Saves HTML report to reports/generated/.
    Returns drift summary dict.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric

    feature_cols = joblib.load(SAVED_DIR / "feature_cols.joblib")

    df_ref = load_window_df(1)[feature_cols]
    df_curr = load_window_df(window_id)[feature_cols]

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(
        reference_data=df_ref,
        current_data=df_curr
    )

    result_dict = report.as_dict()
    dataset_drift = None
    n_drifted = 0
    n_features = len(feature_cols)

    for metric in result_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            res = metric.get("result", {})
            dataset_drift = res.get("dataset_drift", False)
            n_drifted = res.get("number_of_drifted_columns", 0)
            break

    report_path = REPORT_DIR / f"drift_report_window_{window_id}.html"
    if save_html:
        report.save_html(str(report_path))
        print(f"Saved: {report_path}")

    return {
        "window_id": window_id,
        "dataset_drift": bool(dataset_drift),
        "n_drifted": n_drifted,
        "n_features": n_features,
        "drift_pct": round(n_drifted / n_features * 100, 1),
        "report_path": str(report_path),
    }


def generate_all_reports() -> list:
    """Generate Evidently reports for Windows 2, 3, 4."""
    results = []
    for w in [2, 3, 4]:
        print(f"Generating Evidently report for Window {w}...")
        try:
            result = generate_drift_report(w)
            results.append(result)
            print(f"  Window {w}: {result['n_drifted']}/{result['n_features']} "
                  f"features drifted ({result['drift_pct']}%)")
        except Exception as e:
            print(f"  Window {w} failed: {e}")
            results.append({
                "window_id": w,
                "dataset_drift": None,
                "error": str(e)
            })
    return results


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - Evidently AI Reports")
    print("--------------------------------------")
    results = generate_all_reports()
    print("\nReports saved to reports/generated/")
