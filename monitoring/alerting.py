"""
ModelWatch | monitoring/alerting.py | Drift alerting rules engine
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

PSI_AMBER_THRESHOLD = 0.10
PSI_RED_THRESHOLD = 0.25
AUC_AMBER_DEGRADATION = -3.0
AUC_RED_DEGRADATION = -8.0
ANOMALY_AMBER_RATE = 8.0
ANOMALY_RED_RATE = 15.0
AE_AMBER_RATIO = 1.5
AE_RED_RATIO = 3.0
SHAP_AMBER_CORR = 0.80
SHAP_RED_CORR = 0.60


def check_psi_alerts(psi_df: pd.DataFrame) -> list:
    """Generate alerts from PSI results."""
    alerts = []
    for _, row in psi_df.iterrows():
        if row["psi"] >= PSI_RED_THRESHOLD:
            alerts.append({
                "level": "RED",
                "type": "DATA_DRIFT",
                "feature": row["feature"],
                "metric": "PSI",
                "value": row["psi"],
                "message": f"Severe data drift in {row['feature']} "
                           f"(PSI={row['psi']:.3f} > {PSI_RED_THRESHOLD}). "
                           f"Retraining strongly recommended.",
                "timestamp": datetime.now().isoformat()
            })
        elif row["psi"] >= PSI_AMBER_THRESHOLD:
            alerts.append({
                "level": "AMBER",
                "type": "DATA_DRIFT",
                "feature": row["feature"],
                "metric": "PSI",
                "value": row["psi"],
                "message": f"Moderate data drift in {row['feature']} "
                           f"(PSI={row['psi']:.3f}). Monitor closely.",
                "timestamp": datetime.now().isoformat()
            })
    return alerts


def check_performance_alerts(
    perf_df: pd.DataFrame,
    window_id: int
) -> list:
    """Generate alerts from performance degradation."""
    from monitoring.performance_tracker import compute_degradation

    alerts = []
    degraded = compute_degradation(perf_df)
    row = degraded[degraded["window_id"] == window_id]

    if row.empty:
        return alerts

    deg = float(row["auc_roc_degradation_pct"].iloc[0])
    auc = float(row["auc_roc"].iloc[0])

    if deg <= AUC_RED_DEGRADATION:
        alerts.append({
            "level": "RED",
            "type": "PERFORMANCE_DEGRADATION",
            "feature": "model_auc",
            "metric": "AUC-ROC",
            "value": auc,
            "message": f"Critical model performance degradation. "
                       f"AUC dropped {deg:.1f}% to {auc:.3f}. "
                       f"Immediate retraining required.",
            "timestamp": datetime.now().isoformat()
        })
    elif deg <= AUC_AMBER_DEGRADATION:
        alerts.append({
            "level": "AMBER",
            "type": "PERFORMANCE_DEGRADATION",
            "feature": "model_auc",
            "metric": "AUC-ROC",
            "value": auc,
            "message": f"Model performance declining. "
                       f"AUC dropped {deg:.1f}% to {auc:.3f}. "
                       f"Schedule retraining review.",
            "timestamp": datetime.now().isoformat()
        })
    return alerts


def check_anomaly_alerts(anomaly_result: dict) -> list:
    """Generate alerts from Isolation Forest anomaly rates."""
    alerts = []
    rate = anomaly_result["anomaly_rate"]
    w = anomaly_result["window_id"]

    if rate >= ANOMALY_RED_RATE:
        alerts.append({
            "level": "RED",
            "type": "ANOMALY_SPIKE",
            "feature": "anomaly_rate",
            "metric": "Isolation Forest",
            "value": rate,
            "message": f"High anomaly rate in Window {w}: "
                       f"{rate:.1f}% of records flagged as anomalous. "
                       f"Significant distribution shift detected.",
            "timestamp": datetime.now().isoformat()
        })
    elif rate >= ANOMALY_AMBER_RATE:
        alerts.append({
            "level": "AMBER",
            "type": "ANOMALY_SPIKE",
            "feature": "anomaly_rate",
            "metric": "Isolation Forest",
            "value": rate,
            "message": f"Elevated anomaly rate in Window {w}: "
                       f"{rate:.1f}%. Monitor for further increase.",
            "timestamp": datetime.now().isoformat()
        })
    return alerts


def check_autoencoder_alerts(ae_result: dict) -> list:
    """Generate alerts from autoencoder reconstruction error."""
    alerts = []
    ratio = ae_result["drift_ratio"]
    w = ae_result["window_id"]

    if ratio >= AE_RED_RATIO:
        alerts.append({
            "level": "RED",
            "type": "RECONSTRUCTION_ERROR",
            "feature": "autoencoder_error",
            "metric": "Autoencoder",
            "value": ratio,
            "message": f"Autoencoder reconstruction error {ratio:.1f}x "
                       f"baseline in Window {w}. Data looks fundamentally "
                       f"different from training distribution.",
            "timestamp": datetime.now().isoformat()
        })
    elif ratio >= AE_AMBER_RATIO:
        alerts.append({
            "level": "AMBER",
            "type": "RECONSTRUCTION_ERROR",
            "feature": "autoencoder_error",
            "metric": "Autoencoder",
            "value": ratio,
            "message": f"Autoencoder reconstruction error elevated at "
                       f"{ratio:.1f}x baseline in Window {w}.",
            "timestamp": datetime.now().isoformat()
        })
    return alerts


def check_shap_alerts(shap_result: dict) -> list:
    """Generate alerts from SHAP importance drift."""
    alerts = []
    corr = shap_result["spearman_correlation"]
    w = shap_result["window_id"]

    if corr < SHAP_RED_CORR:
        alerts.append({
            "level": "RED",
            "type": "BEHAVIOUR_DRIFT",
            "feature": "shap_importance",
            "metric": "SHAP Rank Correlation",
            "value": corr,
            "message": f"Model behaviour has significantly shifted in "
                       f"Window {w}. SHAP rank correlation={corr:.3f}. "
                       f"Model is making decisions differently than baseline.",
            "timestamp": datetime.now().isoformat()
        })
    elif corr < SHAP_AMBER_CORR:
        alerts.append({
            "level": "AMBER",
            "type": "BEHAVIOUR_DRIFT",
            "feature": "shap_importance",
            "metric": "SHAP Rank Correlation",
            "value": corr,
            "message": f"Moderate model behaviour drift in Window {w}. "
                       f"SHAP rank correlation={corr:.3f}.",
            "timestamp": datetime.now().isoformat()
        })
    return alerts


def run_all_alerts(
    window_id: int,
    psi_df: pd.DataFrame = None,
    perf_df: pd.DataFrame = None,
    anomaly_result: dict = None,
    ae_result: dict = None,
    shap_result: dict = None,
) -> list:
    """
    Run all alert checks and return combined alert list.
    Sorted RED first then AMBER.
    """
    all_alerts = []

    if psi_df is not None:
        top_psi = psi_df.head(5)
        all_alerts.extend(check_psi_alerts(top_psi))

    if perf_df is not None:
        all_alerts.extend(check_performance_alerts(perf_df, window_id))

    if anomaly_result is not None:
        all_alerts.extend(check_anomaly_alerts(anomaly_result))

    if ae_result is not None:
        all_alerts.extend(check_autoencoder_alerts(ae_result))

    if shap_result is not None:
        all_alerts.extend(check_shap_alerts(shap_result))

    level_order = {"RED": 0, "AMBER": 1, "GREEN": 2}
    all_alerts.sort(key=lambda x: level_order.get(x["level"], 3))

    return all_alerts


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - Alerting System")
    print("--------------------------------------")

    from monitoring.psi import compute_psi_all_features
    from monitoring.performance_tracker import get_performance_all_windows
    from models.isolation_forest import score_window as iso_score
    from models.autoencoder import score_window as ae_score
    from monitoring.shap_drift import compute_shap_drift

    for w in [2, 3, 4]:
        print(f"\nWindow {w} alerts:")
        psi = compute_psi_all_features(w)
        perf = get_performance_all_windows()
        iso = iso_score(w)
        ae = ae_score(w)
        shap = compute_shap_drift(w)

        alerts = run_all_alerts(
            window_id=w,
            psi_df=psi,
            perf_df=perf,
            anomaly_result=iso,
            ae_result=ae,
            shap_result=shap,
        )

        for a in alerts:
            print(f"  [{a['level']}] {a['type']}: {a['message'][:80]}...")

        if w == 4:
            red_count = sum(1 for a in alerts if a["level"] == "RED")
            assert red_count >= 2, (
                f"Window 4 should have at least 2 RED alerts, got {red_count}"
            )

    print("\nALL ALERTING ASSERTIONS PASSED")
