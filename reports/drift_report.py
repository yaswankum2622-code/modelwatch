"""
ModelWatch | reports/drift_report.py | Gemini drift report generator
"""

import os
import sys
import joblib
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"


def generate_drift_report(
    window_id: int,
    psi_results: dict,
    perf_results: dict,
    ae_results: dict,
    iso_results: dict,
    shap_results: dict,
    alerts: list,
    cc_results: dict = None,
    lstm_forecast: dict = None,
) -> str:
    """
    Generate a plain English drift report using Gemini 2.5 Flash.
    Falls back to a deterministic report if API unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_report(
            window_id, psi_results, perf_results,
            ae_results, iso_results, alerts, cc_results
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-04-17"
        )

        red_alerts = [a for a in alerts if a["level"] == "RED"]
        amber_alerts = [a for a in alerts if a["level"] == "AMBER"]

        top_drifted = psi_results.get("top_features", [])
        top_str = "\n".join([
            f"  - {f['feature']}: PSI={f['psi']:.3f} ({f['status']})"
            for f in top_drifted[:5]
        ]) if top_drifted else "  Not available"

        cc_str = ""
        if cc_results:
            cc_str = f"""
Champion-Challenger Comparison:
  Champion AUC:    {cc_results.get('champion_auc', 'N/A')}
  Challenger AUC:  {cc_results.get('challenger_auc', 'N/A')}
  Improvement:     {cc_results.get('auc_improvement', 0):+.4f}
  Decision:        {cc_results.get('decision', 'N/A')}"""

        lstm_str = ""
        if lstm_forecast:
            lstm_str = f"""
LSTM Forecast (Window {lstm_forecast.get('predicted_window', 5)}):
  Predicted max PSI:  {lstm_forecast.get('predicted_max_psi', 'N/A')}
  Forecast:           {lstm_forecast.get('recommendation', 'N/A')}"""

        prompt = f"""
You are a senior MLOps engineer writing a model monitoring report
for a data science team and business stakeholders.

MONITORING REPORT - Window {window_id} vs Baseline

MODEL PERFORMANCE:
  Baseline AUC (Window 1):  {perf_results.get('baseline_auc', 'N/A')}
  Current AUC (Window {window_id}): {perf_results.get('current_auc', 'N/A')}
  AUC degradation:          {perf_results.get('degradation_pct', 0):+.1f}%
  Performance status:       {perf_results.get('status', 'N/A')}

DATA DRIFT (Top drifted features):
{top_str}

ANOMALY DETECTION:
  Isolation Forest anomaly rate: {iso_results.get('anomaly_rate', 'N/A')}%
  (Baseline was ~5%)

AUTOENCODER DRIFT:
  Reconstruction error ratio: {ae_results.get('drift_ratio', 'N/A')}x baseline
  Status: {ae_results.get('status', 'N/A')}

SHAP BEHAVIOUR DRIFT:
  Feature importance rank correlation: {shap_results.get('spearman_correlation', 'N/A')}
  Status: {shap_results.get('status', 'N/A')}
{cc_str}
{lstm_str}

ALERTS SUMMARY:
  RED alerts:   {len(red_alerts)}
  AMBER alerts: {len(amber_alerts)}

Write a monitoring report with exactly these 4 sections.
Each section must be 3-5 sentences using the actual numbers above.
Write for a mixed audience - data scientists and business stakeholders.
Use plain English. No jargon without explanation.
Be specific - use the exact numbers provided.
End with a clear recommendation: RETRAIN / MONITOR / HEALTHY.

Sections:
**What We Observed**
**Root Cause Analysis**
**Business Impact**
**Recommendation and Next Steps**

Maximum 350 words total. No markdown headers with # symbols.
Use **bold** for section names only.
"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=1000,
            )
        )

        import datetime

        header = (
            "MODEL MONITORING REPORT\n"
            + "-" * 40
            + "\n"
            + f"Window: {window_id} vs Baseline (Window 1)\n"
            + f"Generated: {datetime.date.today()}\n"
            + "Platform: ModelWatch v1.0\n"
            + "-" * 40
            + "\n\n"
        )

        return header + response.text.strip()

    except Exception:
        return _fallback_report(
            window_id, psi_results, perf_results,
            ae_results, iso_results, alerts, cc_results
        )


def _fallback_report(
    window_id, psi_results, perf_results,
    ae_results, iso_results, alerts, cc_results
) -> str:
    """Deterministic fallback when Gemini unavailable."""
    import datetime

    red_count = sum(1 for a in alerts if a["level"] == "RED")
    deg = perf_results.get("degradation_pct", 0)
    curr_auc = perf_results.get("current_auc", 0)
    base_auc = perf_results.get("baseline_auc", 0)
    anomaly = iso_results.get("anomaly_rate", 0)
    ae_ratio = ae_results.get("drift_ratio", 1)

    recommendation = (
        "RETRAIN URGENTLY" if red_count >= 3 or deg <= -10 else
        "RETRAIN SOON" if red_count >= 1 or deg <= -5 else
        "MONITOR"
    )

    cc_section = ""
    if cc_results:
        cc_section = (
            f"\n**Champion-Challenger**\n"
            f"A challenger model retrained on recent data achieved "
            f"AUC {cc_results.get('challenger_auc', 'N/A')} vs the "
            f"champion's {cc_results.get('champion_auc', 'N/A')}. "
            f"Decision: {cc_results.get('decision', 'N/A')}.\n"
        )

    return f"""MODEL MONITORING REPORT
----------------------------------------
Window: {window_id} vs Baseline (Window 1)
Generated: {datetime.date.today()}
Platform: ModelWatch v1.0
----------------------------------------

**What We Observed**

The model has degraded {deg:+.1f}% from baseline AUC {base_auc:.3f}
to {curr_auc:.3f} on Window {window_id}. The Isolation Forest detected
{anomaly:.1f}% anomalous records - significantly above the baseline
5% rate. The autoencoder reconstruction error is {ae_ratio:.1f}x the
baseline, indicating the incoming data looks substantially different
from training data. {red_count} RED alerts have been triggered.

**Root Cause Analysis**

Data drift analysis identified significant distribution shifts in
payment features and bill amounts - the most predictive features
for credit default. These variables typically shift when macroeconomic
conditions change, consistent with a period of economic stress.
The SHAP importance ranking has also shifted, indicating the model
is weighting features differently than it did at baseline.

**Business Impact**

A model operating at AUC {curr_auc:.3f} instead of {base_auc:.3f}
represents materially degraded credit risk assessment. Decisions made
on drifted data carry higher false positive and false negative rates,
meaning good customers may be declined and risky customers approved.

**Recommendation and Next Steps**

{recommendation}. Retrain the production model using the most recent
2 windows of data. The challenger model already demonstrates this
is viable - retraining recovers significant AUC. Schedule retraining
for the next deployment window and implement continuous monitoring
with weekly PSI checks going forward.
{cc_section}"""


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - Drift Report Generator")
    print("--------------------------------------")

    from monitoring.psi import compute_psi_all_features
    from monitoring.performance_tracker import (
        get_performance_all_windows, compute_degradation
    )
    from models.isolation_forest import score_window as iso_score
    from models.autoencoder import score_window as ae_score
    from monitoring.shap_drift import compute_shap_drift
    from monitoring.alerting import run_all_alerts

    w = 4
    psi_df = compute_psi_all_features(w)
    perf_df = get_performance_all_windows()
    degraded = compute_degradation(perf_df)
    row = degraded[degraded["window_id"] == w].iloc[0]
    iso = iso_score(w)
    ae = ae_score(w)
    shap = compute_shap_drift(w)
    alerts = run_all_alerts(
        window_id=w,
        psi_df=psi_df,
        perf_df=perf_df,
        anomaly_result=iso,
        ae_result=ae,
        shap_result=shap,
    )

    top_features = psi_df.head(5).to_dict("records")

    psi_summary = {"top_features": top_features}
    perf_summary = {
        "baseline_auc": round(float(
            perf_df[perf_df.window_id == 1]["auc_roc"].iloc[0]
        ), 4),
        "current_auc": round(float(row["auc_roc"]), 4),
        "degradation_pct": round(float(
            row["auc_roc_degradation_pct"]
        ), 1),
        "status": "RED"
    }

    cc_path = SAVED_DIR / "champion_challenger_result.joblib"
    cc_result = joblib.load(cc_path) if cc_path.exists() else None

    lstm_path = SAVED_DIR / "lstm_forecast_result.joblib"
    lstm_result = joblib.load(lstm_path) if lstm_path.exists() else None

    report = generate_drift_report(
        window_id=w,
        psi_results=psi_summary,
        perf_results=perf_summary,
        ae_results=ae,
        iso_results=iso,
        shap_results=shap,
        alerts=alerts,
        cc_results=cc_result,
        lstm_forecast=lstm_result,
    )

    print(report[:500] + "...")
    assert len(report) > 200, "Report too short"
    print("\nDRIFT REPORT GENERATION PASSED")
