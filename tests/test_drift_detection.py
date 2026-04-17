import pytest
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

DB_EXISTS = os.path.exists("data/modelwatch.db")
MODELS_EXIST = os.path.exists("models/saved/lgbm_baseline.joblib")
ALL_EXIST = DB_EXISTS and MODELS_EXIST


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestPSI:

    def test_psi_returns_dataframe(self):
        from monitoring.psi import compute_psi_all_features
        df = compute_psi_all_features(4)
        assert isinstance(df, pd.DataFrame)
        assert "psi" in df.columns
        assert "feature" in df.columns
        assert "status" in df.columns

    def test_psi_values_are_non_negative(self):
        from monitoring.psi import compute_psi_all_features
        df = compute_psi_all_features(4)
        assert (df["psi"] >= 0).all()

    def test_window4_has_high_psi(self):
        from monitoring.psi import compute_psi_all_features
        df = compute_psi_all_features(4)
        assert df["psi"].max() > 0.25, \
            "Window 4 should have at least one RED PSI feature"

    def test_psi_increases_with_drift(self):
        from monitoring.psi import compute_psi_all_features
        psi_w2 = compute_psi_all_features(2)["psi"].mean()
        psi_w4 = compute_psi_all_features(4)["psi"].mean()
        assert psi_w4 > psi_w2, \
            "Window 4 mean PSI should exceed Window 2"

    def test_status_values_are_valid(self):
        from monitoring.psi import compute_psi_all_features
        df = compute_psi_all_features(4)
        valid = {"GREEN", "AMBER", "RED"}
        found = set(df["status"].unique())
        assert found.issubset(valid)

    def test_window4_has_red_features(self):
        from monitoring.psi import compute_psi_all_features
        df = compute_psi_all_features(4)
        assert (df["status"] == "RED").sum() > 0, \
            "Window 4 should have RED features"


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestStatisticalTests:

    def test_run_all_tests_returns_dataframe(self):
        from monitoring.statistical_tests import run_all_tests
        df = run_all_tests(4)
        assert isinstance(df, pd.DataFrame)
        assert "drift_detected" in df.columns

    def test_window4_detects_multiple_drifted_features(self):
        from monitoring.statistical_tests import run_all_tests
        df = run_all_tests(4)
        assert df["drift_detected"].sum() > 5, \
            "Window 4 should show drift in multiple features"

    def test_js_divergence_non_negative(self):
        from monitoring.statistical_tests import run_all_tests
        df = run_all_tests(4)
        assert (df["js_divergence"] >= 0).all()


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestPerformanceTracker:

    def test_returns_4_windows(self):
        from monitoring.performance_tracker import \
            get_performance_all_windows
        df = get_performance_all_windows()
        assert len(df) == 4

    def test_window4_health_is_red_or_amber(self):
        from monitoring.performance_tracker import (
            get_performance_all_windows, get_health_status
        )
        perf = get_performance_all_windows()
        status = get_health_status(perf, 4)
        assert status in ["RED", "AMBER"], \
            f"Window 4 should not be GREEN, got {status}"

    def test_degradation_computed_correctly(self):
        from monitoring.performance_tracker import (
            get_performance_all_windows, compute_degradation
        )
        perf = get_performance_all_windows()
        degraded = compute_degradation(perf)
        w1_deg = float(
            degraded[degraded.window_id == 1][
                "auc_roc_degradation_pct"
            ].iloc[0]
        )
        assert abs(w1_deg) < 0.01, \
            "Window 1 vs itself should have ~0% degradation"


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestAnomalyDetection:

    def test_isolation_forest_anomaly_rate_increases(self):
        from models.isolation_forest import score_window
        w1 = score_window(1)["anomaly_rate"]
        w4 = score_window(4)["anomaly_rate"]
        assert w4 > w1, \
            f"Window 4 anomaly {w4:.1f}% should exceed W1 {w1:.1f}%"

    def test_anomaly_rate_between_0_and_100(self):
        from models.isolation_forest import score_window
        for w in [1, 2, 3, 4]:
            rate = score_window(w)["anomaly_rate"]
            assert 0 <= rate <= 100

    def test_autoencoder_drift_ratio_increases(self):
        from models.autoencoder import score_window as ae_score
        r1 = ae_score(1)["drift_ratio"]
        r4 = ae_score(4)["drift_ratio"]
        assert r4 > r1, \
            f"Window 4 AE ratio {r4:.2f} should exceed W1 {r1:.2f}"

    def test_autoencoder_status_valid(self):
        from models.autoencoder import score_window as ae_score
        valid = {"HEALTHY", "MILD", "MODERATE", "SEVERE"}
        for w in [1, 2, 3, 4]:
            status = ae_score(w)["status"]
            assert status in valid


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestAlerting:

    def test_window4_generates_red_alerts(self):
        from monitoring.psi import compute_psi_all_features
        from monitoring.performance_tracker import \
            get_performance_all_windows
        from models.isolation_forest import score_window as iso_score
        from models.autoencoder import score_window as ae_score
        from monitoring.shap_drift import compute_shap_drift
        from monitoring.alerting import run_all_alerts

        alerts = run_all_alerts(
            window_id=4,
            psi_df=compute_psi_all_features(4),
            perf_df=get_performance_all_windows(),
            anomaly_result=iso_score(4),
            ae_result=ae_score(4),
            shap_result=compute_shap_drift(4),
        )
        red = sum(1 for a in alerts if a["level"] == "RED")
        assert red >= 2, \
            f"Window 4 should have >=2 RED alerts, got {red}"

    def test_alert_has_required_fields(self):
        from monitoring.psi import compute_psi_all_features
        from monitoring.alerting import check_psi_alerts
        psi_df = compute_psi_all_features(4)
        alerts = check_psi_alerts(psi_df)
        if alerts:
            required = ["level", "type", "feature", "message", "timestamp"]
            for field in required:
                assert field in alerts[0], \
                    f"Alert missing field: {field}"
