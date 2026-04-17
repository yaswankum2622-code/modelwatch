import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

DB_EXISTS = os.path.exists("data/modelwatch.db")
MODELS_EXIST = os.path.exists("models/saved/lgbm_baseline.joblib")
ALL_EXIST = DB_EXISTS and MODELS_EXIST


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestSHAPDrift:

    def test_shap_drift_returns_dict(self):
        from monitoring.shap_drift import compute_shap_drift
        result = compute_shap_drift(4)
        assert isinstance(result, dict)
        assert "spearman_correlation" in result
        assert "status" in result
        assert "top_movers" in result

    def test_spearman_between_neg1_and_1(self):
        from monitoring.shap_drift import compute_shap_drift
        result = compute_shap_drift(4)
        corr = result["spearman_correlation"]
        assert -1.0 <= corr <= 1.0

    def test_window4_shap_shows_drift(self):
        from monitoring.shap_drift import compute_shap_drift
        result = compute_shap_drift(4)
        assert result["status"] in ["MODERATE", "SEVERE"], \
            f"Window 4 should show SHAP drift, got {result['status']}"

    def test_shap_status_is_valid(self):
        from monitoring.shap_drift import compute_shap_drift
        valid = {"HEALTHY", "MILD", "MODERATE", "SEVERE"}
        for w in [2, 3, 4]:
            r = compute_shap_drift(w)
            assert r["status"] in valid


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestLSTMForecast:

    def test_forecast_result_exists(self):
        import joblib
        from pathlib import Path
        path = Path("models/saved/lstm_forecast_result.joblib")
        assert path.exists(), "LSTM forecast result not saved"

    def test_forecast_has_required_keys(self):
        import joblib
        from pathlib import Path
        result = joblib.load(
            Path("models/saved/lstm_forecast_result.joblib")
        )
        required = [
            "predicted_window",
            "predicted_mean_psi",
            "predicted_max_psi",
            "recommendation"
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_predicted_psi_positive(self):
        import joblib
        from pathlib import Path
        result = joblib.load(
            Path("models/saved/lstm_forecast_result.joblib")
        )
        assert result["predicted_max_psi"] > 0
        assert result["predicted_mean_psi"] > 0

    def test_recommendation_is_valid(self):
        import joblib
        from pathlib import Path
        result = joblib.load(
            Path("models/saved/lstm_forecast_result.joblib")
        )
        valid = {
            "RETRAIN URGENTLY",
            "RETRAIN SOON",
            "MONITOR",
            "HEALTHY"
        }
        assert result["recommendation"] in valid


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestChampionChallenger:

    def test_cc_result_has_required_keys(self):
        import joblib
        from pathlib import Path
        cc = joblib.load(
            Path("models/saved/champion_challenger_result.joblib")
        )
        required = [
            "champion_auc", "challenger_auc",
            "auc_improvement", "decision"
        ]
        for key in required:
            assert key in cc

    def test_auc_values_in_valid_range(self):
        import joblib
        from pathlib import Path
        cc = joblib.load(
            Path("models/saved/champion_challenger_result.joblib")
        )
        assert 0.4 <= cc["champion_auc"] <= 1.0
        assert 0.4 <= cc["challenger_auc"] <= 1.0

    def test_decision_is_valid(self):
        import joblib
        from pathlib import Path
        cc = joblib.load(
            Path("models/saved/champion_challenger_result.joblib")
        )
        assert cc["decision"] in [
            "PROMOTE CHALLENGER", "KEEP CHAMPION"
        ]

    def test_challenger_promoted_when_better(self):
        import joblib
        from pathlib import Path
        cc = joblib.load(
            Path("models/saved/champion_challenger_result.joblib")
        )
        if cc["auc_improvement"] > 0.01:
            assert cc["decision"] == "PROMOTE CHALLENGER"
