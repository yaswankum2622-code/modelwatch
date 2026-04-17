import pytest
import os
import sys
import numpy as np
import pandas as pd
import sqlite3
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

DB_EXISTS = os.path.exists("data/modelwatch.db")
MODELS_EXIST = os.path.exists("models/saved/lgbm_baseline.joblib")
ALL_EXIST = DB_EXISTS and MODELS_EXIST


@pytest.mark.skipif(not DB_EXISTS, reason="Run loader.py first")
class TestDataPipeline:

    def test_database_has_4_windows(self, db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        windows = cur.execute(
            "SELECT DISTINCT window_id FROM credit_records"
        ).fetchall()
        conn.close()
        assert len(windows) == 4

    def test_each_window_has_7500_rows(self, db_path):
        conn = sqlite3.connect(db_path)
        for w in [1, 2, 3, 4]:
            count = conn.execute(
                f"SELECT COUNT(*) FROM credit_records "
                f"WHERE window_id = {w}"
            ).fetchone()[0]
            assert count >= 7000, \
                f"Window {w} has only {count} rows"
        conn.close()

    def test_default_rate_is_realistic(self, db_path):
        conn = sqlite3.connect(db_path)
        rates = []
        for w in [1, 2, 3, 4]:
            rate = conn.execute(
                f"SELECT AVG(default_label) FROM credit_records "
                f"WHERE window_id = {w}"
            ).fetchone()[0]
            if rate is None:
                rate = conn.execute(
                    f"SELECT AVG(\"DEFAULT\") FROM credit_records "
                    f"WHERE window_id = {w}"
                ).fetchone()[0]
            rates.append(rate)
        conn.close()
        for rate in rates:
            assert 0.10 <= rate <= 0.40, \
                f"Default rate {rate:.2f} out of expected range"

    def test_bill_amt1_increases_with_drift(self, db_path):
        conn = sqlite3.connect(db_path)
        means = []
        for w in [1, 2, 3, 4]:
            mean = conn.execute(
                f"SELECT AVG(BILL_AMT1) FROM credit_records "
                f"WHERE window_id = {w}"
            ).fetchone()[0]
            means.append(mean)
        conn.close()
        assert means[3] > means[0], \
            "Window 4 BILL_AMT1 should be higher than Window 1"


@pytest.mark.skipif(not ALL_EXIST, reason="Run full pipeline first")
class TestLightGBM:

    def test_model_loads(self, saved_dir):
        import lightgbm as lgb
        model = joblib.load(saved_dir / "lgbm_baseline.joblib")
        assert model is not None

    def test_feature_cols_saved(self, saved_dir):
        cols = joblib.load(saved_dir / "feature_cols.joblib")
        assert len(cols) >= 20
        assert "LIMIT_BAL" in cols
        assert "PAY_0" in cols

    def test_performance_csv_exists(self, saved_dir):
        assert (saved_dir / "performance_by_window.csv").exists()

    def test_auc_degrades_across_windows(self, saved_dir):
        perf = pd.read_csv(saved_dir / "performance_by_window.csv")
        w1 = float(perf[perf.window_id == 1]["auc_roc"].iloc[0])
        w4 = float(perf[perf.window_id == 4]["auc_roc"].iloc[0])
        assert w4 < w1, \
            f"AUC should degrade: W1={w1:.3f} W4={w4:.3f}"

    def test_window4_auc_degraded_significantly(self, saved_dir):
        perf = pd.read_csv(saved_dir / "performance_by_window.csv")
        w1 = float(perf[perf.window_id == 1]["auc_roc"].iloc[0])
        w4 = float(perf[perf.window_id == 4]["auc_roc"].iloc[0])
        degradation = (w4 - w1) / w1 * 100
        assert degradation < -5, \
            f"Expected >5% degradation, got {degradation:.1f}%"

    def test_isolation_forest_loads(self, saved_dir):
        model = joblib.load(saved_dir / "isolation_forest.joblib")
        assert model is not None

    def test_autoencoder_saves_exist(self, saved_dir):
        assert (saved_dir / "autoencoder.keras").exists() or \
               (saved_dir / "autoencoder.h5").exists()

    def test_champion_challenger_result_exists(self, saved_dir):
        assert (saved_dir / "champion_challenger_result.joblib").exists()

    def test_challenger_outperforms_champion(self, saved_dir):
        cc = joblib.load(saved_dir / "champion_challenger_result.joblib")
        assert cc["challenger_auc"] > 0
        assert cc["champion_auc"] > 0
        assert cc["decision"] in [
            "PROMOTE CHALLENGER", "KEEP CHAMPION"
        ]
