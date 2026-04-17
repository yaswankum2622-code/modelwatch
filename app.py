"""
ModelWatch | app.py | Streamlit Space entry point with runtime bootstrap
"""

from __future__ import annotations

from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "modelwatch.db"
CORE_ARTIFACTS = [
    PROJECT_ROOT / "models" / "saved" / "lgbm_baseline.joblib",
    PROJECT_ROOT / "models" / "saved" / "feature_cols.joblib",
    PROJECT_ROOT / "models" / "saved" / "isolation_forest.joblib",
    PROJECT_ROOT / "models" / "saved" / "autoencoder.keras",
    PROJECT_ROOT / "models" / "saved" / "performance_by_window.csv",
]
EXTENDED_ARTIFACTS = [
    PROJECT_ROOT / "models" / "saved" / "lstm_forecast_result.joblib",
    PROJECT_ROOT / "models" / "saved" / "champion_challenger_result.joblib",
]


def _missing(paths: list[Path]) -> list[Path]:
    """Return the subset of paths that do not exist."""
    return [path for path in paths if not path.exists()]


def ensure_runtime_artifacts() -> None:
    """Create the database and model artifacts when a fresh deployment starts."""
    if not DB_PATH.exists():
        print("[ModelWatch] Database missing. Running loader and drift injector.")
        from data.loader import run_loader
        from data.drift_injector import run_drift_injection

        run_loader()
        run_drift_injection()

    if _missing(CORE_ARTIFACTS):
        print("[ModelWatch] Core model artifacts missing. Training baseline detectors.")
        from models.lgbm_model import train_lgbm
        from models.isolation_forest import train_isolation_forest
        from models.autoencoder import train_autoencoder

        train_lgbm()
        train_isolation_forest()
        train_autoencoder()

    if _missing(EXTENDED_ARTIFACTS):
        print("[ModelWatch] Forecast and challenger artifacts missing. Building them now.")
        from models.lstm_forecast import train_lstm_forecaster
        from models.champion_challenger import retrain_on_recent_data

        train_lstm_forecaster()
        retrain_on_recent_data()


if __name__ == "__main__":
    ensure_runtime_artifacts()
    runpy.run_path(
        str(PROJECT_ROOT / "dashboard" / "app.py"),
        run_name="__main__",
    )
