"""
ModelWatch | data.drift_injector | Synthetic drift injection utilities
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import build_window_summary
from database.db import DB_PATH, get_connection, read_table, replace_table

BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
DRIFT_CONFIG = {
    2: {
        "bill_scale": 1.07,
        "limit_scale": 0.99,
        "payment_scale": 0.97,
        "status_probability": 0.18,
        "status_steps": 1,
        "label": "mild increase",
    },
    3: {
        "bill_scale": 1.17,
        "limit_scale": 0.95,
        "payment_scale": 0.92,
        "status_probability": 0.28,
        "status_steps": 2,
        "label": "moderate increase",
    },
    4: {
        "bill_scale": 1.47,
        "limit_scale": 0.90,
        "payment_scale": 0.84,
        "status_probability": 0.40,
        "status_steps": 3,
        "label": "severe increase",
    },
}
RANDOM_SEED = 2026


def _apply_window_drift(
    frame: pd.DataFrame,
    window_id: int,
    config: dict[str, float | int | str],
    generator: np.random.Generator,
) -> pd.DataFrame:
    """Apply one drift profile to a single monitoring window."""
    drifted = frame.copy()
    mask = drifted["window_id"] == window_id
    window_slice = drifted.loc[mask].copy()

    bill_values = window_slice[BILL_COLUMNS].to_numpy(dtype=float) * float(config["bill_scale"])
    drifted.loc[mask, BILL_COLUMNS] = np.rint(bill_values).astype(int)

    payment_values = (
        window_slice[PAY_AMOUNT_COLUMNS].to_numpy(dtype=float) * float(config["payment_scale"])
    )
    drifted.loc[mask, PAY_AMOUNT_COLUMNS] = np.rint(np.clip(payment_values, 0, None)).astype(int)

    limit_balance = window_slice["LIMIT_BAL"].to_numpy(dtype=float) * float(config["limit_scale"])
    drifted.loc[mask, "LIMIT_BAL"] = np.rint(np.clip(limit_balance, 10_000, None) / 1_000) * 1_000

    increments = generator.binomial(
        n=int(config["status_steps"]),
        p=float(config["status_probability"]),
        size=(mask.sum(), len(PAY_STATUS_COLUMNS)),
    )
    updated_status = np.clip(
        window_slice[PAY_STATUS_COLUMNS].to_numpy(dtype=int) + increments,
        -2,
        8,
    )
    drifted.loc[mask, PAY_STATUS_COLUMNS] = updated_status.astype(int)
    drifted["LIMIT_BAL"] = drifted["LIMIT_BAL"].astype(int)
    return drifted


def inject_mild(frame: pd.DataFrame, generator: np.random.Generator) -> pd.DataFrame:
    """Inject the mild drift profile into window two."""
    return _apply_window_drift(frame, window_id=2, config=DRIFT_CONFIG[2], generator=generator)


def inject_moderate(frame: pd.DataFrame, generator: np.random.Generator) -> pd.DataFrame:
    """Inject the moderate drift profile into window three."""
    return _apply_window_drift(frame, window_id=3, config=DRIFT_CONFIG[3], generator=generator)


def inject_severe(frame: pd.DataFrame, generator: np.random.Generator) -> pd.DataFrame:
    """Inject the severe drift profile into window four."""
    return _apply_window_drift(frame, window_id=4, config=DRIFT_CONFIG[4], generator=generator)


def inject_drift(raw_frame: pd.DataFrame, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Apply progressive synthetic drift to windows two through four."""
    generator = np.random.default_rng(random_seed)
    drifted = inject_mild(raw_frame, generator)
    drifted = inject_moderate(drifted, generator)
    drifted = inject_severe(drifted, generator)
    return drifted


def run_drift_injection(db_path: str | Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw records from SQLite, inject drift, and persist the result."""
    with get_connection(db_path) as connection:
        raw_frame = read_table(connection, "credit_records_raw")
        if raw_frame.empty:
            raise ValueError("credit_records_raw is empty. Run the data loader first.")

        drifted_frame = inject_drift(raw_frame)
        summary = pd.concat(
            [
                build_window_summary(raw_frame, dataset_stage="raw"),
                build_window_summary(drifted_frame, dataset_stage="drifted"),
            ],
            ignore_index=True,
        )
        replace_table(connection, "credit_records", drifted_frame)
        replace_table(connection, "window_summary", summary)

    return drifted_frame, build_window_summary(drifted_frame, dataset_stage="drifted")


def main() -> None:
    """Run the drift injector as a command-line entry point."""
    _, summary = run_drift_injection()

    print("\n" + "-" * 54)
    print(" ModelWatch - Drift Injector")
    print("-" * 54)
    for row in summary.itertuples(index=False):
        suffix = ""
        if row.window_id in DRIFT_CONFIG:
            suffix = f"   <- {DRIFT_CONFIG[row.window_id]['label']}"
        print(f"{row.window_name} BILL_AMT1 mean:  {row.bill_amt1_mean:,.0f}{suffix}")
    print("\nALL DRIFT INJECTED")


if __name__ == "__main__":
    main()
