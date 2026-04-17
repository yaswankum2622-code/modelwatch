"""
ModelWatch | ci_setup.py | Synthetic database for CI testing.
Real UCI data is 2.7MB but requires xlrd and Excel parsing.
This generates a realistic synthetic dataset for CI.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

N_PER_WINDOW = 7_500
WINDOW_IDS = [1, 2, 3, 4]
N_TOTAL = N_PER_WINDOW * len(WINDOW_IDS)

FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def _build_raw_frame() -> pd.DataFrame:
    """Generate a baseline-like credit dataset before drift is applied."""
    rows = []
    base_timestamp = pd.Timestamp("2026-01-01 00:00:00")

    for window_id in WINDOW_IDS:
        for row_number in range(1, N_PER_WINDOW + 1):
            record_id = (window_id - 1) * N_PER_WINDOW + row_number

            limit_bal = float(
                np.clip(np.random.lognormal(mean=11.05, sigma=0.58), 10_000, 800_000)
            )
            sex = int(np.random.choice([1, 2], p=[0.39, 0.61]))
            education = int(np.random.choice([1, 2, 3, 4], p=[0.35, 0.47, 0.16, 0.02]))
            marriage = int(np.random.choice([0, 1, 2, 3], p=[0.01, 0.46, 0.49, 0.04]))
            age = int(np.random.randint(21, 76))

            delinquency_bias = np.clip(
                0.08
                + 0.12 * (limit_bal < 120_000)
                + 0.06 * (age > 55)
                + 0.04 * (education >= 3),
                0.05,
                0.35,
            )
            pay_status = np.random.choice(
                [-2, -1, 0, 1, 2, 3],
                size=6,
                p=[
                    0.04,
                    0.06,
                    0.90 - delinquency_bias,
                    delinquency_bias * 0.55,
                    delinquency_bias * 0.30,
                    delinquency_bias * 0.15,
                ],
            ).astype(int)

            bill_anchor = np.clip(np.random.normal(limit_bal * 0.38, 22_000), 0, None)
            bill_noise = np.random.normal(0, 6_500, 6)
            bill_amts = np.clip(bill_anchor + bill_noise + np.linspace(3500, -1500, 6), 0, None)

            pay_ratio = np.clip(np.random.normal(0.18, 0.09), 0.02, 0.65)
            pay_amts = np.clip(bill_amts * pay_ratio + np.random.normal(0, 850, 6), 0, None)

            utilization = bill_amts.mean() / max(limit_bal, 1.0)
            delay_load = np.maximum(pay_status, 0).sum()
            payment_gap = bill_amts.sum() - pay_amts.sum()

            risk_score = (
                -3.55
                + 0.72 * delay_load
                + 1.55 * utilization
                + 0.0000022 * payment_gap
                + 0.22 * (limit_bal < 75_000)
                + 0.16 * (age < 27)
            )
            default_prob = 1.0 / (1.0 + np.exp(-risk_score))
            default_label = int(np.random.random() < default_prob)

            rows.append({
                "record_id": record_id,
                "window_id": window_id,
                "window_name": f"window_{window_id}",
                "window_row_number": row_number,
                "timestamp": (
                    base_timestamp
                    + pd.Timedelta(days=(window_id - 1) * 30)
                    + pd.Timedelta(minutes=row_number - 1)
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "LIMIT_BAL": int(round(limit_bal / 1000) * 1000),
                "SEX": sex,
                "EDUCATION": education,
                "MARRIAGE": marriage,
                "AGE": age,
                "PAY_0": int(pay_status[0]),
                "PAY_2": int(pay_status[1]),
                "PAY_3": int(pay_status[2]),
                "PAY_4": int(pay_status[3]),
                "PAY_5": int(pay_status[4]),
                "PAY_6": int(pay_status[5]),
                "BILL_AMT1": float(round(bill_amts[0], 2)),
                "BILL_AMT2": float(round(bill_amts[1], 2)),
                "BILL_AMT3": float(round(bill_amts[2], 2)),
                "BILL_AMT4": float(round(bill_amts[3], 2)),
                "BILL_AMT5": float(round(bill_amts[4], 2)),
                "BILL_AMT6": float(round(bill_amts[5], 2)),
                "PAY_AMT1": float(round(pay_amts[0], 2)),
                "PAY_AMT2": float(round(pay_amts[1], 2)),
                "PAY_AMT3": float(round(pay_amts[2], 2)),
                "PAY_AMT4": float(round(pay_amts[3], 2)),
                "PAY_AMT5": float(round(pay_amts[4], 2)),
                "PAY_AMT6": float(round(pay_amts[5], 2)),
                "default_label": default_label,
            })

    return pd.DataFrame(rows)


def _inject_drift(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply progressive synthetic drift to windows 2-4."""
    drifted = raw_df.copy()
    rng = np.random.default_rng(2026)

    config = {
        2: {"bill_scale": 1.07, "limit_scale": 0.99, "payment_scale": 0.96, "status_prob": 0.18, "status_steps": 1},
        3: {"bill_scale": 1.18, "limit_scale": 0.95, "payment_scale": 0.92, "status_prob": 0.28, "status_steps": 2},
        4: {"bill_scale": 1.45, "limit_scale": 0.90, "payment_scale": 0.84, "status_prob": 0.40, "status_steps": 3},
    }

    for window_id, params in config.items():
        mask = drifted["window_id"] == window_id

        bills = drifted.loc[mask, BILL_COLS].to_numpy(dtype=float) * params["bill_scale"]
        drifted.loc[mask, BILL_COLS] = np.round(bills, 2)

        pay_amts = drifted.loc[mask, PAY_AMT_COLS].to_numpy(dtype=float) * params["payment_scale"]
        drifted.loc[mask, PAY_AMT_COLS] = np.round(np.clip(pay_amts, 0, None), 2)

        limits = drifted.loc[mask, "LIMIT_BAL"].to_numpy(dtype=float) * params["limit_scale"]
        drifted.loc[mask, "LIMIT_BAL"] = np.round(np.clip(limits, 10_000, None) / 1000) * 1000

        increments = rng.binomial(
            n=params["status_steps"],
            p=params["status_prob"],
            size=(mask.sum(), len(PAY_STATUS_COLS)),
        )
        pay_status = drifted.loc[mask, PAY_STATUS_COLS].to_numpy(dtype=int)
        drifted.loc[mask, PAY_STATUS_COLS] = np.clip(pay_status + increments, -2, 8)

    drifted["LIMIT_BAL"] = drifted["LIMIT_BAL"].astype(int)
    return drifted


def _build_summary(frame: pd.DataFrame, dataset_stage: str) -> pd.DataFrame:
    """Create a monitoring summary table compatible with the app."""
    summary = (
        frame.groupby(["window_id", "window_name"], as_index=False)
        .agg(
            total_records=("record_id", "size"),
            default_rate=("default_label", "mean"),
            bill_amt1_mean=("BILL_AMT1", "mean"),
        )
        .sort_values("window_id")
    )
    summary.insert(0, "dataset_stage", dataset_stage)
    return summary


def _persist(raw_df: pd.DataFrame, drifted_df: pd.DataFrame) -> None:
    """Write the synthetic CI database in the same shape as production."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("reports/generated", exist_ok=True)

    conn = sqlite3.connect("data/modelwatch.db")

    raw_df.to_sql("credit_records_raw", conn, if_exists="replace", index=False)
    drifted_df.to_sql("credit_records", conn, if_exists="replace", index=False)

    for window_id in WINDOW_IDS:
        drifted_df[drifted_df.window_id == window_id].to_sql(
            f"window_{window_id}",
            conn,
            if_exists="replace",
            index=False,
        )

    window_summary = pd.concat(
        [
            _build_summary(raw_df, "raw"),
            _build_summary(drifted_df, "drifted"),
        ],
        ignore_index=True,
    )
    window_summary.to_sql("window_summary", conn, if_exists="replace", index=False)

    predictions = pd.DataFrame(columns=[
        "record_id",
        "window_id",
        "window_name",
        "prediction_proba",
        "prediction_label",
        "actual_label",
        "timestamp",
    ])
    predictions.to_sql("predictions", conn, if_exists="replace", index=False)
    conn.close()


def main() -> None:
    """Generate a complete synthetic database for CI."""
    print("Generating synthetic credit card dataset for CI...")

    raw_df = _build_raw_frame()
    drifted_df = _inject_drift(raw_df)
    _persist(raw_df, drifted_df)

    total = len(drifted_df)
    print(f"Generated {total:,} records across 4 windows")
    print(f"Default rate: {drifted_df['default_label'].mean() * 100:.1f}%")
    print("Saved to: data/modelwatch.db")

    assert len(drifted_df) == N_TOTAL
    assert drifted_df["window_id"].nunique() == 4
    assert set(FEATURE_COLS).issubset(drifted_df.columns)
    assert drifted_df["BILL_AMT1"].mean() > 0

    w1_bill = float(drifted_df[drifted_df.window_id == 1]["BILL_AMT1"].mean())
    w4_bill = float(drifted_df[drifted_df.window_id == 4]["BILL_AMT1"].mean())
    assert w4_bill > w1_bill, "Drift should increase bill amounts"

    w1_delay = float(drifted_df[drifted_df.window_id == 1]["PAY_0"].mean())
    w4_delay = float(drifted_df[drifted_df.window_id == 4]["PAY_0"].mean())
    assert w4_delay > w1_delay, "Drift should increase delinquency"

    print("ALL CI ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
