"""
ModelWatch | tests.test_drift_detection | Drift detection tests
"""

from __future__ import annotations

import sqlite3

from data.drift_injector import run_drift_injection


def test_loader_creates_expected_window_sizes(loaded_database) -> None:
    """The loader should split the dataset into four equal windows."""
    with sqlite3.connect(loaded_database) as connection:
        rows = connection.execute(
            """
            SELECT window_id, COUNT(*) AS record_count
            FROM credit_records_raw
            GROUP BY window_id
            ORDER BY window_id
            """
        ).fetchall()

    assert [row[0] for row in rows] == [1, 2, 3, 4]
    assert all(row[1] == 7_500 for row in rows)


def test_drift_injection_increases_bill_amt1_means(loaded_database) -> None:
    """The injector should progressively raise BILL_AMT1 across windows."""
    _, summary = run_drift_injection(db_path=loaded_database)

    means = summary["bill_amt1_mean"].round(0).tolist()
    assert means[0] < means[1] < means[2] < means[3]
    assert means[3] - means[0] > 20_000
