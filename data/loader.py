"""
ModelWatch | data.loader | Data loading utilities
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import DB_PATH, get_connection, initialize_database, replace_table

DATASET_PATH = Path(__file__).resolve().with_name("default of credit card clients.xls")
TARGET_COLUMN = "default payment next month"
WINDOW_SIZE = 7_500
NUM_WINDOWS = 4
RANDOM_SEED = 760
FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
ORDERED_COLUMNS = [
    "record_id",
    "window_id",
    "window_name",
    "window_row_number",
    "timestamp",
    *FEATURE_COLUMNS,
    "default_label",
]


def load_source_dataframe(dataset_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Read and normalize the UCI credit default dataset."""
    frame = pd.read_excel(dataset_path, header=1)
    frame = frame.rename(columns={"ID": "record_id", TARGET_COLUMN: "default_label"})
    missing_columns = [column for column in ["record_id", *FEATURE_COLUMNS, "default_label"] if column not in frame.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_list}")
    return frame[["record_id", *FEATURE_COLUMNS, "default_label"]].copy()


def assign_windows(frame: pd.DataFrame, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Split the dataset into four deterministic monitoring windows."""
    shuffled = frame.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    expected_size = WINDOW_SIZE * NUM_WINDOWS
    if len(shuffled) != expected_size:
        raise ValueError(f"Expected {expected_size} records, found {len(shuffled)}")

    shuffled["window_id"] = (shuffled.index // WINDOW_SIZE) + 1
    shuffled["window_name"] = "window_" + shuffled["window_id"].astype(str)
    shuffled["window_row_number"] = shuffled.groupby("window_id").cumcount() + 1

    base_timestamp = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = (
        base_timestamp
        + pd.to_timedelta((shuffled["window_id"] - 1) * 30, unit="D")
        + pd.to_timedelta(shuffled["window_row_number"] - 1, unit="m")
    )
    shuffled["timestamp"] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    return shuffled[ORDERED_COLUMNS]


def build_window_summary(frame: pd.DataFrame, dataset_stage: str) -> pd.DataFrame:
    """Summarize each monitoring window for logging and database storage."""
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


def validate_windowed_frame(frame: pd.DataFrame) -> None:
    """Validate the loaded dataset before persisting it."""
    if len(FEATURE_COLUMNS) != 23:
        raise AssertionError("Feature count changed unexpectedly")
    if len(frame) != WINDOW_SIZE * NUM_WINDOWS:
        raise AssertionError("Windowed dataset size is incorrect")
    if frame["record_id"].nunique() != len(frame):
        raise AssertionError("Record ids must remain unique")
    if frame[["record_id", *FEATURE_COLUMNS, "default_label"]].isna().any().any():
        raise AssertionError("Dataset contains null values")
    window_counts = frame["window_id"].value_counts().sort_index()
    if not window_counts.eq(WINDOW_SIZE).all():
        raise AssertionError("Each window must contain 7,500 records")


def persist_windowed_dataset(frame: pd.DataFrame, db_path: str | Path = DB_PATH) -> None:
    """Write the raw and active baseline data into SQLite."""
    summary = build_window_summary(frame, dataset_stage="raw")
    with get_connection(db_path) as connection:
        initialize_database(connection)
        replace_table(connection, "credit_records_raw", frame)
        replace_table(connection, "credit_records", frame)
        replace_table(connection, "window_summary", summary)
        replace_table(connection, "predictions", pd.DataFrame())


def run_loader(
    dataset_path: str | Path = DATASET_PATH,
    db_path: str | Path = DB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full data loading pipeline."""
    source_frame = load_source_dataframe(dataset_path)
    windowed_frame = assign_windows(source_frame)
    validate_windowed_frame(windowed_frame)
    persist_windowed_dataset(windowed_frame, db_path=db_path)
    return windowed_frame, build_window_summary(windowed_frame, dataset_stage="raw")


def _format_db_path(db_path: str | Path) -> str:
    path = Path(db_path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    """Run the loader as a command-line entry point."""
    frame, summary = run_loader()

    print("\n" + "-" * 54)
    print(" ModelWatch - Data Loader")
    print("-" * 54)
    print(f"Loaded {len(frame):,} records with {len(FEATURE_COLUMNS)} features")
    print(f"Default rate: {frame['default_label'].mean() * 100:.1f}%")
    for row in summary.itertuples(index=False):
        print(
            f"{row.window_name}: {row.total_records:,} records  "
            f"default_rate={row.default_rate * 100:.1f}%"
        )
    print(f"Written to: {_format_db_path(DB_PATH)}")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
