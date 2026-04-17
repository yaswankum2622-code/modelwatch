"""
ModelWatch | database.db | Database access utilities
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "modelwatch.db"
SCHEMA_PATH = Path(__file__).resolve().with_name("schema.sql")
REPLACEABLE_TABLES = {
    "credit_records_raw",
    "credit_records",
    "window_summary",
    "predictions",
}


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection for the project database."""
    resolved_path = Path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the project tables if they do not exist."""
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    connection.commit()


def replace_table(
    connection: sqlite3.Connection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    """Replace the contents of a managed table with a dataframe."""
    if table_name not in REPLACEABLE_TABLES:
        raise ValueError(f"Unsupported table replacement requested: {table_name}")
    connection.execute(f"DELETE FROM {table_name}")
    if not frame.empty:
        frame.to_sql(table_name, connection, if_exists="append", index=False)
    connection.commit()


def read_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Load a full table into a dataframe."""
    return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
