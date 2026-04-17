"""
ModelWatch | tests.conftest | Shared pytest fixtures
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import run_loader


@pytest.fixture()
def loaded_database(tmp_path: Path) -> Path:
    """Create a temporary SQLite database populated by the loader."""
    db_path = tmp_path / "modelwatch_test.db"
    run_loader(db_path=db_path)
    return db_path
