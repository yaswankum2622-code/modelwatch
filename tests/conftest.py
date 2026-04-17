import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))


def pytest_configure(config):
    print("\n------------------------------------")
    print(" ModelWatch - Test Suite")
    print("------------------------------------")


@pytest.fixture(scope="session")
def db_path():
    return "data/modelwatch.db"


@pytest.fixture(scope="session")
def saved_dir():
    from pathlib import Path
    return Path("models/saved")


@pytest.fixture(scope="session")
def feature_cols():
    import joblib
    from pathlib import Path
    return joblib.load(Path("models/saved/feature_cols.joblib"))
