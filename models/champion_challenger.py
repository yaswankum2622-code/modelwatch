"""
ModelWatch | models/champion_challenger.py | Retrain on latest data and compare
"""

import sys
import joblib
import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "modelwatch.db"
SAVED_DIR = Path(__file__).parent / "saved"


def retrain_on_recent_data(
    recent_windows: list = None,
    test_window: int = 4
) -> dict:
    """
    Train challenger model on recent windows.
    Evaluate both champion and challenger on test window.
    Return comparison and decision.
    """
    if recent_windows is None:
        recent_windows = [3, 4]

    feature_cols = joblib.load(SAVED_DIR / "feature_cols.joblib")
    champion = joblib.load(SAVED_DIR / "lgbm_baseline.joblib")

    conn = sqlite3.connect(DB_PATH)
    window_frames = {
        w: pd.read_sql(
            "SELECT * FROM credit_records WHERE window_id = ?",
            conn,
            params=(w,),
        )
        for w in set(recent_windows + [test_window])
    }
    conn.close()

    test_df_full = window_frames[test_window].copy()
    target_col = "DEFAULT" if "DEFAULT" in test_df_full.columns else "default_label"

    holdout_train, evaluation_df = train_test_split(
        test_df_full,
        test_size=0.3,
        random_state=42,
        stratify=test_df_full[target_col].values.astype(int),
    )

    challenger_parts = []
    for w in recent_windows:
        if w == test_window:
            challenger_parts.append(holdout_train)
        else:
            challenger_parts.append(window_frames[w])
    train_df = pd.concat(challenger_parts, ignore_index=True)

    available = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[available].fillna(0).values
    y_train = train_df[target_col].values.astype(int)
    X_test = evaluation_df[available].fillna(0).values
    y_test = evaluation_df[target_col].values.astype(int)

    print("Training challenger model on recent windows...")
    challenger = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
        verbose=-1
    )
    challenger.fit(X_train, y_train)

    champ_prob = champion.predict_proba(X_test)[:, 1]
    champ_pred = champion.predict(X_test)
    chall_prob = challenger.predict_proba(X_test)[:, 1]
    chall_pred = challenger.predict(X_test)

    champ_auc = float(roc_auc_score(y_test, champ_prob))
    champ_f1 = float(f1_score(y_test, champ_pred, zero_division=0))
    chall_auc = float(roc_auc_score(y_test, chall_prob))
    chall_f1 = float(f1_score(y_test, chall_pred, zero_division=0))

    auc_improvement = chall_auc - champ_auc
    decision = "PROMOTE CHALLENGER" if auc_improvement > 0.01 else "KEEP CHAMPION"

    result = {
        "champion_auc": round(champ_auc, 4),
        "champion_f1": round(champ_f1, 4),
        "challenger_auc": round(chall_auc, 4),
        "challenger_f1": round(chall_f1, 4),
        "auc_improvement": round(auc_improvement, 4),
        "decision": decision,
        "test_window": test_window,
        "train_windows": recent_windows,
    }

    if decision == "PROMOTE CHALLENGER":
        joblib.dump(challenger, SAVED_DIR / "lgbm_challenger.joblib")
        print("Challenger saved: models/saved/lgbm_challenger.joblib")

    joblib.dump(result, SAVED_DIR / "champion_challenger_result.joblib")

    return result


if __name__ == "__main__":
    print("--------------------------------------")
    print(" ModelWatch - Champion vs Challenger")
    print("--------------------------------------")

    result = retrain_on_recent_data()

    print(f"\nChampion  AUC={result['champion_auc']:.4f}  "
          f"F1={result['champion_f1']:.4f}")
    print(f"Challenger AUC={result['challenger_auc']:.4f}  "
          f"F1={result['challenger_f1']:.4f}")
    print(f"Improvement: {result['auc_improvement']:+.4f}")
    print(f"Decision: {result['decision']}")

    assert result["challenger_auc"] > 0
    assert result["decision"] in ["PROMOTE CHALLENGER", "KEEP CHAMPION"]
    print("\nALL CHAMPION-CHALLENGER ASSERTIONS PASSED")
