import json
from pathlib import Path

import pandas as pd

DRIFT_STATS_PATH = Path("app/monitoring/training_stats.json")
NUMERIC_COLUMNS = ["tenure", "monthlycharges", "totalcharges"]
Z_THRESHOLD = 3


def compute_training_stats(df: pd.DataFrame):
    stats = {col: {"mean": df[col].mean(), "std": df[col].std()} for col in NUMERIC_COLUMNS}

    DRIFT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)


def detect_drift(input_df: pd.DataFrame) -> dict:
    """Flag numeric inputs that sit more than 3 standard deviations from the training mean."""
    if not DRIFT_STATS_PATH.exists():
        return {"drift_detected": False, "details": "No baseline stats found"}

    with open(DRIFT_STATS_PATH) as f:
        stats = json.load(f)

    flags = {}
    for col, s in stats.items():
        if s["std"] == 0:
            flags[col] = False
        else:
            flags[col] = abs(input_df[col].iloc[0] - s["mean"]) / s["std"] > Z_THRESHOLD

    return {"drift_detected": any(flags.values()), "feature_drift": flags}
