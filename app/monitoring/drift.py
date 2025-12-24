import json
from pathlib import Path
import pandas as pd


DRIFT_STATS_PATH = Path("app/monitoring/training_stats.json")


def compute_training_stats(df: pd.DataFrame):
    """
    Compute baseline statistics from training data.
    """
    stats = {}

    numeric_columns = ["tenure", "monthlycharges", "totalcharges"]

    for col in numeric_columns:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std()
        }

    DRIFT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DRIFT_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)

def detect_drift(input_df: pd.DataFrame) -> dict:
    """
    Detect drift by comparing input data against training statistics.
    """
    if not DRIFT_STATS_PATH.exists():
        return {"drift_detected": False, "details": "No baseline stats found"}

    with open(DRIFT_STATS_PATH, "r") as f:
        stats = json.load(f)

    drift_flags = {}

    for col, values in stats.items():
        mean = values["mean"]
        std = values["std"]

        if std == 0:
            drift_flags[col] = False
            continue

        z_score = abs(input_df[col].iloc[0] - mean) / std
        drift_flags[col] = z_score > 3  # 3-sigma rule

    drift_detected = any(drift_flags.values())

    return {
        "drift_detected": drift_detected,
        "feature_drift": drift_flags
    }

