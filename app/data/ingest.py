from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/telco_churn.csv")


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)
