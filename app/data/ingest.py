import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/telco_churn.csv")

def load_raw_data() -> pd.DataFrame:
    """
    Load Raw Churn DataSet from CSV.
    Returns a pandas DataFrame
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError("Raw Data FILE NOT FOUND")
    
    df = pd.read_csv(RAW_DATA_PATH)

    return df