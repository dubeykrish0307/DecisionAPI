import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw churn dataset:
    - Fix data types
    - Handle missing values
    - Standardize column names
    """
    df = df.copy()

    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Convert totalcharges to numeric (it may contain spaces)
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # Drop rows with missing totalcharges
    df = df.dropna(subset=["totalcharges"])

    # Convert churn to binary
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    return df