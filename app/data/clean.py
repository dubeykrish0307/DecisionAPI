import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    # TotalCharges is blank for brand-new customers, which pandas reads as text
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df = df.dropna(subset=["totalcharges"])

    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
    return df
