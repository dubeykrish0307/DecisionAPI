import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_feature_pipeline():
    """
    Build a feature engineering pipeline that transforms raw inputs
    into model-ready numeric features.
    """

    numeric_features = [
        "tenure",
        "monthlycharges",
        "totalcharges",
    ]

    categorical_features = [
        "gender",
        "contract",
        "internetservice",
        "paymentmethod",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor