from app.monitoring.drift import compute_training_stats
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from app.data.ingest import load_raw_data
from app.data.clean import clean_data
from app.data.features import build_feature_pipeline

MODEL_DIR = Path("app/models/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_model():
    # Load and prepare data
    df = clean_data(load_raw_data())

    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # Compute drift baseline on training data
    compute_training_stats(X_train)


    # Build pipeline
    feature_pipeline = build_feature_pipeline()

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(
        steps=[
            ("features", feature_pipeline),
            ("model", model)
        ]
    )

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC-AUC: {auc:.4f}")

    # Save model
    model_path = MODEL_DIR / "churn_model_v1.pkl"
    joblib.dump(clf, model_path)

    return auc, model_path