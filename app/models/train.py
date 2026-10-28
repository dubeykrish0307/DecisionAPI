import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.data.clean import clean_data
from app.data.features import build_feature_pipeline
from app.data.ingest import load_raw_data
from app.models.registry import MODEL_DIR, version_number
from app.monitoring.drift import compute_training_stats

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_next_model_version(model_dir=MODEL_DIR) -> int:
    existing = [version_number(p) for p in model_dir.glob("churn_model_v*.pkl")]
    return max(existing, default=0) + 1


def train_model():
    df = clean_data(load_raw_data())
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # the drift check compares live requests against the training split
    compute_training_stats(X_train)

    clf = Pipeline(steps=[
        ("features", build_feature_pipeline()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    clf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC: {auc:.4f}")

    version = get_next_model_version()
    model_path = MODEL_DIR / f"churn_model_v{version}.pkl"
    joblib.dump(clf, model_path)

    return auc, model_path, f"v{version}"
