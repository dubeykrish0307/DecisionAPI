from functools import lru_cache
from pathlib import Path

import joblib

MODEL_DIR = Path("app/models/artifacts")


def version_number(model_path: Path) -> int:
    return int(model_path.stem.rsplit("_v", 1)[1])


@lru_cache(maxsize=1)
def _load(model_path: Path, modified_ns: int):
    # modified_ns is only part of the cache key, so an overwritten file gets reloaded
    return joblib.load(model_path)


def get_latest_model():
    """Return the newest churn model and its version string, e.g. "v3"."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Model directory does not exist.")

    model_files = list(MODEL_DIR.glob("churn_model_v*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained models found.")

    latest = max(model_files, key=version_number)
    model = _load(latest, latest.stat().st_mtime_ns)
    return model, latest.stem.replace("churn_model_", "")
