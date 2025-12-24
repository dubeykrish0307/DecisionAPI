import joblib
from pathlib import Path
from typing import Tuple

MODEL_DIR = Path("app/models/artifacts")

def get_latest_model() -> Tuple[object, str]:
    """
    Load the latest version of the churn model.
    Returns the model and its version string.
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Model directory does not exist.")
    
    model_files = sorted(MODEL_DIR.glob("churn_model_v*.pkl"))

    if not model_files:
        raise FileNotFoundError("No trained models found.")
    
    latest_model_path = model_files[-1]
    model = joblib.load(latest_model_path)

    version = latest_model_path.stem.replace("churn_model_", "")

    return model, version