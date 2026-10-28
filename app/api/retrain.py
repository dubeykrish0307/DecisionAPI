from fastapi import APIRouter, HTTPException

from app.models.train import train_model

router = APIRouter()


@router.post("/retrain")
def retrain_model():
    try:
        auc, model_path, version = train_model()
    except Exception:
        raise HTTPException(status_code=500, detail="Retraining failed")

    return {
        "message": "Model retrained successfully",
        "model_version": version,
        "roc_auc": round(auc, 4),
        "model_path": str(model_path),
    }
