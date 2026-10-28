import pandas as pd
from fastapi import APIRouter, HTTPException

from app.core.database import PredictionLog, SessionLocal
from app.models.registry import get_latest_model
from app.monitoring.drift import detect_drift
from app.schemas.request import ChurnPredictionRequest
from app.schemas.response import ChurnPredictionResponse

router = APIRouter()

# API field names -> the column names the model was trained on
FEATURE_NAMES = {
    "tenure_months": "tenure",
    "monthly_charges": "monthlycharges",
    "total_charges": "totalcharges",
    "contract_type": "contract",
    "internet_service": "internetservice",
    "payment_method": "paymentmethod",
}


def map_api_to_model_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=FEATURE_NAMES)


def risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    if probability < 0.6:
        return "MEDIUM"
    return "HIGH"


def log_prediction(request: ChurnPredictionRequest, probability: float, risk: str, version: str):
    db = SessionLocal()
    try:
        db.add(PredictionLog(
            **request.model_dump(),
            churn_probability=probability,
            risk_level=risk,
            model_version=version,
        ))
        db.commit()
    finally:
        db.close()


@router.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    try:
        model, version = get_latest_model()
        features = map_api_to_model_features(pd.DataFrame([request.model_dump()]))

        # TODO: return this in the response and store it with the log
        detect_drift(features)

        probability = model.predict_proba(features)[0][1]
        risk = risk_level(probability)
        log_prediction(request, round(probability, 4), risk, version)

        return ChurnPredictionResponse(
            churn_probability=round(probability, 4),
            risk_level=risk,
            model_version=version,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
