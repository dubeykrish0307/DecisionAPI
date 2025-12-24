from app.monitoring.drift import detect_drift

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.request import ChurnPredictionRequest
from app.schemas.response import ChurnPredictionResponse
from app.models.registry import get_latest_model

from app.core.database import SessionLocal, PredictionLog

router = APIRouter()

def map_api_to_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map API input fields to model feature names.
    """
    return df.rename(
        columns={
            "tenure_months": "tenure",
            "monthly_charges": "monthlycharges",
            "total_charges": "totalcharges",
            "contract_type": "contract",
            "internet_service": "internetservice",
            "payment_method": "paymentmethod",
        }
    )



@router.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    try:
        # Load latest model
        model, version = get_latest_model()

        # Convert request to DataFrame
        input_df = pd.DataFrame([request.dict()])
        input_df = map_api_to_model_features(input_df)
        drift_info = detect_drift(input_df)


        # Make prediction
        churn_proba = model.predict_proba(input_df)[0][1]

        # Assign risk level
        if churn_proba < 0.3:
            risk_level = "LOW"
        elif churn_proba < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"


        db = SessionLocal()
        try:
            log = PredictionLog(
                gender=request.gender,
                senior_citizen=request.senior_citizen,
                tenure_months=request.tenure_months,
                contract_type=request.contract_type,
                internet_service=request.internet_service,
                monthly_charges=request.monthly_charges,
                total_charges=request.total_charges,
                payment_method=request.payment_method,
                churn_probability=round(churn_proba, 4),
                risk_level=risk_level,
                model_version=version,
            )

            db.add(log)
            db.commit()
        finally:
            db.close()



        return ChurnPredictionResponse(
            churn_probability=round(churn_proba, 4),
            risk_level=risk_level,
            model_version=version
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )

