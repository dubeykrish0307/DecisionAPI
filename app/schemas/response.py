from pydantic import BaseModel


class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
    model_version: str

