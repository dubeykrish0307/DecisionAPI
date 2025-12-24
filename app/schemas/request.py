from pydantic import BaseModel, Field

class ChurnPredictionRequest(BaseModel):
    gender: str = Field(..., example="Male")
    senior_citizen: int = Field(..., ge=0, le=1, example=0)
    tenure_months: int = Field(..., ge=0, example=18)
    contract_type: str = Field(..., example="Month-to-month")
    internet_service: str = Field(..., example="Fiber optic")
    monthly_charges: float = Field(..., gt=0, example=85.5)
    total_charges: float = Field(..., ge=0, example=1539.0)
    payment_method: str = Field(..., example="Electronic check")
