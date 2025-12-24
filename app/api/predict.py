from fastapi import APIRouter
from app.schemas.request import ChurnPredictionRequest
from app.schemas.response import ChurnPredictionResponse

router = APIRouter()


@router.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    pass
