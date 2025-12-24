from fastapi import FastAPI
from app.api.predict import router as predict_router
from app.core.database import init_db

app = FastAPI(
    title="Decision API",
    description="End-to-end customer churn prediction system",
    version="1.0.0"
)

init_db()

app.include_router(predict_router)


@app.get("/")
def health_check():
    return {"status": "API is running"}
