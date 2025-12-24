from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime


DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    gender = Column(String)
    senior_citizen = Column(Integer)
    tenure_months = Column(Integer)
    contract_type = Column(String)
    internet_service = Column(String)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    payment_method = Column(String)

    churn_probability = Column(Float)
    risk_level = Column(String)
    model_version = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
