# Decision API — End-to-End Production ML System (Customer Churn)

A **production-ready Customer Churn Prediction system** that takes raw business data and turns it into a **live decision API**.  
This project demonstrates how machine learning is actually used in real companies — not notebooks, but **systems**.

---

## Live Application

- **Public API**: https://decisionapi.onrender.com  
- **Swagger UI**: https://decisionapi.onrender.com/docs  

---

## Problem Statement — Customer Churn

Customer churn occurs when a customer stops using a service.

In subscription businesses (telecom, SaaS, streaming):
- Retaining customers is cheaper than acquiring new ones
- Early churn prediction enables proactive retention strategies

**Goal**:  
Predict the **probability of churn** and classify customers into **LOW / MEDIUM / HIGH risk**.

---

## High-Level Architecture

Client  
→ FastAPI  
→ Request Validation  
→ Feature Engineering Pipeline  
→ ML Model (Latest Version)  
→ Prediction  
→ Logging (SQLite)  
→ Drift Detection  
→ Optional Retraining  

---

## Project Structure

DecisionAPI/
├── app/
│ ├── main.py
│ ├── api/
│ │ ├── predict.py
│ │ └── retrain.py
│ ├── core/
│ │ └── database.py
│ ├── data/
│ │ ├── ingest.py
│ │ ├── clean.py
│ │ └── features.py
│ ├── models/
│ │ ├── train.py
│ │ └── registry.py
│ ├── monitoring/
│ │ └── drift.py
│ └── schemas/
│ ├── request.py
│ └── response.py
├── data/raw/telco_churn.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── predictions.db
└── README.md


---

## Dataset

**Telco Customer Churn Dataset**

- ~7,000 customers
- Binary target: `Churn`
- Mapping:
  - `Yes → 1`
  - `No  → 0`

---

## Feature Engineering

### Numeric Features
- `tenure`
- `monthlycharges`
- `totalcharges`

→ Standardized using **StandardScaler**

### Categorical Features
- `gender`
- `contract`
- `internetservice`
- `paymentmethod`

→ One-hot encoded using **OneHotEncoder**

All preprocessing is implemented inside a **single scikit-learn Pipeline** to guarantee consistency between training and inference.

---

## Model

- **Logistic Regression**
- Probabilistic output
- Interpretable baseline
- Evaluated using **ROC-AUC**

---

## Model Versioning

Each training run automatically creates a new model version:
`churn_model_v1.pkl`
`churn_model_v2.pkl`
`churn_model_v3.pkl`


The API always loads the **latest available model version** dynamically.

---

## API Endpoints

### Health Check

GET /

Response:
```json
{ "status": "API is running" }
``` 

### Predict Churn 

POST /predict

Request: 
```json
{
  "gender": "Male",
  "senior_citizen": 0,
  "tenure_months": 18,
  "contract_type": "Month-to-month",
  "internet_service": "Fiber optic",
  "monthly_charges": 85.5,
  "total_charges": 1539.0,
  "payment_method": "Electronic check"
}
```
Response: 
```json
{
  "churn_probability": 0.6533,
  "risk_level": "HIGH",
  "model_version": "v2"
}
```

### Retrain Model

POST /retrain

Response:
```json
{
  "message": "Model retrained successfully",
  "model_version": "v3",
  "roc_auc": 0.8421,
  "model_path": "app/models/artifacts/churn_model_v3.pkl"
}
```

### Prediction Logging
Every prediction is stored in SQLite with:

- Input features
- Churn probability
- Risk level
- Model version
- Timestamp

This enables:
- Auditing
- Debugging
- Monitoring
- Future retraining

### Data Drift Detection
- Training data statistics (mean, std) are stored
- Incoming requests are compared using Z-score
- Drift flagged when values exceed 3σ

### Run Locally (Without Docker)
```
git clone https://github.com/dubeykrish0307/DecisionAPI.git
cd DecisionAPI
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Run with Docker
```
docker build -t decision-api .
docker run -p 8000:8000 decision-api
```
Or : 
```
docker-compose up --build
```

### What This Project Demonstrates
- End-to-end ML system design
- Production-grade API development
- Strict schema validation
- Feature engineering discipline
- Model versioning and retraining
- Data drift awareness
- Docker-based deployment
- Real-world ML engineering mindset

### Author
Krish Dubey
Incoming Bachelor’s Student — Computer Science
Focus: Machine Learning Systems & Backend Engineering
