from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "customer_churn_model.joblib")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-style FastAPI service for real-time customer churn prediction.",
    version="1.0.0"
)

class CustomerRequest(BaseModel):
    tenure_months: float = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    usage_hours: float = Field(..., ge=0)
    satisfaction_score: float = Field(..., ge=0, le=10)
    contract_type: int = Field(..., ge=0, le=2)
    payment_delay_days: int = Field(..., ge=0)

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    recommendation: str

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` from the project root."
        )
    return joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {
        "service": "Customer Churn Prediction API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRequest):
    try:
        model = load_model()
        values = np.array([[
            customer.tenure_months,
            customer.monthly_charges,
            customer.total_charges,
            customer.support_tickets,
            customer.usage_hours,
            customer.satisfaction_score,
            customer.contract_type,
            customer.payment_delay_days
        ]])
        prediction = int(model.predict(values)[0])
        probability = float(model.predict_proba(values)[0][1])

        if probability >= 0.70:
            risk, recommendation = "High", "Prioritize retention outreach and a personalized offer."
        elif probability >= 0.40:
            risk, recommendation = "Medium", "Monitor engagement and consider proactive support."
        else:
            risk, recommendation = "Low", "Maintain normal customer engagement."

        return PredictionResponse(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level=risk,
            recommendation=recommendation
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
