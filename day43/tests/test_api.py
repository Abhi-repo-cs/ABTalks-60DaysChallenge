from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

sample_customer = {
    "tenure_months": 8,
    "monthly_charges": 129.50,
    "total_charges": 1036.00,
    "support_tickets": 5,
    "usage_hours": 24.5,
    "satisfaction_score": 4.2,
    "contract_type": 0,
    "payment_delay_days": 8
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_level"] in ["Low", "Medium", "High"]
