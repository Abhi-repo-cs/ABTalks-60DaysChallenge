import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"

def test_valid_prediction():
    client = app.test_client()
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert "prediction" in response.json

def test_wrong_feature_count():
    client = app.test_client()
    response = client.post("/predict", json={"features": [1, 2]})
    assert response.status_code == 422

def test_out_of_range_feature():
    client = app.test_client()
    response = client.post("/predict", json={"features": [1, 2, 3000, 4]})
    assert response.status_code == 422

def test_invalid_json():
    client = app.test_client()
    response = client.post(
        "/predict",
        data="not-json",
        content_type="application/json"
    )
    assert response.status_code == 400

def test_metrics():
    client = app.test_client()
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_requests" in response.json
