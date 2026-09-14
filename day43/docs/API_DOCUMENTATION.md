# API Documentation

## 1. Service

**Name:** Customer Churn Prediction API  
**Framework:** FastAPI  
**Model:** Random Forest Classifier  
**Input:** JSON customer attributes  
**Output:** Churn probability, binary prediction, risk level, and retention recommendation.

## 2. Endpoints

### GET `/`

Returns service metadata.

### GET `/health`

Returns service health and whether the model artifact exists.

### POST `/predict`

Generates a real-time churn prediction.

#### Request fields

| Field | Type | Description |
|---|---|---|
| tenure_months | number | Customer tenure in months |
| monthly_charges | number | Current monthly charge |
| total_charges | number | Historical total charges |
| support_tickets | integer | Number of support tickets |
| usage_hours | number | Estimated usage hours |
| satisfaction_score | number | Satisfaction from 0 to 10 |
| contract_type | integer | 0=monthly, 1=annual, 2=two-year |
| payment_delay_days | integer | Recent payment delay days |

#### Response fields

| Field | Type | Description |
|---|---|---|
| churn_probability | number | Probability from 0 to 1 |
| churn_prediction | integer | 0=no churn, 1=churn |
| risk_level | string | Low, Medium, or High |
| recommendation | string | Suggested business action |

## 3. cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "@sample_request.json"
```

On macOS/Linux replace `^` with `\`.

## 4. Architecture flow

1. Client sends customer JSON.
2. FastAPI receives the request.
3. Pydantic validates data types and constraints.
4. The trained Random Forest model produces a class and probability.
5. The API converts probability into a business risk level.
6. A retention recommendation is returned as JSON.

## 5. Error handling

- `422` — request validation failure.
- `500` — model artifact unavailable or server-side model failure.

## 6. Production improvements

- OAuth2/API key authentication
- HTTPS/TLS
- Rate limiting
- Centralized logs
- Prometheus/Grafana monitoring
- Model drift monitoring
- Model versioning
- CI/CD deployment
- Docker/Kubernetes
- Feature store and schema validation
