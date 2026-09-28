# API Usage Guide

## Overview

The API exposes machine-learning and operational functionality to the dashboard or external clients.

> Replace the example endpoint names and schemas below with the exact endpoints implemented in the project before publishing the final README.

## Base URL

Local development:

```text
http://localhost:8000
```

Production:

```text
<YOUR_DEPLOYED_API_URL>
```

## 1. Health Check

### Request

```http
GET /health
```

### Example

```bash
curl http://localhost:8000/health
```

### Expected response

```json
{
  "status": "healthy"
}
```

## 2. Prediction

### Request

```http
POST /predict
Content-Type: application/json
```

### Example body

```json
{
  "feature_1": 10,
  "feature_2": 25,
  "feature_3": 0.72
}
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{"feature_1":10,"feature_2":25,"feature_3":0.72}"
```

### Example response

```json
{
  "prediction": 1,
  "probability": 0.84,
  "status": "high_risk"
}
```

## 3. Python Client Example

```python
import requests

url = "http://localhost:8000/predict"

payload = {
    "feature_1": 10,
    "feature_2": 25,
    "feature_3": 0.72
}

response = requests.post(url, json=payload, timeout=10)
response.raise_for_status()

print(response.json())
```

## 4. Validation Errors

A client should expect a validation response when required fields are missing or invalid.

Example:

```json
{
  "detail": [
    {
      "loc": ["body", "feature_1"],
      "msg": "field required"
    }
  ]
}
```

The exact schema depends on the API framework and validation implementation.

## 5. Error Handling

Recommended status codes:

| Status | Meaning |
|---|---|
| 200 | Successful request |
| 400 | Invalid request |
| 422 | Schema / validation failure |
| 500 | Unexpected server error |
| 503 | Service or model unavailable |

## 6. API Contract Checklist

Before final submission, verify:

- [ ] Endpoint names match the implementation.
- [ ] Request schema matches the model.
- [ ] Response schema matches the implementation.
- [ ] Validation behavior is documented.
- [ ] Error responses are documented.
- [ ] Health endpoint works.
- [ ] Production URL is correct.
- [ ] Authentication requirements are documented if applicable.
- [ ] No secrets are included in examples.

## 7. Interactive API Documentation

If the application uses FastAPI, interactive documentation is commonly available at:

```text
/docs
/redoc
```

For example:

```text
http://localhost:8000/docs
```

Use the generated OpenAPI documentation as the source of truth for the final endpoint list.
