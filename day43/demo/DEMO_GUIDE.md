# Prediction Demo Screenshot Guide

The project cannot automatically capture your local browser, so use these screenshots after starting the API.

## Screenshot 1 — Swagger API

1. Run `uvicorn app.main:app --reload`.
2. Open `http://127.0.0.1:8000/docs`.
3. Capture the Swagger UI showing `GET /health` and `POST /predict`.

## Screenshot 2 — Prediction

1. Expand `POST /predict`.
2. Click **Try it out**.
3. Paste `sample_request.json`.
4. Click **Execute**.
5. Capture the request and JSON response showing:
   - churn_probability
   - churn_prediction
   - risk_level
   - recommendation

## Screenshot 3 — Health check

Open `http://127.0.0.1:8000/health` and capture the successful response.

## Screenshot 4 — Automated tests

Run:

```bash
pytest -q
```

Capture the terminal showing passing tests.

## Suggested filenames

- `01_swagger.png`
- `02_prediction.png`
- `03_health.png`
- `04_tests.png`
