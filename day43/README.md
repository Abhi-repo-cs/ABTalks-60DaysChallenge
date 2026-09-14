# Data Science Deployment — Day 43

## Customer Churn Prediction API

A complete deployment-focused machine learning project that exposes a customer churn classifier through a **FastAPI REST API**.

### Architecture

```text
Customer / Business App
        |
        | JSON POST /predict
        v
   FastAPI API
        |
        v
Validation (Pydantic)
        |
        v
Random Forest Model
        |
        v
Probability + Risk + Recommendation
```

### Project structure

- `train_model.py` — trains the best-performing model and saves it.
- `app/main.py` — FastAPI application and `/predict` endpoint.
- `app/customer_churn_model.joblib` — generated trained model.
- `tests/test_api.py` — API tests.
- `sample_request.json` — ready-to-use request payload.
- `model_metrics.csv` — holdout evaluation metrics.
- `Dockerfile` — containerized deployment.
- `docs/API_DOCUMENTATION.md` — endpoint and architecture documentation.
- `demo/DEMO_GUIDE.md` — screenshot checklist for submission.

## Run locally

### 1. Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Start API

```bash
uvicorn app.main:app --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health: `http://127.0.0.1:8000/health`

### 5. Test with Swagger

Open `/docs`, select `POST /predict`, click **Try it out**, paste the JSON from `sample_request.json`, and execute.

### 6. Run automated tests

```bash
pytest -q
```

## Example response

```json
{
  "churn_probability": 0.78,
  "churn_prediction": 1,
  "risk_level": "High",
  "recommendation": "Prioritize retention outreach and a personalized offer."
}
```

The exact probability depends on the generated model and should be read from the API response rather than copied from this example.

## Docker

```bash
docker build -t customer-churn-api .
docker run -p 8000:8000 customer-churn-api
```

Then visit `http://127.0.0.1:8000/docs`.

## Deployment notes

For production, add authentication/API keys, structured logging, request IDs, model versioning, monitoring, rate limiting, HTTPS, CI/CD, and a proper model registry. Never commit secrets.

## Submission checklist

- [x] Working API code
- [x] API documentation
- [ ] Prediction demo screenshots
- [x] GitHub-ready repository
- [x] LinkedIn reflection draft
