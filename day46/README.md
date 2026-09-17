# Day 46 — Production Monitoring

**Phase:** Production Monitoring  
**Focus:** Logging, API validation, exception handling, prediction tracking, and reliability documentation.

## What this project demonstrates

- Structured application logging
- API input validation
- Safe handling of invalid predictions and runtime exceptions
- Request and prediction output tracking
- Health/status endpoint
- Reliability and monitoring documentation
- Automated tests for valid/invalid API inputs

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python app/main.py
```

API: `http://127.0.0.1:5000`

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Prediction example:

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":[5.1,3.5,1.4,0.2]}"
```

## Monitoring

Application logs are written to `logs/production.log` after the first run.

The `/metrics` endpoint exposes lightweight request counters suitable for local monitoring.

See `reports/LOGGING_VALIDATION_REPORT.md` and `reports/MONITORING_GUIDE.md`.
