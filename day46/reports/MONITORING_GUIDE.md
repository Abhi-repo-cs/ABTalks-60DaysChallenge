# Monitoring Guide

## Endpoints

### Health

`GET /health`

Expected response:

```json
{"status":"healthy","service":"day46-production-monitoring"}
```

### Metrics

`GET /metrics`

Use this to inspect request volume, prediction success/failure counts, validation errors, and average latency.

### Prediction

`POST /predict`

Body:

```json
{"features":[5.1,3.5,1.4,0.2]}
```

## Suggested operational checks

1. Confirm `/health` returns HTTP 200.
2. Review logs for repeated `validation_error` events.
3. Review logs for `prediction_error` and `unhandled_exception`.
4. Watch average latency for sudden increases.
5. Compare prediction requests with successful predictions.
6. Investigate unusual error spikes before deployment changes.

## Example monitoring sequence

```bash
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/metrics
```

Then submit both valid and invalid prediction requests and inspect `logs/production.log`.
