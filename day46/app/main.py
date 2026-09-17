import logging
import time
from pathlib import Path
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError, field_validator

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "production.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("production-monitor")

app = Flask(__name__)
metrics = {
    "total_requests": 0,
    "prediction_requests": 0,
    "successful_predictions": 0,
    "validation_errors": 0,
    "prediction_errors": 0,
    "total_latency_ms": 0.0,
}

class PredictionInput(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)

    @field_validator("features")
    @classmethod
    def validate_features(cls, values):
        if any(not (-1000 <= x <= 1000) for x in values):
            raise ValueError("Each feature must be between -1000 and 1000.")
        return values

def safe_predict(features):
    # Deterministic demo predictor; replace with a trained model in production.
    score = 0.25 * features[0] + 0.15 * features[1] + 0.45 * features[2] + 0.15 * features[3]
    if score < 1.0:
        label = "low"
    elif score < 2.5:
        label = "medium"
    else:
        label = "high"
    return {"label": label, "score": round(float(score), 4)}

@app.before_request
def before_request():
    request.start_time = time.perf_counter()
    metrics["total_requests"] += 1

@app.after_request
def after_request(response):
    latency_ms = (time.perf_counter() - request.start_time) * 1000
    metrics["total_latency_ms"] += latency_ms
    logger.info(
        "request method=%s path=%s status=%s latency_ms=%.2f",
        request.method, request.path, response.status_code, latency_ms
    )
    return response

@app.get("/health")
def health():
    logger.info("health_check status=healthy")
    return jsonify({"status": "healthy", "service": "day46-production-monitoring"})

@app.get("/metrics")
def get_metrics():
    avg = metrics["total_latency_ms"] / metrics["total_requests"] if metrics["total_requests"] else 0
    return jsonify({
        **metrics,
        "average_latency_ms": round(avg, 2)
    })

@app.post("/predict")
def predict():
    metrics["prediction_requests"] += 1
    request_id = f"req-{int(time.time() * 1000)}"
    payload = request.get_json(silent=True)

    if payload is None:
        metrics["validation_errors"] += 1
        logger.warning("validation_error request_id=%s reason=invalid_json", request_id)
        return jsonify({"error": "Request body must be valid JSON.", "request_id": request_id}), 400

    try:
        data = PredictionInput.model_validate(payload)
        logger.info("prediction_request request_id=%s features_count=%s", request_id, len(data.features))
    except ValidationError as exc:
        metrics["validation_errors"] += 1
        logger.warning("validation_error request_id=%s details=%s", request_id, exc.errors())
        return jsonify({
            "error": "Invalid input data.",
            "details": exc.errors(),
            "request_id": request_id
        }), 422

    try:
        output = safe_predict(data.features)
        if not output or "label" not in output or "score" not in output:
            raise ValueError("Model returned an invalid prediction structure.")
        if not isinstance(output["score"], (int, float)):
            raise ValueError("Prediction score is not numeric.")

        metrics["successful_predictions"] += 1
        logger.info(
            "prediction_output request_id=%s label=%s score=%s",
            request_id, output["label"], output["score"]
        )
        return jsonify({"request_id": request_id, "prediction": output})
    except Exception:
        metrics["prediction_errors"] += 1
        logger.exception("prediction_error request_id=%s", request_id)
        return jsonify({
            "error": "Prediction failed safely. Please retry later.",
            "request_id": request_id
        }), 500

@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    logger.exception("unhandled_exception path=%s", request.path)
    return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    logger.info("application_startup service=day46-production-monitoring")
    app.run(host="127.0.0.1", port=5000, debug=False)
