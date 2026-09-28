from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import WaterQualityRequest

import json
import logging
import time
import uuid
from pathlib import Path
import joblib
import pandas as pd
from pydantic import Field
from typing import Optional
import uvicorn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ai-service")

class PredictionRequest(WaterQualityRequest):
    model_type: Optional[str] = Field(
        default="rf",
        description="Model: lr, svm, rf hoặc knn",
    )

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

MODEL_PATHS = {
    "lr": MODELS_DIR / "logistic_regression.joblib",
    "svm": MODELS_DIR / "svm.joblib",
    "rf": MODELS_DIR / "rf_model.joblib",
    "knn": MODELS_DIR / "knn_model.joblib",
}

MODEL_NAMES = {
    "lr": "Logistic Regression",
    "svm": "Support Vector Machine",
    "rf": "Random Forest",
    "knn": "K-Nearest Neighbors",
}

models = {}


def load_models():
    models.clear()

    for model_key, model_path in MODEL_PATHS.items():
        if model_path.exists():
            models[model_key] = joblib.load(model_path)
            logger.info(
                "Loaded model=%s path=%s",
                model_key,
                model_path,
            )
        else:
            logger.warning(
                "Model file not found: %s",
                model_path,
            )


load_models()
# Đường dẫn tới metadata.json
METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "metadata.json"
)


app = FastAPI(
    title="Water Quality AI Service",
    description="AI Service for Water Potability Classification",
    version="1.0.0",
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID")

    if not request_id:
        request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4()),
    )

    logger.warning(
        "request_id=%s validation_error path=%s",
        request_id,
        request.url.path,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Invalid input data",
            "request_id": request_id,
            "errors": exc.errors(),
        },
        headers={
            "X-Request-ID": request_id,
        },
    )


@app.exception_handler(Exception)
async def system_exception_handler(
    request: Request,
    exc: Exception,
):
    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4()),
    )

    logger.exception(
        "request_id=%s system_error path=%s",
        request_id,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "request_id": request_id,
        },
        headers={
            "X-Request-ID": request_id,
        },
    )


@app.get("/")
def root(request: Request):
    return {
        "service": "Water Quality AI Service",
        "status": "running",
        "request_id": request.state.request_id,
    }


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "service": "ai-service",
        "model_loaded": len(models) > 0,
        "available_models": list(models.keys()),
        "model_count": len(models),
        "request_id": request.state.request_id,
    }


@app.get("/model-info")
def model_info(request: Request):
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model metadata is not available",
                "request_id": request.state.request_id,
            },
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return {
        "status": "ok",
        "request_id": request.state.request_id,
        "model": metadata,
    }


@app.post("/validate")
def validate_input(
    data: WaterQualityRequest,
    request: Request,
):
    return {
        "status": "valid",
        "request_id": request.state.request_id,
        "data": data.model_dump(),
    }

@app.post("/predict")
def predict(
    data: PredictionRequest,
    request: Request,
):
    selected_model = (
        data.model_type.lower()
        if data.model_type
        else "rf"
    )

    if selected_model not in MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "model_type không hợp lệ. "
                    "Chỉ chấp nhận: lr, svm, rf, knn"
                ),
                "request_id": request.state.request_id,
            },
        )

    if selected_model not in models:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    f"Model '{selected_model}' "
                    "chưa được load"
                ),
                "request_id": request.state.request_id,
            },
        )

    model = models[selected_model]

    input_dict = data.model_dump()
    input_dict.pop("model_type", None)

    input_data = pd.DataFrame([input_dict])

    prediction = int(
        model.predict(input_data)[0]
    )

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            input_data
        )[0]

        probability = float(
            probabilities[prediction]
        )

        model_version = None

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        model_version = metadata.get("version")

    return {
        "prediction": prediction,
        "label": (
            "Uống được (Potable)"
            if prediction == 1
            else "Không nên uống (Non-Potable)"
        ),
        "probability": (
            round(probability, 4)
            if probability is not None
            else None
        ),
        "model_type": selected_model,
        "model_used": MODEL_NAMES[selected_model],
        "model_version": model_version,
        "request_id": request.state.request_id,
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )