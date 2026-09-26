import os
import httpx
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError
from typing import Optional

from database import check_database_connection
from history import get_prediction_history, save_prediction_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("backend")


app = FastAPI(
    title="Water Quality Backend Service",
    description="Backend API Gateway kết nối Frontend với AI Service",
    version="1.0.0",
)


# Cấu hình CORS để Frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# URL AI Service
AI_SERVICE_URL = os.getenv(
    "AI_SERVICE_URL",
    "http://ai-service:8000",
)

class WaterInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float
    model_type: Optional[str] = "rf"

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


@app.exception_handler(PyMongoError)
async def database_exception_handler(
    request: Request,
    exc: PyMongoError,
):
    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4()),
    )

    logger.exception(
        "request_id=%s database_error path=%s",
        request_id,
        request.url.path,
    )

    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "message": "Database service unavailable",
            "request_id": request_id,
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
        "service": "Water Quality Backend Service",
        "status": "running",
        "request_id": request.state.request_id,
    }


@app.get("/health")
def health(request: Request):
    database_connected = check_database_connection()

    return {
        "status": "ok",
        "service": "backend",
        "database_connected": database_connected,
        "ai_service_url": AI_SERVICE_URL,
        "request_id": request.state.request_id,
    }

@app.post("/api/predict")
async def predict_water_potability(data: WaterInput, request: Request):
    request_id = request.state.request_id

    try:
        payload = data.model_dump()

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{AI_SERVICE_URL}/predict",
                json=payload,
                headers={
                    "X-Request-ID": request_id
                },
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Lỗi phản hồi từ AI Service",
            )

        prediction_result = response.json()

        features = payload.copy()
        features.pop("model_type", None)

        save_prediction_history(
            request_id=request_id,
            features=features,
            prediction=prediction_result["prediction"],
            probability=prediction_result.get("probability"),
            model_version=prediction_result.get("model_version"),
        )

        prediction_result["request_id"] = request_id

        return prediction_result

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Không thể kết nối đến AI Service: {exc}",
        )

@app.get("/api/history")
def history(
    request: Request,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
):
    items = get_prediction_history(limit=limit)

    return {
        "status": "ok",
        "count": len(items),
        "data": items,
        "request_id": request.state.request_id,
    }