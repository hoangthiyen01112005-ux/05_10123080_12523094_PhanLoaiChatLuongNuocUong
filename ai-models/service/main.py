from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import WaterQualityRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import time
import uuid

import uvicorn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ai-service")


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
        "model_loaded": False,
        "request_id": request.state.request_id,
    }


@app.post("/validate")
def validate_input(data: WaterQualityRequest, request: Request):
    return {
        "status": "valid",
        "request_id": request.state.request_id,
        "data": data.model_dump(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )