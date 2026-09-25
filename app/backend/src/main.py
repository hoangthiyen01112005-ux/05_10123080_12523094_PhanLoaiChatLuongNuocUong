import os
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from database import check_database_connection


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


@app.get("/")
def root():
    return {
        "service": "Water Quality Backend Service",
        "status": "running",
    }


@app.get("/health")
def health():
    database_connected = check_database_connection()

    return {
        "status": "ok",
        "service": "backend",
        "database_connected": database_connected,
        "ai_service_url": AI_SERVICE_URL,
    }