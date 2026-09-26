import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Water Quality Backend Service",
    description="Backend API Gateway kết nối Frontend với AI Service",
    version="1.0.0"
)

# Cấu hình CORS để Frontend (Nginx/Browser) gọi API không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lấy URL của AI Service từ biến môi trường (mặc định cho Docker Compose là http://ai-service:8000)
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:8000")

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

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend Gateway đang hoạt động"}

@app.post("/api/predict")
async def predict_water_potability(data: WaterInput):
    """API Gateway chuyển tiếp request sang AI Service để lấy kết quả dự đoán."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{AI_SERVICE_URL}/predict",
                json=data.dict()
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail="Lỗi phản hồi từ AI Service"
                )
                
            return response.json()
            
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503, 
            detail=f"Không thể kết nối đến AI Service: {exc}"
        )