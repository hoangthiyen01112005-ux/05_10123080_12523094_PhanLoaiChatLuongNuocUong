import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Water Potability AI Service",
    description="API dịch vụ dự đoán chất lượng nước uống",
    version="1.0.0"
)

# 1. Định nghĩa Cấu trúc Dữ liệu Đầu vào (Pydantic Schema dựa trên 9 Features)
class WaterInput(BaseModel):
    ph: float = Field(..., example=7.0)
    Hardness: float = Field(..., example=204.89)
    Solids: float = Field(..., example=20791.32)
    Chloramines: float = Field(..., example=7.30)
    Sulfate: float = Field(..., example=368.51)
    Conductivity: float = Field(..., example=564.30)
    Organic_carbon: float = Field(..., example=10.37)
    Trihalomethanes: float = Field(..., example=86.99)
    Turbidity: float = Field(..., example=2.96)

# 2. Load Mô hình đã huấn luyện
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ AI Service đã load mô hình thành công từ {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"⚠️ Cảnh báo: Không thể load mô hình tại {MODEL_PATH}. Lỗi: {e}")

@app.get("/")
def health_check():
    """Endpoint kiểm tra trạng thái hoạt động của Service."""
    return {"status": "ok", "service": "Water Potability AI Service"}

@app.post("/predict")
def predict(data: WaterInput):
    """Endpoint nhận 9 chỉ số nước và trả về kết quả dự đoán (0 hoặc 1)."""
    if model is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa sẵn sàng.")
    
    # Chuyển dữ liệu JSON nhận được thành DataFrame đúng định dạng cột
    input_data = pd.DataFrame([data.dict()])
    
    # Thực hiện dự đoán
    prediction = int(model.predict(input_data)[0])
    
    # Lấy xác suất nếu mô hình hỗ trợ
    probability = None
    if hasattr(model, "predict_proba"):
        prob_array = model.predict_proba(input_data)[0]
        probability = float(prob_array[prediction])

    return {
        "prediction": prediction,
        "label": "Uống được (Potable)" if prediction == 1 else "Không nên uống (Non-Potable)",
        "probability": probability
    }