import os
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Water Potability AI Service",
    description="API dịch vụ dự đoán chất lượng nước uống hỗ trợ đa mô hình (KNN & Random Forest)",
    version="2.0.0"
)

# 1. Định nghĩa Cấu trúc Dữ liệu Đầu vào (Thêm thuộc tính model_type)
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
    model_type: Optional[str] = Field("rf", example="rf", description="Loại mô hình sử dụng: 'rf' (Random Forest) hoặc 'knn' (K-Nearest Neighbors)")

# 2. Đường dẫn tới các file mô hình
BASE_DIR = os.path.dirname(__file__)
RF_MODEL_PATH = os.path.join(BASE_DIR, "../models/rf_model.joblib")
KNN_MODEL_PATH = os.path.join(BASE_DIR, "../models/knn_model.joblib")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "../models/model.joblib")

# Load tập trung các mô hình
models = {}

def load_models():
    # Load Random Forest
    if os.path.exists(RF_MODEL_PATH):
        models["rf"] = joblib.load(RF_MODEL_PATH)
        print("✅ Load Random Forest Model thành công.")
    elif os.path.exists(DEFAULT_MODEL_PATH):
        models["rf"] = joblib.load(DEFAULT_MODEL_PATH)

    # Load KNN
    if os.path.exists(KNN_MODEL_PATH):
        models["knn"] = joblib.load(KNN_MODEL_PATH)
        print("✅ Load KNN Model thành công.")

load_models()

@app.get("/")
def health_check():
    """Endpoint kiểm tra trạng thái hoạt động của Service."""
    return {
        "status": "ok", 
        "service": "Water Potability AI Service",
        "available_models": list(models.keys())
    }

@app.get("/models")
def get_available_models():
    """Endpoint trả về danh sách các mô hình khả dụng cho Frontend hiển thị."""
    return {
        "models": [
            {"id": "rf", "name": "Random Forest Classifier"},
            {"id": "knn", "name": "K-Nearest Neighbors (KNN)"}
        ]
    }

@app.post("/predict")
def predict(data: WaterInput):
    """Endpoint nhận 9 chỉ số nước + loại mô hình và trả về kết quả dự đoán (0 hoặc 1)."""
    selected_model_key = data.model_type.lower() if data.model_type else "rf"
    
    # Kiểm tra xem mô hình yêu cầu có sẵn sàng không
    if selected_model_key not in models:
        # Nếu truyền sai key hoặc mô hình chưa load được, dùng mặc định 'rf' hoặc báo lỗi
        if "rf" in models:
            selected_model_key = "rf"
        else:
            raise HTTPException(status_code=500, detail=f"Mô hình '{selected_model_key}' chưa sẵn sàng.")

    model = models[selected_model_key]

    # Chuyển dữ liệu JSON đầu vào thành DataFrame (loại bỏ trường model_type khi đưa vào mô hình)
    input_dict = data.dict()
    input_dict.pop("model_type", None)
    
    input_data = pd.DataFrame([input_dict])
    
    # Thực hiện dự đoán
    prediction = int(model.predict(input_data)[0])
    
    # Lấy xác suất độ tin cậy
    probability = None
    if hasattr(model, "predict_proba"):
        prob_array = model.predict_proba(input_data)[0]
        probability = float(prob_array[prediction])

    return {
        "prediction": prediction,
        "label": "Uống được (Potable)" if prediction == 1 else "Không nên uống (Non-Potable)",
        "probability": round(probability, 4) if probability is not None else None,
        "model_used": "Random Forest" if selected_model_key == "rf" else "KNN"
    }