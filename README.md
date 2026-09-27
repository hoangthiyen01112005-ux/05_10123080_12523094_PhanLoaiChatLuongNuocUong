# 🌊 Water Potability Classification System

Hệ thống phân loại chất lượng nước uống dựa trên 9 chỉ số hóa lý của nước.

Dự án sử dụng 4 mô hình Machine Learning:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

## Kiến trúc hệ thống

Người dùng / Browser
|
v
Frontend (HTML/CSS/JS)
|
v
Nginx
|
v
Backend API
/ \
 v v
AI Service MongoDB
|
v
Machine Learning Models
(LR / SVM / KNN / RF)

### Luồng xử lý dự đoán

Browser
↓
Frontend
↓
Nginx
↓
Backend /api/predict
↓
AI Service /predict
↓
Model được chọn
↓
AI Service trả kết quả
↓
Backend lưu lịch sử vào MongoDB
↓
Backend trả kết quả về Frontend

Toàn bộ hệ thống được đóng gói và chạy bằng Docker Compose.

### 1. Bài toán

Mục tiêu
Dự đoán một mẫu nước có thể uống được hay không dựa trên các chỉ số hóa lý.
Đây là bài toán:
Supervised Binary Classification

Target
Cột mục tiêu:
Potability

Ý nghĩa:
| Giá trị | Ý nghĩa |
|---|---|
| 0 | Không uống được - Non-Potable |
| 1 | Uống được - Potable |

### 2. Dataset

Dataset sử dụng:
Water Potability Dataset

Nguồn:
Kaggle

File dữ liệu trong project:
ai-models/data/dataset.zip

Thông tin dataset:

- Số mẫu: 3276
- Tổng số cột: 10
- Số feature: 9
- Target: Potability

### 3. Danh sách 9 Feature

| Feature           | Ý nghĩa                   |
| ----------------- | ------------------------- |
| `ph`              | Độ pH của nước            |
| `Hardness`        | Độ cứng của nước          |
| `Solids`          | Tổng chất rắn hòa tan     |
| `Chloramines`     | Hàm lượng Chloramines     |
| `Sulfate`         | Hàm lượng Sulfate         |
| `Conductivity`    | Độ dẫn điện               |
| `Organic_carbon`  | Hàm lượng carbon hữu cơ   |
| `Trihalomethanes` | Hàm lượng Trihalomethanes |
| `Turbidity`       | Độ đục của nước           |

### 4. Tiền xử lý dữ liệu

Quy trình preprocessing:
Dataset
|
v
Train/Test Split
|
v
Median Imputation
|
v
StandardScaler
|
v
Machine Learning Model

Cấu hình chia dữ liệu:
test_size = 0.2
random_state = 42
stratify = y

Missing value được xử lý bằng:
SimpleImputer(strategy="median")

Chuẩn hóa dữ liệu:
StandardScaler

Quá trình xử lý được đặt trong Pipeline để hạn chế Data Leakage.
Schema input được lưu tại:
ai-models/models/schema.json

### 5. Các mô hình Machine Learning

Project tích hợp 4 mô hình:

### Logistic Regression

Artifact:
ai-models/models/logistic_regression.joblib

### Support Vector Machine

Artifact:
ai-models/models/svm.joblib

### K-Nearest Neighbors

Artifact:
ai-models/models/knn_model.joblib

### Random Forest

Artifact:
ai-models/models/rf_model.joblib

### 6. So sánh hiệu năng 4 mô hình

Kết quả đánh giá trên cùng tập Test gồm 656 mẫu:

| Model               |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |     PR-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.5305 |     0.4217 | **0.5469** |     0.4762 |     0.5488 |     0.4864 |
| SVM                 |     0.6220 |     0.5159 |     0.5078 | **0.5118** |     0.6440 |     0.5662 |
| KNN                 |     0.6113 |     0.5039 |     0.2539 |     0.3377 |     0.6021 |     0.4825 |
| Random Forest       | **0.6540** | **0.6198** |     0.2930 |     0.3979 | **0.6652** | **0.6000** |

Nhận xét:

- Logistic Regression có Recall cao nhất.
- SVM có F1-Score cao nhất.
- Random Forest có Accuracy, Precision, ROC-AUC và PR-AUC cao nhất.
  Random Forest được lựa chọn làm mô hình chính của hệ thống.

### 7. Model Metadata

Metadata được lưu tại:
ai-models/models/metadata.json

Phiên bản hiện tại:
1.0.0

Final model:
Random Forest

Artifact:
rf_model.joblib

### 8. Kiến trúc hệ thống

Hệ thống gồm 4 service chính.
Frontend
Công nghệ:
HTML
CSS
JavaScript
Bootstrap
Nginx

Chức năng:

- Nhập 9 chỉ số chất lượng nước
- Chọn một trong 4 mô hình
- Gửi yêu cầu dự đoán
- Hiển thị kết quả phân loại
- Hiển thị probability
- Hiển thị Model Info
- Hiển thị lịch sử dự đoán
- Hiển thị bảng so sánh 4 mô hình
  Local URL:
  http://localhost:8080

Backend
Framework:
FastAPI

Local URL:
http://localhost:8001

Chức năng:

- Nhận request từ Frontend
- Gán và truyền request_id
- Gọi AI Service
- Trả kết quả dự đoán
- Lưu lịch sử dự đoán vào MongoDB
- Health Check
- Model Info Proxy
- Error Handling
- CORS
- Logging
  AI Service
  Framework:
  FastAPI

Địa chỉ nội bộ Docker:
http://ai-service:8000

Chức năng:

- Load 4 model khi service khởi động
- Validate dữ liệu đầu vào
- Prediction
- Probability
- Model Info
- Health Check
- Request ID
- Logging
- Error Handling
  MongoDB
  MongoDB được sử dụng để lưu lịch sử dự đoán.
  Local port:
  27017

Một record lịch sử gồm:
request_id
features
prediction
probability
model_version
created_at

### 9. API Backend

Health Check
GET /health

Ví dụ:
http://localhost:8001/health

Prediction
POST /api/predict

### Ví dụ request:

{
"ph": 7.0,
"Hardness": 204.89,
"Solids": 20791.32,
"Chloramines": 7.30,
"Sulfate": 368.51,
"Conductivity": 564.30,
"Organic_carbon": 10.37,
"Trihalomethanes": 86.99,
"Turbidity": 2.96,
"model_type": "rf"
}

### Ví dụ response:

{
"prediction": 0,
"probability": 0.8164,
"model_type": "rf",
"model_used": "Random Forest",
"model_version": "1.0.0",
"request_id": "example-request-id"
}

History
GET /api/history?limit=10

Model Info
GET /api/model-info

### 10. AI Service API

Các endpoint chính:
POST /predict
POST /validate
GET /health
GET /model-info

AI Service hỗ trợ:
lr
svm
knn
rf

### 11. Docker

Các service trong docker-compose.yml:
frontend
backend
ai-service
mongodb

Khởi động toàn bộ hệ thống:
docker compose up -d --build

Kiểm tra trạng thái:
docker compose ps

Xem AI log:
docker compose logs ai-service

Xem Backend log:
docker compose logs backend

Tắt hệ thống:
docker compose down

### 12. Chạy Frontend

Sau khi Docker khởi động:
http://localhost:8080

### 13. Automated Test

AI Service
Di chuyển vào:
cd ai-models\service

Chạy:
python -m pytest tests -v

Kết quả hiện tại:
8 passed

Backend
Di chuyển vào:
cd app\backend

Chạy:
python -m pytest tests -v

Kết quả hiện tại:
8 passed

### 14. Request ID Logging

Hệ thống hỗ trợ truyền cùng một request_id xuyên suốt:
Frontend
|
v
Backend
|
v
AI Service

Request ID đã kiểm thử:
e2e-log-test-001

Backend log:
request_id=e2e-log-test-001
method=POST
path=/api/predict
status=200
latency_ms=116.81

AI Service log:
request_id=e2e-log-test-001
method=POST
path=/predict
status=200
latency_ms=92.40

Kết quả: PASS

### 15. Load Test

File load test:
load_test.py

Cấu hình:
Concurrent users : 10
Duration : 60 giây

| Chỉ số              |     Kết quả |
| ------------------- | ----------: |
| Concurrent users    |          10 |
| Actual duration     |  60.46 giây |
| Total requests      |         776 |
| Successful requests |         776 |
| Failed requests     |           0 |
| Requests/second     | 12.84 req/s |
| P50 latency         |   737.95 ms |
| P95 latency         |  1128.86 ms |
| Error rate          |       0.00% |

Chạy: python load_test.py

Chi tiết kết quả: docs/test_results.md

### 16. Public Deploy / Smoke Test

Hệ thống đã được public bằng ngrok qua Frontend port:
8080

Lệnh:
ngrok http 8080

Public URL tại thời điểm kiểm thử:
https://reveal-copied-cornfield.ngrok-free.dev

URL ngrok có thể thay đổi sau khi tunnel được khởi động lại.

Smoke test đã xác nhận:
Public Frontend : PASS
Public Predict : PASS
Backend API : PASS
AI Service : PASS
MongoDB History : PASS

### 17. Cấu trúc thư mục chính

WaterQualityProject/
│
├── ai-models/
│ ├── colab/
│ │ ├── 01_eda.ipynb
│ │ ├── 02_preprocess.ipynb
│ │ ├── 03_train.ipynb
│ │ └── 04_evaluate.ipynb
│ │
│ ├── data/
│ │ ├── dataset.zip
│ │ └── DATA.md
│ │
│ ├── models/
│ │ ├── logistic_regression.joblib
│ │ ├── svm.joblib
│ │ ├── knn_model.joblib
│ │ ├── rf_model.joblib
│ │ ├── metadata.json
│ │ └── schema.json
│ │
│ ├── service/
│ │ ├── main.py
│ │ ├── Dockerfile
│ │ └── tests/
│ │
│ ├── src/
│ └── requirements.txt
│
├── app/
│ ├── backend/
│ │ ├── src/
│ │ ├── tests/
│ │ └── Dockerfile
│ │
│ └── frontend/
│ ├── index.html
│ └── Dockerfile
│
├── docs/
│ ├── figures/
│ ├── baocao.docx
│ └── test_results.md
│
├── docker-compose.yml
├── load_test.py
├── .env.example
├── .gitignore
└── README.md

### 18. Git Workflow

Các branch chính của nhóm:
main
develop
feature/dat
feature/HoangYen

Hai thành viên phát triển trên branch riêng, review phần của nhau và hợp nhất vào develop sau khi kiểm thử.

### 19. Kết quả kiểm thử tổng hợp

| Hạng mục                      | Kết quả    |
| ----------------------------- | ---------- |
| AI Service automated tests    | PASS - 8/8 |
| Backend automated tests       | PASS - 8/8 |
| Request ID Backend -> AI      | PASS       |
| Docker End-to-End             | PASS       |
| MongoDB history               | PASS       |
| Model version tracking        | PASS       |
| Load test 10 concurrent users | PASS       |
| Public Frontend               | PASS       |
| Public prediction             | PASS       |

### 20. Lưu ý

Dự án được xây dựng cho mục đích học tập môn Học máy cơ bản.
Kết quả dự đoán của hệ thống không được sử dụng thay thế cho các phương pháp kiểm nghiệm chất lượng nước trong thực tế.

```

```
