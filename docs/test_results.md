# TEST RESULTS - WATER POTABILITY CLASSIFICATION

## 1. Thông tin môi trường kiểm thử

- Hệ thống: Water Potability Classification
- Môi trường: Docker Compose
- Frontend: http://localhost:8080
- Backend: http://localhost:8001
- AI Service: http://ai-service:8000
- Database: MongoDB
- Final model: Random Forest
- Model version: 1.0.0

---

## 2. Kiểm thử Request ID xuyên suốt hệ thống

### Mục tiêu

Kiểm tra một `request_id` có được truyền xuyên suốt từ:

Frontend/Nginx -> Backend -> AI Service

hay không.

### Request ID kiểm thử

```text
e2e-log-test-001
```

### Kết quả API

{
"prediction": 0,
"probability": 0.8164,
"model_type": "rf",
"model_used": "Random Forest",
"model_version": "1.0.0",
"request_id": "e2e-log-test-001"
}

### Backend log

request_id=e2e-log-test-001
method=POST
path=/api/predict
status=200
latency_ms=116.81

### AI Service log

request_id=e2e-log-test-001
method=POST
path=/predict
status=200
latency_ms=92.40

### Kết luận

Cùng một request_id là e2e-log-test-001 xuất hiện tại cả Backend và AI Service.
Điều này chứng minh request được truyền đúng qua luồng:
Frontend
|
v
Backend
|
v
AI Service
|
v
Machine Learning Model

Kết quả: PASS

### 3. Load Test API dự đoán

### Endpoint kiểm thử

POST http://localhost:8080/api/predict

### Cấu hình kiểm thử

- Concurrent users: 10
- Thời gian cấu hình: 60 giây
- Model sử dụng: Random Forest
- Công cụ: Python + httpx
- File kiểm thử: load_test.py

### Kết quả thực tế

| Chỉ số              |     Kết quả |
| ------------------- | ----------: |
| Concurrent users    |          10 |
| Actual duration     |  60.24 giây |
| Total requests      |         929 |
| Successful requests |         929 |
| Failed requests     |           0 |
| Requests/second     | 15.42 req/s |
| P50 latency         |   620.65 ms |
| P95 latency         |   861.78 ms |
| Error rate          |       0.00% |

### Nhận xét

Trong quá trình kiểm thử với 10 người dùng đồng thời:

- Toàn bộ 776 request đều xử lý thành công.
- Không có request thất bại.
- Error rate bằng 0.00%.
- Hệ thống xử lý trung bình 12.84 request/giây.
- 50% request có thời gian phản hồi nhỏ hơn hoặc bằng 737.95 ms.
- 95% request có thời gian phản hồi nhỏ hơn hoặc bằng 1128.86 ms.
  Kết quả: PASS

### 4. Public Deploy bằng ngrok

### Public URL tại thời điểm kiểm thử

https://reveal-copied-cornfield.ngrok-free.dev

### Public URL trên được tunnel tới:

http://localhost:8080

Lưu ý: URL ngrok có thể thay đổi khi tunnel được khởi động lại.

### Smoke Test

Các bước đã thực hiện:

1. Khởi động toàn bộ hệ thống bằng Docker Compose.
2. Public cổng Frontend 8080 bằng ngrok.
3. Mở public URL trên trình duyệt.
4. Giao diện Water Potability Classification hiển thị thành công.
5. Chọn mô hình Random Forest.
6. Gửi dữ liệu mẫu để dự đoán.
7. Hệ thống trả kết quả prediction thành công.
8. Lịch sử dự đoán được lưu thông qua Backend và MongoDB.

### Kết quả

Public Frontend: PASS
Public Predict: PASS
Backend API: PASS
AI Service: PASS
MongoDB History: PASS

### 5. Kiểm thử Docker End-to-End

### Luồng hệ thống đã được kiểm tra:

Public/Frontend
|
v
Nginx
|
v
Backend
|
v
AI Service
|
v
Random Forest
|
v
Backend
|
v
MongoDB

### Kết quả dự đoán mẫu:

{
"prediction": 0,
"probability": 0.8164,
"model_type": "rf",
"model_used": "Random Forest",
"model_version": "1.0.0"
}

### MongoDB lưu thành công:

prediction = 0
probability = 0.8164
model_version = 1.0.0

Kết quả: PASS

### 6. Automated Test

### AI Service

8 passed

Các nhóm chức năng đã kiểm thử:

- health
- request_id
- validation input hợp lệ
- validation ngoài phạm vi
- validation thiếu field
- model info
- system error handler
- prediction với 4 model

### Backend

8 passed

Các nhóm chức năng đã kiểm thử:

- health
- request_id
- history
- database error
- system error handler
- CORS
- predict và lưu history
- model info proxy

### 7. Tổng kết kiểm thử

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

Hệ thống đã thực hiện thành công các kiểm thử functional, integration, end-to-end, logging, load test và public smoke test.
