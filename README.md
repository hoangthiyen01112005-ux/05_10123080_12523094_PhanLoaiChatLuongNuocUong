# 🌊 Water Potability Prediction System

Hệ thống dự đoán khả năng uống được của nước (Water Potability Prediction) dựa trên 9 chỉ số hóa lý. Dự án được triển khai theo kiến trúc Microservices với 3 lớp độc lập: Frontend, Backend API và AI Service, được đóng gói hoàn chỉnh bằng Docker.

---

## 🛠️ Kiến trúc Hệ thống

Hệ thống bao gồm 3 dịch vụ chính hoạt động cùng nhau:

1. **Frontend (Web App):** Giao diện người dùng cho phép nhập 9 thông số hóa lý của nước, thực hiện kiểm tra dữ liệu đầu vào (client-side validation) và hiển thị kết quả dự đoán.
2. **Backend (API Service):** Tiếp nhận yêu cầu từ Frontend, kiểm tra tính hợp lệ của dữ liệu theo `schema.json`, gán mã theo dõi `request_id` cho từng yêu cầu và chuyển tiếp tới AI Service.
3. **AI Service:** Nạp mô hình máy học (`model.joblib`) lúc khởi chạy, thực hiện tiền xử lý dữ liệu và đưa ra dự đoán nhãn chất lượng nước.

---

## 📋 9 Chỉ số Đầu vào (Features)

* **pH:** Độ pH của nước ($0 - 14$).
* **Hardness:** Độ cứng của nước (mg/L).
* **Solids:** Tổng chất rắn hòa tan - TDS (ppm).
* **Chloramines:** Lượng Chloramines (ppm).
* **Sulfate:** Lượng Sulfate hòa tan (mg/L).
* **Conductivity:** Độ dẫn điện của nước ($\mu S/cm$).
* **Organic Carbon:** Hàm lượng carbon hữu cơ (ppm).
* **Trihalomethanes:** Lượng Trihalomethanes ($\mu g/L$).
* **Turbidity:** Độ đục của nước (NTU).

---

## 🚀 Hướng dẫn Khởi chạy (với Docker Compose)

### **Yêu cầu môi trường**
* [Docker](https://www.docker.com/) và [Docker Compose](https://docs.docker.com/compose/) đã được cài đặt trên máy.

### **Các bước thực hiện**

1. **Clone repository về máy:**
   ```bash
   git clone <URL_REPO_CUA_BAN>
   cd <TEN_THU_MUC_REPO>