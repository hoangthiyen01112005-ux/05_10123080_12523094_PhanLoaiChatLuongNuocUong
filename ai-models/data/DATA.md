# Thông tin Tập dữ liệu Chất lượng Nước (Water Potability)

## 1. Nguồn dữ liệu & Lưu trữ
- File lưu trữ: `ai-models/data/dataset.zip`
- Tập dữ liệu gốc: `water_potability.csv` (được nén trong zip)

## 2. Thông số tổng quan
- **Số lượng mẫu (Rows):** 3276
- **Tổng số cột (Columns):** 10 (9 Features + 1 Target)
- **Số dòng trùng lặp:** 0

## 3. Cột mục tiêu (Target)
- **Tên cột:** `Potability`
- **Kiểu dữ liệu:** `int64`
- **Ý nghĩa:** `0` (Không thể uống), `1` (An toàn để uống)

## 4. Danh sách 9 Đặc trưng (Features)
1. `ph`: Độ pH của nước
2. `Hardness`: Độ cứng của nước
3. `Solids`: Tổng chất rắn hòa tan (TDS)
4. `Chloramines`: Nồng độ Chloramines
5. `Sulfate`: Nồng độ Sulfate
6. `Conductivity`: Độ dẫn điện
7. `Organic_carbon`: Hàm lượng carbon hữu cơ
8. `Trihalomethanes`: Nồng độ Trihalomethanes
9. `Turbidity`: Độ đục của nước


## 5. Chiến lược Tiền xử lý Dữ liệu (Data Preprocessing Strategy)

### 5.1. Xử lý giá trị thiếu (Missing Values)
- **Phương pháp:** Sử dụng `SimpleImputer(strategy='median')`.
- **Lý do chọn Median:** Dữ liệu các chỉ số hóa học của nước có chứa các giá trị ngoại lệ (outliers), dùng Trung vị (Median) giúp tránh bị lệch giá trị so với Trung bình (Mean).
- **Cột bị thiếu chính:** `ph`, `Sulfate`, `Trihalomethanes`.

### 5.2. Chuẩn hóa đặc trưng (Feature Scaling)
- **Phương pháp:** Sử dụng `StandardScaler()` (Z-score Normalization).
- **Lý do:** Đưa tất cả 9 đặc trưng về cùng quy mô (Mean = 0, Variance = 1), giúp các thuật toán phân loại (KNN, SVM, Logistic Regression, v.v.) hội tụ nhanh và không bị phân biệt đối xử do chênh lệch đơn vị đo.

### 5.3. Phân chia tập dữ liệu (Data Splitting)
- **Tỷ lệ phân chia:** 80% Train / 20% Test.
- **Phương pháp:** `stratify=y` để đảm bảo tỷ lệ lớp `Potability` (0 và 1) đồng nhất giữa 2 tập Train và Test.
- **Kích thước thực tế:**
  - Tập Train: 2,620 mẫu.
  - Tập Test: 656 mẫu.

## 6. Kết quả Huấn luyện & Lựa chọn Mô hình (Model Training)

### 6.1. Các Mô hình Đã Huấn Luyện (Thành viên 1)
- **K-Nearest Neighbors (KNN):** Tinh chỉnh tham số `n_neighbors` và `weights`.
- **Random Forest Classifier:** Tinh chỉnh tham số `n_estimators` và `max_depth`.

### 6.2. Kết quả Đánh giá Trên Tập Test
- **Mô hình được lưu chính thức:** Lưu tại `ai-models/models/model.joblib`.
- **File Schema:** `ai-models/models/schema.json` định nghĩa chuẩn cấu trúc 9 đầu vào.
- **File Metadata:** `ai-models/models/metadata.json` chứa các chỉ số Accuracy, Precision, Recall, F1-score của mô hình tốt nhất.