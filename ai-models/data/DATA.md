# Thông tin Tập dữ liệu Chất lượng Nước (Water Potability)

## 1. Nguồn dữ liệu & Lưu trữ
- File lưu trữ: `ai-models/data/dataset.zip`
- Tập dữ liệu gốc: `water_potability.csv` (được nén trong zip)

## 2. Thông số tổng quan
- **Số lượng mẫu (Rows):** 3,276
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
- **Lý do chọn Median:** Dữ liệu chứa các giá trị ngoại lệ (outliers), dùng Trung vị (Median) giúp tránh bị lệch giá trị so với Trung bình (Mean).
- **Các cột bị thiếu:** `ph`, `Sulfate`, `Trihalomethanes`.

### 5.2. Chuẩn hóa đặc trưng (Feature Scaling)
- **Phương pháp:** Sử dụng `StandardScaler()` (Z-score Normalization).
- **Lý do:** Đưa tất cả 9 đặc trưng về cùng quy mô (Mean = 0, Variance = 1) bên trong Pipeline để chống rò rỉ dữ liệu (Data Leakage).

### 5.3. Phân chia tập dữ liệu (Data Splitting)
- **Tỷ lệ phân chia:** 80% Train / 20% Test (`random_state=42`).
- **Phương pháp:** `stratify=y` đảm bảo tỷ lệ lớp `Potability` (0 và 1) đồng nhất giữa 2 tập Train và Test.
- **Kích thước thực tế:**
  - Tập Train: 2,620 mẫu.
  - Tập Test: 656 mẫu.

## 6. Kết quả Huấn luyện & Đóng gói Mô hình (Model Training - Member 1)

### 6.1. Các Mô hình Đã Huấn Luyện
- **K-Nearest Neighbors (KNN):** Tinh chỉnh tham số `n_neighbors` và `weights` bằng `GridSearchCV`.
- **Random Forest Classifier:** Tinh chỉnh tham số `n_estimators` và `max_depth` bằng `GridSearchCV`.

### 6.2. Lưu trữ Artifacts (Models & Schema)
Để phục vụ tích hợp đa mô hình trên Backend/Frontend, hệ thống lưu trữ đầy đủ các artifacts sau tại thư mục `ai-models/models/`:
- `knn_model.joblib`: Pipeline mô hình KNN hoàn chỉnh.
- `rf_model.joblib`: Pipeline mô hình Random Forest hoàn chỉnh.
- `model.joblib`: Mô hình chiến thắng mặc định (Random Forest).
- `schema.json`: Định nghĩa cấu trúc chuẩn của 9 đặc trưng đầu vào.
- `metadata.json`: Lưu trữ thông số hiệu năng và thông tin kỹ thuật của mô hình chiến thắng.

## 7. Báo cáo Đánh giá & So sánh Mô hình (Detailed Model Evaluation)

### 7.1. Các Biểu đồ Đánh giá Chi tiết
Các hình ảnh trực quan hóa được lưu tại thư mục `docs/figures/`:
- **Ma trận nhầm lẫn gộp:** `docs/figures/confusion_matrices_both.png` (Ma trận Confusion Matrix nằm cạnh nhau của cả KNN và Random Forest).
- **Đường cong ROC So sánh:** `docs/figures/roc_comparison.png` (So sánh trực quan chỉ số AUC giữa KNN và Random Forest trên cùng 1 hệ trục).
- **Mức độ quan trọng thuộc tính:** `docs/figures/feature_importance_rf.png` (Top các thuộc tính hóa lý ảnh hưởng nhất của Random Forest: Sulfate, pH, Hardness).

### 7.2. Tóm tắt So sánh Kết quả Trên Tập Test
| Chỉ số (Metric) | KNN | Random Forest | Mô hình tối ưu hơn |
| :--- | :---: | :---: | :---: |
| **Accuracy** | ~0.61 | **~0.65** | Random Forest |
| **Precision (Lớp 1)** | ~0.52 | **~0.62** | Random Forest |
| **Recall (Lớp 1)** | ~0.25 | **~0.29** | Random Forest |
| **F1-Score (Lớp 1)** | ~0.34 | **~0.40** | Random Forest |
| **AUC (ROC)** | ~0.58 | **~0.67** | Random Forest |

## 8. Cấu trúc Mô-đun Code (Source Code Structure)
- `01_eda.ipynb`: Phân tích khám phá dữ liệu & xuất 3 biểu đồ EDA.
- `02_preprocess.ipynb`: Xây dựng Pipeline tiền xử lý & kiểm tra rò rỉ dữ liệu.
- `03_train.ipynb`: Huấn luyện, đo thời gian, tinh chỉnh GridSearchCV & xuất 3 file models.
- `04_evaluate.ipynb`: Tính toán metrics, xuất bảng so sánh và 3 file đồ thị đánh giá.