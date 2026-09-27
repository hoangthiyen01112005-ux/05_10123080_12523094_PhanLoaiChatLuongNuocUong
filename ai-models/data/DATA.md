# Thông tin Tập dữ liệu Chất lượng Nước (Water Potability)

## 1. Nguồn dữ liệu & Lưu trữ

- **Nguồn dữ liệu:** Kaggle - Water Potability Dataset
- **File lưu trữ:** `ai-models/data/dataset.zip`
- **Tập dữ liệu gốc:** `water_potability.csv` được nén trong `dataset.zip`

---

## 2. Thông số tổng quan

- **Số lượng mẫu (Rows):** 3276
- **Tổng số cột (Columns):** 10
- **Số Features:** 9
- **Số Target:** 1
- **Số dòng trùng lặp:** 0
- **Loại bài toán:** Binary Classification

Dataset được sử dụng để dự đoán một mẫu nước có thể uống được hay không dựa trên các chỉ số chất lượng nước.

---

## 3. Cột mục tiêu (Target)

- **Tên cột:** `Potability`
- **Kiểu dữ liệu:** `int64`
- **Ý nghĩa:**
  - `0`: Not Potable - Không thể uống
  - `1`: Potable - Có thể uống

### Phân bố lớp

| Potability |   Số mẫu |    Tỷ lệ |
| ---------: | -------: | -------: |
|          0 |     1998 |   60.99% |
|          1 |     1278 |   39.01% |
|   **Tổng** | **3276** | **100%** |

Dữ liệu có sự chênh lệch giữa hai lớp, trong đó lớp `0` có số lượng mẫu lớn hơn lớp `1`.

---

## 4. Danh sách 9 Đặc trưng (Features)

1. `ph`: Độ pH của nước
2. `Hardness`: Độ cứng của nước
3. `Solids`: Tổng chất rắn hòa tan
4. `Chloramines`: Nồng độ Chloramines
5. `Sulfate`: Nồng độ Sulfate
6. `Conductivity`: Độ dẫn điện
7. `Organic_carbon`: Hàm lượng Carbon hữu cơ
8. `Trihalomethanes`: Nồng độ Trihalomethanes
9. `Turbidity`: Độ đục của nước

---

## 5. Chiến lược Tiền xử lý Dữ liệu

### 5.1. Xử lý Missing Values

Missing Values xuất hiện tại 3 Features:

| Feature           | Missing Count | Missing Percent |
| ----------------- | ------------: | --------------: |
| `ph`              |           491 |          14.99% |
| `Sulfate`         |           781 |          23.84% |
| `Trihalomethanes` |           162 |           4.95% |

Target `Potability` không có Missing Value.

Phương pháp xử lý:

`SimpleImputer(strategy="median")`

Median được sử dụng vì ít bị ảnh hưởng bởi các giá trị ngoại lai hơn Mean.

Imputer chỉ được fit trên Training Set để tránh Data Leakage.

### 5.2. Kiểm tra Outlier

Outlier được kiểm tra bằng phương pháp IQR.

Kết quả kiểm tra:

| Feature           | Outlier Count | Outlier Percent |
| ----------------- | ------------: | --------------: |
| `Hardness`        |            83 |           2.53% |
| `Chloramines`     |            61 |           1.86% |
| `Solids`          |            47 |           1.43% |
| `ph`              |            46 |           1.65% |
| `Sulfate`         |            41 |           1.64% |
| `Trihalomethanes` |            33 |           1.06% |
| `Organic_carbon`  |            25 |           0.76% |
| `Turbidity`       |            19 |           0.58% |
| `Conductivity`    |            11 |           0.34% |

Các Outlier không bị loại bỏ trực tiếp khỏi Dataset.

### 5.3. Chuẩn hóa đặc trưng

Đối với Logistic Regression và SVM, dữ liệu được chuẩn hóa bằng:

`StandardScaler()`

Pipeline xử lý:

`Missing Values -> Median Imputation -> StandardScaler -> Model`

Việc chuẩn hóa giúp đưa các Features về cùng thang đo và hạn chế ảnh hưởng của sự khác biệt về đơn vị đo.

---

## 6. Phân chia Train/Test

Dataset được chia theo tỷ lệ:

- **Training Set:** 80%
- **Test Set:** 20%
- **Random State:** 42
- **Stratify:** `stratify=y`

Kích thước thực tế:

- `X_train`: 2620 mẫu
- `X_test`: 656 mẫu
- `y_train`: 2620 mẫu
- `y_test`: 656 mẫu

Phân bố lớp:

| Dataset | Potability = 0 | Potability = 1 |
| ------- | -------------: | -------------: |
| Train   |           1598 |           1022 |
| Test    |            400 |            256 |

Kiểm tra Train/Test overlap:

`Train/Test overlap: 0`

Không có mẫu nào xuất hiện đồng thời trong Training Set và Test Set.

---

## 7. Logistic Regression

Mô hình Logistic Regression đã được:

- Huấn luyện Baseline
- Đánh giá bằng 5-Fold Stratified Cross Validation
- Hyperparameter Tuning bằng GridSearchCV
- Đánh giá cuối cùng trên Test Set

Cấu hình sau Hyperparameter Tuning:

- `C = 0.01`
- `class_weight = balanced`
- `solver = liblinear`

Kết quả trên Test Set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.5305 |
| Precision | 0.4217 |
| Recall    | 0.5469 |
| F1-score  | 0.4762 |
| ROC-AUC   | 0.5488 |
| PR-AUC    | 0.4864 |

Confusion Matrix:

- `TN = 208`
- `FP = 192`
- `FN = 116`
- `TP = 140`

---

## 8. Support Vector Machine (SVM)

Mô hình SVM đã được:

- Huấn luyện Baseline
- Đánh giá bằng 5-Fold Stratified Cross Validation
- Hyperparameter Tuning bằng GridSearchCV
- Đánh giá cuối cùng trên Test Set

Cấu hình sau Hyperparameter Tuning:

- `C = 1.0`
- `class_weight = balanced`
- `gamma = scale`
- `kernel = rbf`

Kết quả trên Test Set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.6220 |
| Precision | 0.5159 |
| Recall    | 0.5078 |
| F1-score  | 0.5118 |
| ROC-AUC   | 0.6440 |
| PR-AUC    | 0.5662 |

Confusion Matrix:

- `TN = 278`
- `FP = 122`
- `FN = 126`
- `TP = 130`

---

## 9. Metrics Đánh giá

Các Metrics được sử dụng:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Hyperparameter Tuning chỉ được thực hiện trên Training Set.

Test Set không được sử dụng để Fit Imputer, Fit StandardScaler hoặc lựa chọn Hyperparameter.

Test Set chỉ được sử dụng cho bước Final Evaluation.

---

## 10. Cấu trúc File Liên quan

ai-models/

- data/
  - DATA.md
  - dataset.zip
  - water_potability.csv
- colab/
  - 01_eda.ipynb
  - 02_preprocess.ipynb
  - 03_train.ipynb
  - 04_evaluate.ipynb
- src/
  - preprocess.py

Trong đó:

- `01_eda.ipynb`: Phân tích và trực quan hóa dữ liệu
- `02_preprocess.ipynb`: Tiền xử lý dữ liệu
- `03_train.ipynb`: Huấn luyện và Hyperparameter Tuning Logistic Regression, SVM
- `04_evaluate.ipynb`: Final Evaluation trên Test Set
- `preprocess.py`: Chứa các hàm tiền xử lý dùng chung
- `dataset.zip`: Chứa file dữ liệu gốc `water_potability.csv`

---
