import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




TARGET_COLUMN = "Potability"

FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42




def load_dataset(data_path):
    """
    Đọc dataset Water Potability từ file CSV.
    """

    return pd.read_csv(data_path)



def split_features_target(df):
    """
    Tách 9 đặc trưng đầu vào và target Potability.
    """

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y




def split_train_test(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    """
    Chia dữ liệu 80% Train và 20% Test.

    stratify=y được sử dụng để giữ tỷ lệ lớp.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )



def build_scaled_preprocessor():
    """
    Pipeline tiền xử lý cho các mô hình cần chuẩn hóa.

    Median Imputation -> StandardScaler
    """

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])




def build_unscaled_preprocessor():
    """
    Pipeline chỉ xử lý missing values bằng median.
    """

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
    ])