import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocessing_pipeline():
    """Tạo Pipeline tiền xử lý gồm Imputer và StandardScaler."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

def prepare_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Tách Features/Target và chia tập Train/Test giữ nguyên tỷ lệ lớp."""
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )