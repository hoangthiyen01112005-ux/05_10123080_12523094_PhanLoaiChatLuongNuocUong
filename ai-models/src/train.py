from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocess import (
    load_dataset,
    split_features_target,
    split_train_test,
    build_scaled_preprocessor,
    RANDOM_STATE,
)


# =====================================================
# ĐƯỜNG DẪN PROJECT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "ai-models"
    / "data"
    / "water_potability.csv"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "ai-models"
    / "models"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# LOAD VÀ CHIA DỮ LIỆU
# =====================================================

def prepare_data():
    df = load_dataset(DATA_PATH)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = (
        split_train_test(X, y)
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =====================================================
# LOGISTIC REGRESSION
# =====================================================

def build_logistic_model():
    model = Pipeline([
        (
            "preprocessor",
            build_scaled_preprocessor()
        ),
        (
            "classifier",
            LogisticRegression(
                C=0.01,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
                max_iter=2000,
            )
        ),
    ])

    return model


# =====================================================
# SUPPORT VECTOR MACHINE
# =====================================================

def build_svm_model():
    model = Pipeline([
        (
            "preprocessor",
            build_scaled_preprocessor()
        ),
        (
            "classifier",
            SVC(
                C=1.0,
                class_weight="balanced",
                gamma="scale",
                kernel="rbf",
                random_state=RANDOM_STATE,
            )
        ),
    ])

    return model


# =====================================================
# TRAIN VÀ LƯU MODEL
# =====================================================

def train_models():
    (
        X_train,
        _,
        y_train,
        _,
    ) = prepare_data()

    print(
        "Training samples:",
        X_train.shape
    )

    # Logistic Regression
    logistic_model = build_logistic_model()

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_path = (
        MODELS_DIR
        / "logistic_regression.joblib"
    )

    joblib.dump(
        logistic_model,
        logistic_path
    )

    print(
        "Đã lưu Logistic Regression:"
    )
    print(logistic_path)

    # SVM
    svm_model = build_svm_model()

    svm_model.fit(
        X_train,
        y_train
    )

    svm_path = (
        MODELS_DIR
        / "svm.joblib"
    )

    joblib.dump(
        svm_model,
        svm_path
    )

    print(
        "Đã lưu SVM:"
    )
    print(svm_path)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    train_models()