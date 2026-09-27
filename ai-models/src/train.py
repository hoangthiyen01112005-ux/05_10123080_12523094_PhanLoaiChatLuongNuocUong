from pathlib import Path
import zipfile

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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

DATA_DIR = (
    PROJECT_ROOT
    / "ai-models"
    / "data"
)

CSV_PATH = (
    DATA_DIR
    / "water_potability.csv"
)

ZIP_PATH = (
    DATA_DIR
    / "dataset.zip"
)

TEMP_DATA_DIR = (
    DATA_DIR
    / "_tmp"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "ai-models"
    / "models"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================
# TÌM / GIẢI NÉN DATASET
# =====================================================

def resolve_dataset_path():
    """
    Tìm water_potability.csv.

    Nếu file CSV chưa tồn tại thì giải nén dataset.zip
    vào thư mục data/_tmp.
    """

    if CSV_PATH.exists():
        return CSV_PATH

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            "Không tìm thấy water_potability.csv "
            "hoặc dataset.zip"
        )

    TEMP_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zip_file:
        zip_file.extractall(
            TEMP_DATA_DIR
        )

    csv_files = list(
        TEMP_DATA_DIR.rglob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "Không tìm thấy file CSV trong dataset.zip"
        )

    return csv_files[0]


# =====================================================
# LOAD VÀ CHIA DỮ LIỆU
# =====================================================

def prepare_data():
    data_path = resolve_dataset_path()

    df = load_dataset(
        data_path
    )

    X, y = split_features_target(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_train_test(
        X,
        y,
    )

    print(
        "Dataset:",
        df.shape,
    )

    print(
        "Train:",
        X_train.shape,
    )

    print(
        "Test:",
        X_test.shape,
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

def train_logistic_regression(
    X_train,
    y_train,
):
    model = Pipeline([
        (
            "preprocessor",
            build_scaled_preprocessor(),
        ),
        (
            "classifier",
            LogisticRegression(
                C=0.01,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
                max_iter=2000,
            ),
        ),
    ])

    model.fit(
        X_train,
        y_train,
    )

    model_path = (
        MODELS_DIR
        / "logistic_regression.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        "Đã lưu Logistic Regression:",
        model_path,
    )

    return model


# =====================================================
# SUPPORT VECTOR MACHINE
# =====================================================

def train_svm(
    X_train,
    y_train,
):
    model = Pipeline([
        (
            "preprocessor",
            build_scaled_preprocessor(),
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    model.fit(
        X_train,
        y_train,
    )

    model_path = (
        MODELS_DIR
        / "svm.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        "Đã lưu SVM:",
        model_path,
    )

    return model


# =====================================================
# K-NEAREST NEIGHBORS
# =====================================================

def train_knn(
    X_train,
    y_train,
):
    pipeline = Pipeline([
        (
            "imputer",
            build_scaled_preprocessor()
            .named_steps["imputer"],
        ),
        (
            "scaler",
            build_scaled_preprocessor()
            .named_steps["scaler"],
        ),
        (
            "knn",
            KNeighborsClassifier(),
        ),
    ])

    param_grid = {
        "knn__n_neighbors": [
            3,
            5,
            7,
            9,
            11,
        ],
        "knn__weights": [
            "uniform",
            "distance",
        ],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    best_model = (
        grid_search
        .best_estimator_
    )

    model_path = (
        MODELS_DIR
        / "knn_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    print(
        "KNN best params:",
        grid_search.best_params_,
    )

    print(
        "Đã lưu KNN:",
        model_path,
    )

    return best_model


# =====================================================
# RANDOM FOREST
# =====================================================

def train_random_forest(
    X_train,
    y_train,
):
    pipeline = Pipeline([
        (
            "imputer",
            build_scaled_preprocessor()
            .named_steps["imputer"],
        ),
        (
            "scaler",
            build_scaled_preprocessor()
            .named_steps["scaler"],
        ),
        (
            "rf",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    param_grid = {
        "rf__n_estimators": [
            50,
            100,
            200,
        ],
        "rf__max_depth": [
            None,
            10,
            20,
        ],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    best_model = (
        grid_search
        .best_estimator_
    )

    model_path = (
        MODELS_DIR
        / "rf_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    print(
        "Random Forest best params:",
        grid_search.best_params_,
    )

    print(
        "Đã lưu Random Forest:",
        model_path,
    )

    return best_model


# =====================================================
# TRAIN 4 MODELS
# =====================================================

def train_models():
    (
        X_train,
        _,
        y_train,
        _,
    ) = prepare_data()

    print(
        "\n===== 1. LOGISTIC REGRESSION ====="
    )

    train_logistic_regression(
        X_train,
        y_train,
    )

    print(
        "\n===== 2. SVM ====="
    )

    train_svm(
        X_train,
        y_train,
    )

    print(
        "\n===== 3. KNN ====="
    )

    train_knn(
        X_train,
        y_train,
    )

    print(
        "\n===== 4. RANDOM FOREST ====="
    )

    train_random_forest(
        X_train,
        y_train,
    )

    print(
        "\nHoàn thành huấn luyện 4 model."
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    train_models()