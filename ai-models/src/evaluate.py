from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from preprocess import (
    load_dataset,
    split_features_target,
    split_train_test,
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


# =====================================================
# DANH SÁCH 4 MODEL
# =====================================================

MODEL_PATHS = {
    "lr": (
        MODELS_DIR
        / "logistic_regression.joblib"
    ),

    "svm": (
        MODELS_DIR
        / "svm.joblib"
    ),

    "knn": (
        MODELS_DIR
        / "knn_model.joblib"
    ),

    "rf": (
        MODELS_DIR
        / "rf_model.joblib"
    ),
}


MODEL_NAMES = {
    "lr": "Logistic Regression",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "rf": "Random Forest",
}


# =====================================================
# TÌM / GIẢI NÉN DATASET
# =====================================================

def resolve_dataset_path():
    """
    Tìm file water_potability.csv.

    Nếu chưa có CSV thì giải nén dataset.zip
    vào data/_tmp.
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
# LOAD TEST SET
# =====================================================

def load_test_data():
    """
    Load dataset và tạo đúng Test Set:
    test_size = 0.20
    random_state = 42
    stratify = y
    """

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

    print(
        "Test target:",
        y_test
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    return (
        X_test,
        y_test,
    )


# =====================================================
# LẤY SCORE / PROBABILITY
# =====================================================

def get_prediction_score(
    model,
    X_test,
):
    """
    Lấy score cho ROC-AUC và PR-AUC.

    Ưu tiên predict_proba().
    Nếu model không có predict_proba()
    thì dùng decision_function().
    """

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        return probabilities[:, 1]

    if hasattr(
        model,
        "decision_function",
    ):
        return model.decision_function(
            X_test
        )

    return None


# =====================================================
# ĐÁNH GIÁ 1 MODEL
# =====================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
):
    y_pred = model.predict(
        X_test
    )

    y_score = get_prediction_score(
        model,
        X_test,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = (
        roc_auc_score(
            y_test,
            y_score,
        )
        if y_score is not None
        else np.nan
    )

    pr_auc = (
        average_precision_score(
            y_test,
            y_score,
        )
        if y_score is not None
        else np.nan
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    result = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"MODEL: {model_name}"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Potable (0)",
                "Potable (1)",
            ],
            zero_division=0,
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        f"TN = {tn}"
    )

    print(
        f"FP = {fp}"
    )

    print(
        f"FN = {fn}"
    )

    print(
        f"TP = {tp}"
    )

    return result


# =====================================================
# LOAD 4 MODEL
# =====================================================

def load_models():
    models = {}

    for (
        model_key,
        model_path,
    ) in MODEL_PATHS.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: "
                f"{model_path}"
            )

        models[model_key] = (
            joblib.load(
                model_path
            )
        )

        print(
            f"Đã load "
            f"{MODEL_NAMES[model_key]}: "
            f"{model_path.name}"
        )

    return models


# =====================================================
# ĐÁNH GIÁ 4 MODEL
# =====================================================

def evaluate_models():
    (
        X_test,
        y_test,
    ) = load_test_data()

    models = load_models()

    results = []

    for model_key in [
        "lr",
        "svm",
        "knn",
        "rf",
    ]:
        result = evaluate_model(
            model=models[model_key],
            X_test=X_test,
            y_test=y_test,
            model_name=MODEL_NAMES[
                model_key
            ],
        )

        results.append(
            result
        )

    comparison = pd.DataFrame(
        results
    )

    metric_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL COMPARISON - 4 MODELS"
    )

    print(
        "=" * 70
    )

    print(
        comparison[
            metric_columns
        ].round(4)
    )

    print(
        "\nConfusion Matrix Summary:"
    )

    print(
        comparison[
            [
                "Model",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
        ]
    )

    return comparison


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    evaluate_models()