from pathlib import Path

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


# =====================================================
# LOAD TEST SET
# =====================================================

def load_test_data():
    df = load_dataset(DATA_PATH)

    X, y = split_features_target(df)

    _, X_test, _, y_test = (
        split_train_test(X, y)
    )

    return X_test, y_test


# =====================================================
# HÀM ĐÁNH GIÁ MODEL
# =====================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):
    y_pred = model.predict(
        X_test
    )

    if hasattr(
        model,
        "predict_proba"
    ):
        y_score = (
            model
            .predict_proba(X_test)[:, 1]
        )

    elif hasattr(
        model,
        "decision_function"
    ):
        y_score = (
            model
            .decision_function(X_test)
        )

    else:
        y_score = None


    result = {
        "Model": model_name,

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "ROC_AUC":
            (
                roc_auc_score(
                    y_test,
                    y_score
                )
                if y_score is not None
                else np.nan
            ),

        "PR_AUC":
            (
                average_precision_score(
                    y_test,
                    y_score
                )
                if y_score is not None
                else np.nan
            ),
    }


    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    print(
        f"\n{'=' * 60}"
    )

    print(
        f"MODEL: {model_name}"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        pd.DataFrame(
            [result]
        ).round(4)
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


    result.update({
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

    return result


# =====================================================
# ĐÁNH GIÁ 2 CANDIDATE MODELS
# =====================================================

def evaluate_models():
    X_test, y_test = (
        load_test_data()
    )


    print(
        "Test samples:",
        X_test.shape
    )


    logistic_path = (
        MODELS_DIR
        / "logistic_regression.joblib"
    )

    svm_path = (
        MODELS_DIR
        / "svm.joblib"
    )


    if not logistic_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {logistic_path}"
        )

    if not svm_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {svm_path}"
        )


    logistic_model = joblib.load(
        logistic_path
    )

    svm_model = joblib.load(
        svm_path
    )


    logistic_result = evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    svm_result = evaluate_model(
        svm_model,
        X_test,
        y_test,
        "SVM"
    )


    comparison = pd.DataFrame([
        logistic_result,
        svm_result,
    ])


    print(
        f"\n{'=' * 60}"
    )

    print(
        "LOGISTIC REGRESSION vs SVM"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        comparison[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
                "PR_AUC",
            ]
        ].round(4)
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    evaluate_models()