import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def train_and_save_models(X_train, y_train, models_dir="../models/"):
    """Huấn luyện cả KNN & Random Forest, lưu thành các file joblib tương ứng."""
    
    # 1. Pipeline KNN
    knn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid_knn = {'knn__n_neighbors': [3, 5, 7, 9, 11], 'knn__weights': ['uniform', 'distance']}
    grid_knn = GridSearchCV(knn_pipeline, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    best_knn = grid_knn.best_estimator_
    joblib.dump(best_knn, f"{models_dir}knn_model.joblib")

    # 2. Pipeline Random Forest
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    param_grid_rf = {'rf__n_estimators': [50, 100, 200], 'rf__max_depth': [None, 10, 20]}
    grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    joblib.dump(best_rf, f"{models_dir}rf_model.joblib")

    # Lưu mô hình mặc định (Random Forest) vào model.joblib
    joblib.dump(best_rf, f"{models_dir}model.joblib")

    return {"knn": best_knn, "rf": best_rf}