import joblib
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def train_and_save_model(X_train, y_train, save_path="../models/model.joblib"):
    """Huấn luyện Random Forest và lưu file artifact model.joblib."""
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {'rf__n_estimators': [50, 100], 'rf__max_depth': [None, 10]}
    grid = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    joblib.dump(best_model, save_path)
    return best_model