"""
Hyperparameter Search for Classical ML Models
==============================================
Uses sklearn GridSearchCV / RandomizedSearchCV to tune classifiers.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def tune_svm(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, n_jobs: int = -1):
    """Grid search over SVM hyperparameters."""
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01],
        "svm__kernel": ["rbf", "poly"],
    }
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True))])
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=n_jobs, verbose=1)
    gs.fit(X_train, y_train)
    print(f"Best SVM params : {gs.best_params_}")
    print(f"Best CV accuracy: {gs.best_score_:.4f}")
    joblib.dump(gs.best_estimator_, MODELS_DIR / "svm_tuned.joblib")
    return gs


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, n_iter: int = 20, n_jobs: int = -1):
    """Randomized search over Random Forest hyperparameters."""
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    rs = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, cv=cv,
                            scoring="accuracy", n_jobs=n_jobs, verbose=1, random_state=42)
    rs.fit(X_train, y_train)
    print(f"Best RF params  : {rs.best_params_}")
    print(f"Best CV accuracy: {rs.best_score_:.4f}")
    joblib.dump(rs.best_estimator_, MODELS_DIR / "random_forest_tuned.joblib")
    return rs


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 14))
    y = rng.integers(2, 5, size=500)
    tune_svm(X, y, cv=3)
    tune_random_forest(X, y, cv=3, n_iter=5)
