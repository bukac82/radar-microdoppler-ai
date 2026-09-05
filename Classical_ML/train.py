"""
Classical ML Training
=====================
Train SVM, Random Forest, XGBoost, and k-NN classifiers on extracted features.
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_svm(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVC:
    """Train a Support Vector Machine classifier.

    Args:
        X_train: Feature matrix (n_samples, n_features).
        y_train: Integer labels.
        **kwargs: Passed to SVC constructor.

    Returns:
        Fitted SVC model.
    """
    params = dict(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    params.update(kwargs)
    model = SVC(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "svm.joblib")
    print(f"SVM trained. Saved to {MODELS_DIR / 'svm.joblib'}")
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    params = dict(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "random_forest.joblib")
    print(f"Random Forest trained. Saved to {MODELS_DIR / 'random_forest.joblib'}")
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, **kwargs):
    """Train an XGBoost classifier. Requires xgboost package."""
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    # XGBoost expects labels 0-indexed
    label_map = {v: i for i, v in enumerate(sorted(set(y_train)))}
    y_mapped = np.array([label_map[y] for y in y_train])
    params = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                  use_label_encoder=False, eval_metric="mlogloss",
                  random_state=42, n_jobs=-1)
    params.update(kwargs)
    model = XGBClassifier(**params)
    model.fit(X_train, y_mapped)
    model._label_map = label_map
    joblib.dump(model, MODELS_DIR / "xgboost.joblib")
    print(f"XGBoost trained. Saved to {MODELS_DIR / 'xgboost.joblib'}")
    return model


def train_knn(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> KNeighborsClassifier:
    """Train a k-Nearest Neighbours classifier."""
    params = dict(n_neighbors=7, metric="euclidean", n_jobs=-1)
    params.update(kwargs)
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "knn.joblib")
    print(f"k-NN trained. Saved to {MODELS_DIR / 'knn.joblib'}")
    return model


def load_model(name: str):
    """Load a previously saved model by name (e.g. 'svm', 'random_forest')."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Train first.")
    return joblib.load(path)


if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 14))
    y = rng.integers(2, 5, size=300)  # labels 2, 3, or 4

    train_svm(X, y)
    train_random_forest(X, y)
    train_knn(X, y)
    print("All models trained successfully.")
