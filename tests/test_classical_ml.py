"""
Unit tests for Classical ML module.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Classical_ML.train import train_svm, train_random_forest, train_knn
from Classical_ML.evaluate import evaluate_model


def _make_data(n_train=200, n_test=50, n_features=14):
    rng = np.random.default_rng(7)
    X_tr = rng.standard_normal((n_train, n_features))
    y_tr = rng.integers(2, 5, n_train)
    X_te = rng.standard_normal((n_test, n_features))
    y_te = rng.integers(2, 5, n_test)
    return X_tr, X_te, y_tr, y_te


class TestClassicalML:
    def test_svm_trains_and_predicts(self):
        X_tr, X_te, y_tr, y_te = _make_data()
        model = train_svm(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)
        assert set(preds).issubset({2, 3, 4})

    def test_random_forest_trains_and_predicts(self):
        X_tr, X_te, y_tr, y_te = _make_data()
        model = train_random_forest(X_tr, y_tr, n_estimators=10)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)

    def test_knn_trains_and_predicts(self):
        X_tr, X_te, y_tr, y_te = _make_data()
        model = train_knn(X_tr, y_tr, n_neighbors=3)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)

    def test_evaluate_returns_accuracy(self):
        X_tr, X_te, y_tr, y_te = _make_data()
        model = train_svm(X_tr, y_tr)
        results = evaluate_model(model, X_te, y_te, model_name="test_svm")
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0
