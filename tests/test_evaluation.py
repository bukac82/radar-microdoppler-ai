"""
Unit tests for Evaluation metrics module.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Evaluation.metrics import compute_metrics


class TestMetrics:
    def test_perfect_predictions(self):
        y = np.array([2, 2, 3, 3, 4, 4])
        m = compute_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_random_predictions_in_range(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(2, 5, 200)
        y_pred = rng.integers(2, 5, 200)
        m = compute_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0

    def test_with_proba(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(2, 5, 100)
        y_pred = rng.integers(2, 5, 100)
        proba = rng.dirichlet([1, 1, 1], size=100)
        m = compute_metrics(y_true, y_pred, y_proba=proba)
        assert "roc_auc" in m

    def test_confusion_matrix_shape(self):
        y = np.array([2, 3, 4, 2, 3, 4])
        m = compute_metrics(y, y)
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (3, 3)
