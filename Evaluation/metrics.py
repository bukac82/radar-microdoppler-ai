"""
Core Evaluation Metrics
========================
Accuracy, macro-F1, ROC-AUC, and confusion matrix utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Optional


LABEL_NAMES = {2: "2-blade (UH-1)", 3: "3-blade (Gazelle)", 4: "4-blade (Apache/UH-60)"}
CLASS_NAMES  = ["2-blade", "3-blade", "4-blade"]
LABEL_ORDER  = [2, 3, 4]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth integer labels (N,).
        y_pred: Predicted integer labels (N,).
        y_proba: Predicted class probabilities (N, C). Required for ROC-AUC.

    Returns:
        Dict with keys: accuracy, macro_f1, roc_auc (if proba given), confusion_matrix.
    """
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0),
    }

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            results["roc_auc"] = float(auc)
        except Exception as e:
            results["roc_auc"] = None
            results["roc_auc_error"] = str(e)

    return results


def print_metrics(metrics: dict, model_name: str = "model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*55}")
    print(f" Evaluation: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Macro F1    : {metrics['macro_f1']:.4f}")
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"\n{metrics['report']}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.integers(2, 5, 200)
    y_pred = rng.integers(2, 5, 200)
    m = compute_metrics(y_true, y_pred)
    print_metrics(m, "Random Baseline")
