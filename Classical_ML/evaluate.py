"""
Classical ML Evaluation
========================
Evaluate trained classical ML classifiers.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["2-blade", "3-blade", "4-blade"]
LABEL_ORDER  = [2, 3, 4]


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "model"):
    """Evaluate a fitted sklearn classifier.

    Args:
        model: Fitted sklearn model with .predict() method.
        X_test: Test feature matrix.
        y_test: True labels.
        model_name: Used for file naming when saving outputs.

    Returns:
        dict with 'accuracy', 'report', 'confusion_matrix'.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix (acc={acc:.3f})")
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


if __name__ == "__main__":
    from Classical_ML.train import train_svm
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((300, 14))
    y_tr = rng.integers(2, 5, size=300)
    X_te = rng.standard_normal((100, 14))
    y_te = rng.integers(2, 5, size=100)

    model = train_svm(X_tr, y_tr)
    evaluate_model(model, X_te, y_te, model_name="svm_smoke_test")
