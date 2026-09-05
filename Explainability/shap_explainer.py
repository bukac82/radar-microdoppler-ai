"""
SHAP Explainability for Classical ML Models
============================================
Uses SHAP (SHapley Additive exPlanations) to explain predictions
of tree-based and SVM classifiers on hand-crafted features.

Install: pip install shap
"""

import numpy as np
from pathlib import Path

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARNING] shap not installed. Run: pip install shap")

from Preprocessing.features import FEATURE_NAMES

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def explain_with_shap(model, X_train: np.ndarray, X_test: np.ndarray, model_type: str = "tree"):
    """Compute SHAP values for a trained classifier.

    Args:
        model: Fitted sklearn model (RF, XGBoost, SVM, etc.).
        X_train: Training feature matrix (used to build the explainer background).
        X_test: Test samples to explain.
        model_type: 'tree' for tree-based models, 'kernel' for SVM/k-NN.

    Returns:
        shap.Explanation object with .values, .base_values, .data.
    """
    if not HAS_SHAP:
        raise ImportError("shap not installed. Run: pip install shap")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
    else:
        # KernelExplainer works for any model (slower)
        background = shap.kmeans(X_train, k=min(50, len(X_train)))
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x),
            background,
        )
        shap_values = explainer.shap_values(X_test[:50])  # limit for speed

    return shap_values


def plot_shap_summary(shap_values, X_test: np.ndarray, model_name: str = "model"):
    """Generate and save a SHAP summary beeswarm plot."""
    if not HAS_SHAP:
        raise ImportError("shap not installed.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_NAMES, show=False)
    out = PLOTS_DIR / f"{model_name}_shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {out}")


def plot_shap_bar(shap_values, model_name: str = "model"):
    """Generate and save a SHAP mean absolute bar chart."""
    if not HAS_SHAP:
        raise ImportError("shap not installed.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    shap.plots.bar(shap_values, show=False)
    out = PLOTS_DIR / f"{model_name}_shap_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar plot saved to {out}")


if __name__ == "__main__":
    if not HAS_SHAP:
        print("shap not installed — skipping.")
    else:
        from Classical_ML.train import train_random_forest
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 14))
        y = rng.integers(2, 5, 200)
        rf = train_random_forest(X, y)
        sv = explain_with_shap(rf, X, X[:20], model_type="tree")
        print(f"SHAP values shape: {sv.values.shape}")
