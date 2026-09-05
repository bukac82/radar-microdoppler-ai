"""
Benchmark Runner
================
Evaluates all trained models on the same held-out test set and
produces a comparison table.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

from Evaluation.metrics import compute_metrics, print_metrics

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_benchmark(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_results: bool = True,
) -> dict:
    """Run all models on the same test set.

    Args:
        models: Dict of {model_name: (predict_fn, predict_proba_fn | None)}.
                  predict_fn: callable(X) → y_pred (int labels)
                  predict_proba_fn: callable(X) → proba (N, C) or None
        X_test: Pre-processed test input (feature matrix or spectrogram, depending on models).
        y_test: True integer labels.
        save_results: If True, save JSON report to Evaluation/results/.

    Returns:
        Dict of {model_name: metrics_dict}.
    """
    all_results = {}

    for name, (predict_fn, proba_fn) in models.items():
        print(f"\nEvaluating: {name}")
        y_pred = predict_fn(X_test)
        y_proba = proba_fn(X_test) if proba_fn else None
        metrics = compute_metrics(y_test, y_pred, y_proba)
        print_metrics(metrics, model_name=name)
        all_results[name] = metrics

    # Print comparison table
    print(f"\n{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10} {'ROC-AUC':>10}")
    print("-" * 65)
    for name, m in all_results.items():
        auc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "  N/A  "
        print(f"{name:<30} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {auc:>10}")

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
        # Make serializable
        serializable = {}
        for name, m in all_results.items():
            serializable[name] = {k: v for k, v in m.items() if k != "report"}
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nBenchmark results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 14))
    y = rng.integers(2, 5, 100)

    dummy_models = {
        "Random Classifier": (
            lambda X: rng.integers(2, 5, len(X)),
            None,
        ),
        "Majority Class (4-blade)": (
            lambda X: np.full(len(X), 4),
            None,
        ),
    }

    run_benchmark(dummy_models, X, y)
