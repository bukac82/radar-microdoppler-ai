"""
OSR Evaluation Metrics
=======================
AUROC, FPR@95TPR, and OSCR metrics for Open-Set Recognition evaluation.
"""

import numpy as np


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC for OOD detection.

    Args:
        id_scores: OOD scores for in-distribution samples (lower = more ID).
        ood_scores: OOD scores for out-of-distribution samples (higher = more OOD).

    Returns:
        AUROC score in [0, 1].
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, scores))


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr_target: float = 0.95) -> float:
    """FPR at a given TPR level (e.g., FPR@95TPR).

    Args:
        id_scores: OOD scores for in-distribution samples.
        ood_scores: OOD scores for OOD samples.
        tpr_target: True positive rate target (default 0.95).

    Returns:
        False positive rate at the target TPR.
    """
    from sklearn.metrics import roc_curve
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Find threshold where TPR >= tpr_target
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def oscr(
    closed_set_probs: np.ndarray,
    closed_set_labels: np.ndarray,
    ood_scores_id: np.ndarray,
    ood_scores_ood: np.ndarray,
    n_thresholds: int = 100,
) -> float:
    """Open-Set Classification Rate (OSCR) metric.

    Measures the trade-off between correctly classifying known samples
    and rejecting unknown samples.

    Returns:
        Area under the CCR-FPR curve (OSCR score).
    """
    thresholds = np.linspace(ood_scores_id.min(), ood_scores_ood.max(), n_thresholds)
    ccrs = []
    fprs = []

    y_pred = np.argmax(closed_set_probs, axis=1)
    correct_mask = (y_pred == closed_set_labels)

    for thresh in thresholds:
        # Closed-set: correct AND not flagged as OOD
        id_accepted = ood_scores_id <= thresh
        ccr = float((correct_mask & id_accepted).sum() / len(closed_set_labels))

        # Open-set: fraction of OOD flagged as OOD (TPR for OOD)
        fpr_val = float((ood_scores_ood <= thresh).sum() / len(ood_scores_ood))

        ccrs.append(ccr)
        fprs.append(fpr_val)

    # Area under CCR vs FPR curve
    return float(np.trapz(ccrs, fprs))


def print_osr_report(id_scores, ood_scores, label: str = "Energy"):
    """Print a summary OSR report."""
    auc = auroc(id_scores, ood_scores)
    fpr95 = fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95)
    print(f"\n{'='*40}")
    print(f"OSR Evaluation — {label}")
    print(f"  AUROC       : {auc:.4f}")
    print(f"  FPR@95TPR   : {fpr95:.4f}")
    print(f"{'='*40}\n")
    return {"auroc": auc, "fpr95": fpr95}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    id_scores = rng.normal(-30, 3, 500)   # in-distribution: lower energy
    ood_scores = rng.normal(-10, 5, 200)  # OOD: higher energy
    print_osr_report(id_scores, ood_scores)
