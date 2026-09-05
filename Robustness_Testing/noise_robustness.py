"""
SNR Robustness Testing
=======================
Evaluates model accuracy as a function of Signal-to-Noise Ratio (SNR).
Works with both sklearn-compatible classifiers and PyTorch models.
"""

import numpy as np
from typing import Callable, Iterable


def add_awgn(X_iq: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN at the given SNR to a batch of complex IQ signals.

    Args:
        X_iq: Complex array (N, T).
        snr_db: Desired output SNR in dB.

    Returns:
        Noisy complex array (N, T).
    """
    rng = np.random.default_rng()
    sig_power = np.mean(np.abs(X_iq) ** 2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(X_iq.shape) + 1j * rng.standard_normal(X_iq.shape)
    )
    return X_iq + noise


def snr_sweep(
    predict_fn: Callable,
    X_iq_test: np.ndarray,
    y_test: np.ndarray,
    snr_range: Iterable = range(-5, 31, 5),
    feature_fn: Callable = None,
) -> dict:
    """Evaluate classifier accuracy at each SNR level.

    Args:
        predict_fn: Callable that accepts feature/processed array and returns predicted labels.
        X_iq_test: Clean complex IQ test signals (N, T).
        y_test: True integer labels (N,).
        snr_range: Iterable of SNR values (dB) to test.
        feature_fn: Optional preprocessing function applied to noisy IQ before predict_fn.
                    If None, the raw IQ is passed.

    Returns:
        Dict mapping snr_db (int) → accuracy (float).
    """
    results = {}
    for snr_db in snr_range:
        X_noisy = add_awgn(X_iq_test, snr_db)
        X_input = feature_fn(X_noisy) if feature_fn else X_noisy
        y_pred = predict_fn(X_input)
        acc = float((np.array(y_pred) == np.array(y_test)).mean())
        results[snr_db] = acc
        print(f"  SNR={snr_db:4d} dB  →  acc={acc:.4f}")
    return results


def plot_snr_curve(results: dict, model_name: str = "model", save_path: str = None):
    """Plot accuracy vs SNR curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = sorted(results.keys())
    accs = [results[s] for s in snrs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snrs, accs, "o-", linewidth=2, markersize=6, label=model_name)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Robustness: Accuracy vs SNR", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"SNR curve saved to {save_path}")
    else:
        plt.show()
    plt.close()
    return fig


if __name__ == "__main__":
    # Smoke test with a random classifier
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 500)) + 1j * rng.standard_normal((100, 500))
    y = rng.integers(2, 5, 100)

    def dummy_predict(X_iq):
        return np.random.randint(2, 5, len(X_iq))

    from Preprocessing.features import extract_features
    results = snr_sweep(dummy_predict, X, y, snr_range=range(0, 21, 5))
    print(results)
