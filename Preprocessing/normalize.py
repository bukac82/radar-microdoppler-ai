"""
Normalization Utilities
=======================
Normalization and standardization helpers for IQ signals and feature matrices.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path


def normalize_spectrogram(S: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize a batch of spectrograms per sample.

    Args:
        S: Array of shape (N, F, T).
        method: 'minmax' scales each spectrogram to [0, 1].
                'zscore' standardizes each spectrogram to zero mean / unit std.

    Returns:
        Normalized array of same shape.
    """
    S_norm = np.empty_like(S, dtype=np.float32)
    for i, s in enumerate(S):
        if method == "minmax":
            lo, hi = s.min(), s.max()
            S_norm[i] = (s - lo) / (hi - lo + 1e-10)
        elif method == "zscore":
            S_norm[i] = (s - s.mean()) / (s.std() + 1e-10)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'minmax' or 'zscore'.")
    return S_norm


def fit_feature_scaler(X_train: np.ndarray, scaler_type: str = "standard") -> object:
    """Fit a scaler on training features.

    Args:
        X_train: Feature matrix (n_train, n_features).
        scaler_type: 'standard' (zero mean, unit var) or 'minmax' ([0,1]).

    Returns:
        Fitted sklearn scaler object.
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type '{scaler_type}'.")
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler, path: str | Path):
    """Persist a fitted scaler to disk."""
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def load_scaler(path: str | Path):
    """Load a previously saved scaler."""
    return joblib.load(path)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 14))
    scaler = fit_feature_scaler(X)
    X_scaled = scaler.transform(X)
    print(f"Scaled feature stats — mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
