"""
Unit tests for Preprocessing module.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Preprocessing.spectrogram import compute_stft_spectrogram, compute_spectrogram_single
from Preprocessing.features import extract_features, FEATURE_NAMES
from Preprocessing.normalize import normalize_spectrogram, fit_feature_scaler


def _make_iq(n=20, t=500):
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, t)) + 1j * rng.standard_normal((n, t))


class TestSpectrogram:
    def test_batch_shape(self):
        X = _make_iq(10)
        S = compute_stft_spectrogram(X, nperseg=64, noverlap=48)
        assert S.ndim == 3
        assert S.shape[0] == 10

    def test_single_output(self):
        X = _make_iq(1)[0]
        f, t, spec = compute_spectrogram_single(X)
        assert spec.ndim == 2
        assert len(f) == spec.shape[0]
        assert len(t) == spec.shape[1]

    def test_log_vs_linear(self):
        X = _make_iq(5)
        S_log = compute_stft_spectrogram(X, log_scale=True)
        S_lin = compute_stft_spectrogram(X, log_scale=False)
        assert not np.allclose(S_log, S_lin)


class TestFeatures:
    def test_shape(self):
        X = _make_iq(30)
        F = extract_features(X)
        assert F.shape == (30, len(FEATURE_NAMES))

    def test_no_nans(self):
        X = _make_iq(20)
        F = extract_features(X)
        assert not np.any(np.isnan(F)), "Features contain NaN values"

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 14


class TestNormalize:
    def test_minmax_spectrogram(self):
        rng = np.random.default_rng(0)
        S = rng.standard_normal((10, 33, 15)).astype(np.float32)
        S_norm = normalize_spectrogram(S, method="minmax")
        assert S_norm.min() >= 0.0 - 1e-6
        assert S_norm.max() <= 1.0 + 1e-6

    def test_scaler_fit(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 14))
        scaler = fit_feature_scaler(X)
        X_scaled = scaler.transform(X)
        assert abs(X_scaled.mean()) < 0.1
        assert abs(X_scaled.std() - 1.0) < 0.1
