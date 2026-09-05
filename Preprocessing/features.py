"""
Hand-Crafted Feature Extraction
=================================
Extracts physics-informed and statistical features from complex IQ signals
for classical and quantum ML pipelines.
"""

import numpy as np
from scipy.signal import stft
from scipy.stats import kurtosis, skew


def _blade_flash_rate(X_iq: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Estimate blade flash rate (Hz) as the dominant frequency of the
    instantaneous power envelope."""
    power = np.abs(X_iq) ** 2  # shape (N, T)
    # Take FFT of the power envelope along time axis
    fft_mag = np.abs(np.fft.rfft(power, axis=1))
    freqs = np.fft.rfftfreq(power.shape[1], d=1.0 / fs)
    # Exclude DC
    fft_mag[:, 0] = 0
    peak_idx = np.argmax(fft_mag, axis=1)
    return freqs[peak_idx]


def extract_features(X_iq: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Extract a fixed-size feature vector from each IQ signal.

    Features extracted per sample:
      - Mean, std, max, min of instantaneous amplitude (4)
      - Skewness, kurtosis of instantaneous amplitude (2)
      - Mean, std of instantaneous phase (2)
      - Mean, std of instantaneous frequency (2)
      - Blade flash rate estimate (1)
      - Spectral centroid, spectral spread (2)
      - Total energy (1)

    Total: 14 features per sample.

    Args:
        X_iq: Complex array of shape (n_samples, n_timesteps).
        fs: Sampling frequency in Hz.

    Returns:
        Feature matrix of shape (n_samples, 14).
    """
    amp = np.abs(X_iq)                           # (N, T)
    phase = np.unwrap(np.angle(X_iq), axis=1)    # (N, T)
    inst_freq = np.diff(phase, axis=1) * fs / (2 * np.pi)  # (N, T-1)

    # Amplitude stats
    amp_mean = amp.mean(axis=1)
    amp_std = amp.std(axis=1)
    amp_max = amp.max(axis=1)
    amp_min = amp.min(axis=1)
    amp_skew = skew(amp.astype(np.float64), axis=1)
    amp_kurt = kurtosis(amp.astype(np.float64), axis=1)

    # Phase stats
    phase_mean = phase.mean(axis=1)
    phase_std = phase.std(axis=1)

    # Instantaneous frequency stats
    if_mean = inst_freq.mean(axis=1)
    if_std = inst_freq.std(axis=1)

    # Blade flash rate
    bfr = _blade_flash_rate(X_iq, fs=fs)

    # Spectral features from magnitude spectrum
    # rfft requires real input; use instantaneous amplitude (already real)
    mag_spec = np.abs(np.fft.rfft(amp.astype(np.float64), axis=1))  # (N, F)
    freqs = np.fft.rfftfreq(X_iq.shape[1], d=1.0 / fs)
    total_power = mag_spec.sum(axis=1) + 1e-10
    spectral_centroid = (mag_spec * freqs).sum(axis=1) / total_power
    spectral_spread = np.sqrt(
        (mag_spec * (freqs - spectral_centroid[:, None]) ** 2).sum(axis=1) / total_power
    )

    # Total energy
    energy = (amp ** 2).sum(axis=1)

    features = np.column_stack([
        amp_mean, amp_std, amp_max, amp_min,
        amp_skew, amp_kurt,
        phase_mean, phase_std,
        if_mean, if_std,
        bfr,
        spectral_centroid, spectral_spread,
        energy,
    ])
    return features


FEATURE_NAMES = [
    "amp_mean", "amp_std", "amp_max", "amp_min",
    "amp_skew", "amp_kurt",
    "phase_mean", "phase_std",
    "if_mean", "if_std",
    "blade_flash_rate_hz",
    "spectral_centroid", "spectral_spread",
    "energy",
]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 500)) + 1j * rng.standard_normal((20, 500))
    F = extract_features(X)
    print(f"Feature matrix shape: {F.shape}")
    print(f"Features: {FEATURE_NAMES}")
