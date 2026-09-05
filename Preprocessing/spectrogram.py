"""
Spectrogram Generation
======================
Computes micro-Doppler spectrograms from complex IQ signals using STFT.
"""

import numpy as np
from scipy.signal import stft, get_window


def compute_stft_spectrogram(
    X_iq: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 64,
    noverlap: int = 48,
    window: str = "hann",
    log_scale: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute STFT spectrograms for a batch of complex IQ signals.

    Args:
        X_iq: Complex array of shape (n_samples, n_timesteps).
        fs: Sampling frequency in Hz.
        nperseg: FFT window length.
        noverlap: Number of overlapping samples between windows.
        window: Window type (see scipy.signal.get_window).
        log_scale: If True, return 20*log10(|STFT|). Otherwise return |STFT|.
        eps: Small epsilon added before log to avoid -inf.

    Returns:
        Real array of shape (n_samples, n_freq_bins, n_time_bins).
    """
    spectrograms = []
    for sig in X_iq:
        _, _, Zxx = stft(sig, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        if log_scale:
            mag = 20 * np.log10(mag + eps)
        spectrograms.append(mag)
    return np.array(spectrograms)


def compute_spectrogram_single(
    iq_signal: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 64,
    noverlap: int = 48,
    window: str = "hann",
    log_scale: bool = True,
    eps: float = 1e-10,
):
    """STFT spectrogram for a single IQ signal.

    Returns:
        freqs, times, spectrogram (2-D magnitude array).
    """
    freqs, times, Zxx = stft(iq_signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    if log_scale:
        mag = 20 * np.log10(mag + eps)
    return freqs, times, mag


if __name__ == "__main__":
    # Quick smoke-test
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 500)) + 1j * rng.standard_normal((10, 500))
    S = compute_stft_spectrogram(X)
    print(f"Spectrogram batch shape: {S.shape}")
