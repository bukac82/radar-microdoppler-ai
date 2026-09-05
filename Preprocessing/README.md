# Preprocessing Module

Transforms raw IQ signals into features suitable for ML/DL/QML pipelines.

## Files

| File | Description |
|------|-------------|
| `spectrogram.py` | Short-Time Fourier Transform (STFT) → micro-Doppler spectrograms |
| `features.py` | Hand-crafted feature extraction (PRF, blade flash rate, statistical moments, etc.) |
| `normalize.py` | Normalization and standardization utilities |
| `pipeline.py` | Sklearn-compatible preprocessing pipeline |

## Usage

```python
from Preprocessing.spectrogram import compute_stft_spectrogram
from Preprocessing.features import extract_features
from Dataset.loader import load_dataset, get_iq_matrix

df = load_dataset(nrows=500)
X_iq = get_iq_matrix(df)

# Spectrogram
spectrograms = compute_stft_spectrogram(X_iq)   # shape: (N, freq_bins, time_bins)

# Hand-crafted features
X_feat = extract_features(X_iq)                  # shape: (N, n_features)
```
