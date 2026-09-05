"""
PyTorch Dataset Wrapper for the Micro-Doppler CSV files
=========================================================
Supports lazy chunked loading for the large 2 GB CSVs.
"""

import numpy as np
from pathlib import Path
from typing import Callable, Optional

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class MicroDopplerDataset(Dataset):
        """PyTorch Dataset that wraps pre-loaded IQ arrays and labels.

        Usage:
            df = load_dataset(nrows=5000)
            X_iq = get_iq_matrix(df)
            y = get_labels(df)
            dataset = MicroDopplerDataset(X_iq, y, transform=spectrogram_transform)

        Args:
            X_iq: Complex numpy array (N, T).
            y: Integer label array (N,).
            transform: Optional callable applied to each sample BEFORE returning.
                       Should map (T,) complex array → Tensor.
            label_offset: Subtract this from labels (e.g., 2 → 0-indexed).
        """

        def __init__(
            self,
            X_iq: np.ndarray,
            y: np.ndarray,
            transform: Optional[Callable] = None,
            label_offset: int = 2,
        ):
            self.X = X_iq
            self.y = (y - label_offset).astype(np.int64)
            self.transform = transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            sample = self.X[idx]
            label = self.y[idx]
            if self.transform:
                sample = self.transform(sample)
            else:
                # Default: stack I and Q as 2-channel float tensor
                iq = np.stack([sample.real, sample.imag], axis=0).astype(np.float32)
                sample = torch.tensor(iq)
            return sample, torch.tensor(label, dtype=torch.long)


    def spectrogram_transform(iq_signal: np.ndarray, nperseg: int = 64, noverlap: int = 48):
        """Transform a single complex IQ signal into a (1, F, T) spectrogram tensor."""
        import torch
        from scipy.signal import stft
        _, _, Zxx = stft(iq_signal, fs=1000.0, nperseg=nperseg, noverlap=noverlap)
        mag = 20 * np.log10(np.abs(Zxx) + 1e-10).astype(np.float32)
        # Normalize per spectrogram
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-10)
        return torch.tensor(mag).unsqueeze(0)  # (1, F, T)


    def get_dataloader(dataset: "MicroDopplerDataset", batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> "DataLoader":
        """Convenience factory for a DataLoader."""
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed.")
    else:
        import torch
        rng = np.random.default_rng(0)
        X_dummy = rng.standard_normal((100, 500)) + 1j * rng.standard_normal((100, 500))
        y_dummy = rng.integers(2, 5, size=100)

        ds = MicroDopplerDataset(X_dummy, y_dummy, transform=spectrogram_transform)
        dl = get_dataloader(ds, batch_size=8)
        sample, label = next(iter(dl))
        print(f"Sample shape: {sample.shape}, Label shape: {label.shape}")
