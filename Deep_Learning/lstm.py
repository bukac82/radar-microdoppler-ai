"""
Bidirectional LSTM for Micro-Doppler IQ Time Series
=====================================================
Processes raw complex IQ as a 2-channel (I, Q) real-valued sequence.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class MicroDopplerLSTM(nn.Module):
        """Bidirectional LSTM classifier for IQ time series.

        Args:
            input_size: 2 (I and Q channels).
            hidden_size: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            n_classes: Number of target classes.
            dropout: Dropout between LSTM layers.
        """

        def __init__(
            self,
            input_size: int = 2,
            hidden_size: int = 128,
            num_layers: int = 2,
            n_classes: int = 3,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
                nn.ReLU(),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            """Args:
                x: Tensor (B, T, 2) — batch of IQ sequences.
            Returns:
                Logits (B, n_classes).
            """
            out, _ = self.lstm(x)  # (B, T, 2*hidden)
            # Use last time step output
            out = out[:, -1, :]
            return self.head(out)


    def iq_to_tensor(X_iq_complex, device="cpu"):
        """Convert complex IQ array (N, T) to (N, T, 2) float tensor."""
        import numpy as np
        import torch
        X = np.stack([X_iq_complex.real, X_iq_complex.imag], axis=-1).astype(np.float32)
        return torch.tensor(X, device=device)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed.")
    else:
        import torch
        model = MicroDopplerLSTM(n_classes=3)
        x = torch.randn(8, 500, 2)  # batch of 8, 500 timesteps, I+Q
        logits = model(x)
        print(f"LSTM output shape: {logits.shape}")  # (8, 3)
