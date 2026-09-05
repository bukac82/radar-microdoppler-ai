"""
2-D CNN for Micro-Doppler Spectrogram Classification
======================================================
ResNet-inspired convolutional network operating on STFT spectrograms.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch not installed. Run: pip install torch")


if HAS_TORCH:
    class ResidualBlock(nn.Module):
        """Basic residual block with two 3×3 convolutions."""

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    class MicroDopplerCNN(nn.Module):
        """Lightweight ResNet for micro-Doppler spectrogram classification.

        Args:
            n_classes: Number of target classes (default 3: 2-, 3-, 4-blade).
            in_channels: 1 for grayscale spectrograms.
            dropout: Dropout probability in the classifier head.
        """

        def __init__(self, n_classes: int = 3, in_channels: int = 1, dropout: float = 0.3):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.layer1 = ResidualBlock(32, 64, stride=2)
            self.layer2 = ResidualBlock(64, 128, stride=2)
            self.layer3 = ResidualBlock(128, 256, stride=2)

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            """Args:
                x: Tensor of shape (B, 1, F, T) — batch of spectrograms.
            Returns:
                Logits of shape (B, n_classes).
            """
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return self.head(x)

        def predict_proba(self, x):
            """Softmax probabilities."""
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


    def build_cnn(n_classes: int = 3, in_channels: int = 1) -> "MicroDopplerCNN":
        """Convenience factory for the CNN model."""
        return MicroDopplerCNN(n_classes=n_classes, in_channels=in_channels)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed.")
    else:
        import torch
        model = MicroDopplerCNN(n_classes=3)
        x = torch.randn(4, 1, 33, 15)  # batch of 4 spectrograms
        logits = model(x)
        print(f"CNN output shape: {logits.shape}")  # (4, 3)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
