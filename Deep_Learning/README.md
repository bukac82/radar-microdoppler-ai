# Deep Learning Module

Deep neural network classifiers operating directly on micro-Doppler spectrograms or raw IQ signals.

## Architectures

| File | Architecture | Input |
|------|-------------|-------|
| `cnn.py` | 2-D CNN (ResNet-style) | Spectrogram (F×T) |
| `lstm.py` | Bidirectional LSTM | Raw IQ time series |
| `transformer.py` | Spectrogram Transformer (ViT-inspired) | Spectrogram patches |
| `train_dl.py` | Unified training loop (PyTorch Lightning) | Any above model |
| `dataset_torch.py` | PyTorch Dataset wrapper for the CSV files | — |

## Requirements

```bash
pip install torch torchvision pytorch-lightning
```

## Usage

```python
from Deep_Learning.cnn import MicroDopplerCNN
from Deep_Learning.train_dl import train_model

model = MicroDopplerCNN(n_classes=3)
train_model(model, X_train_spec, y_train, X_val_spec, y_val, epochs=50)
```
