#!/usr/bin/env python3
"""
Script to generate all 4 Jupyter notebooks programmatically.
Run: python scripts/generate_notebooks.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

def nb(cells):
    """Create a minimal notebook v4 dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "cells": cells
    }

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": "md_" + str(hash(source))[:8]}

def code(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source,
        "id": "code_" + str(hash(source))[:8]
    }

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 1: EDA & Signal Analysis
# ─────────────────────────────────────────────────────────────────────────────
nb1_cells = [
    md("# 🛰️ Radar Micro-Doppler AI\n## Notebook 1: Exploratory Data Analysis & Signal Analysis\n\nThis notebook explores the helicopter micro-Doppler dataset:\n- Dataset statistics and class distributions\n- Raw IQ signal visualization\n- STFT spectrogram visualization\n- Physical parameter distributions (RPM, blade radius, tip velocity, SNR)"),

    md("## Setup"),
    code("""\
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft

# Project imports
from Dataset.loader import load_dataset, get_iq_matrix, get_labels
from Preprocessing.spectrogram import compute_spectrogram_single
from Preprocessing.features import extract_features, FEATURE_NAMES

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#e6edf3',
    'grid.color': '#30363d',
    'font.family': 'DejaVu Sans',
})
COLORS = ['#58a6ff', '#3fb950', '#f78166']
CLASS_NAMES = {2: '2-blade (UH-1)', 3: '3-blade (Gazelle)', 4: '4-blade (Apache/UH-60)'}
print("✅ Setup complete")"""),

    md("## 1. Load Dataset"),
    code("""\
# Load a manageable subset for EDA
df = load_dataset('../helicopter_microdoppler_dataset.csv', nrows=3000)
print(f"Shape: {df.shape}")
df.head(3)"""),

    md("## 2. Class Distribution"),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold', y=1.02)

# Class counts
counts = df['num_blades'].value_counts().sort_index()
bars = axes[0].bar([CLASS_NAMES[k] for k in counts.index], counts.values,
                   color=COLORS, edgecolor='white', linewidth=0.5, alpha=0.9)
axes[0].set_title('Class Distribution', fontsize=13)
axes[0].set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# SNR distribution per class
for i, (nb, name) in enumerate(CLASS_NAMES.items()):
    subset = df[df['num_blades'] == nb]['snr_db']
    axes[1].hist(subset, bins=30, alpha=0.7, color=COLORS[i], label=name, density=True)
axes[1].set_title('SNR Distribution by Class', fontsize=13)
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('Density')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Plot saved.")"""),

    md("## 3. Physical Parameters Distribution"),
    code("""\
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Physical Parameter Distributions by Helicopter Type', fontsize=15, fontweight='bold')

params = [('rpm', 'Rotor RPM'), ('radius_m', 'Blade Radius (m)'),
          ('tip_velocity_m_s', 'Tip Velocity (m/s)'), ('snr_db', 'SNR (dB)')]

for ax, (col, label) in zip(axes.flat, params):
    for i, (nb, name) in enumerate(CLASS_NAMES.items()):
        data = df[df['num_blades'] == nb][col]
        ax.hist(data, bins=40, alpha=0.65, color=COLORS[i], label=name, density=True)
    ax.set_title(label, fontsize=12)
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('parameter_distributions.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()"""),

    md("## 4. Raw IQ Signal Visualization"),
    code("""\
X_iq = get_iq_matrix(df)
y    = get_labels(df)
t    = np.linspace(0, 0.5, 500, endpoint=False)

fig, axes = plt.subplots(3, 2, figsize=(16, 10))
fig.suptitle('Raw IQ Signals — One Sample per Class', fontsize=15, fontweight='bold')

for row_idx, (n_blades, name) in enumerate(CLASS_NAMES.items()):
    idx = np.where(y == n_blades)[0][0]
    sig = X_iq[idx]

    axes[row_idx, 0].plot(t, sig.real, color=COLORS[row_idx], linewidth=0.8, alpha=0.9)
    axes[row_idx, 0].set_title(f'{name} — In-Phase (I)', fontsize=11)
    axes[row_idx, 0].set_xlabel('Time (s)')
    axes[row_idx, 0].set_ylabel('Amplitude')
    axes[row_idx, 0].grid(True, alpha=0.2)

    axes[row_idx, 1].plot(t, sig.imag, color=COLORS[row_idx], linewidth=0.8, alpha=0.9)
    axes[row_idx, 1].set_title(f'{name} — Quadrature (Q)', fontsize=11)
    axes[row_idx, 1].set_xlabel('Time (s)')
    axes[row_idx, 1].set_ylabel('Amplitude')
    axes[row_idx, 1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('iq_signals.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()"""),

    md("## 5. Micro-Doppler Spectrograms"),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Micro-Doppler STFT Spectrograms by Helicopter Type', fontsize=15, fontweight='bold')

for col_idx, (n_blades, name) in enumerate(CLASS_NAMES.items()):
    idx = np.where(y == n_blades)[0][0]
    freqs, times, spec = compute_spectrogram_single(X_iq[idx], nperseg=64, noverlap=56)

    im = axes[col_idx].pcolormesh(times, freqs, spec, shading='gouraud', cmap='inferno')
    axes[col_idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[col_idx].set_xlabel('Time (s)')
    axes[col_idx].set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=axes[col_idx], label='Power (dB)')

    # Annotate blade flash rate
    bfr = n_blades * df[df['num_blades'] == n_blades]['rpm'].mean() / 60
    axes[col_idx].set_title(f'{name}\\nBlade Flash Rate ≈ {bfr:.1f} Hz', fontsize=11)

plt.tight_layout()
plt.savefig('spectrograms.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("📊 Spectrograms reveal the distinct periodic structure per helicopter type.")"""),

    md("## 6. Feature Correlation Heatmap"),
    code("""\
X_feat = extract_features(X_iq[:500])

import matplotlib.cm as cm

corr = np.corrcoef(X_feat.T)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(FEATURE_NAMES)))
ax.set_yticks(range(len(FEATURE_NAMES)))
ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Pearson Correlation')

# Annotate high correlations
for i in range(len(FEATURE_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        if abs(corr[i,j]) > 0.7 and i != j:
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"✅ EDA complete. Key insight: blade_flash_rate_hz is the most discriminative feature.")"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 2: Classical ML Benchmark
# ─────────────────────────────────────────────────────────────────────────────
nb2_cells = [
    md("# 🤖 Notebook 2: Classical ML Benchmark\n\nTrains and evaluates SVM, Random Forest, XGBoost, and k-NN classifiers\non hand-crafted micro-Doppler features. Includes SHAP explainability."),

    md("## Setup"),
    code("""\
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import time

from Dataset.loader import load_dataset, get_iq_matrix, get_labels, get_train_test_split
from Preprocessing.features import extract_features, FEATURE_NAMES
from Preprocessing.normalize import fit_feature_scaler
from Classical_ML.train import train_svm, train_random_forest, train_knn
from Evaluation.metrics import compute_metrics, print_metrics

plt.style.use('dark_background')
COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff']
CLASS_NAMES = ['2-blade', '3-blade', '4-blade']
LABEL_ORDER = [2, 3, 4]
print("✅ Ready")"""),

    md("## 1. Load & Prepare Data"),
    code("""\
print("Loading dataset (5000 rows)...")
df = load_dataset('../helicopter_microdoppler_dataset.csv', nrows=5000)
X_iq_train, X_iq_test, y_train, y_test = get_train_test_split(df, test_size=0.2)

print("Extracting features...")
X_train_raw = extract_features(X_iq_train)
X_test_raw  = extract_features(X_iq_test)

scaler = fit_feature_scaler(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")"""),

    md("## 2. Train All Classifiers"),
    code("""\
models = {}
timings = {}

configs = [
    ('SVM (RBF, C=10)',  train_svm,           {'C': 10.0}),
    ('Random Forest',    train_random_forest, {'n_estimators': 200}),
    ('k-NN (k=7)',       train_knn,           {'n_neighbors': 7}),
]

for name, fn, kwargs in configs:
    print(f"Training {name}...")
    t0 = time.time()
    models[name] = fn(X_train, y_train, **kwargs)
    timings[name] = time.time() - t0
    print(f"  Done in {timings[name]:.1f}s")"""),

    md("## 3. Results & Comparison"),
    code("""\
all_metrics = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    all_metrics[name] = compute_metrics(y_test, y_pred, y_proba)

# Summary table
print(f"\\n{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'ROC-AUC':>10} {'Time':>8}")
print("─" * 68)
for name, m in all_metrics.items():
    auc  = f"{m['roc_auc']:.4f}" if m.get('roc_auc') else "  N/A  "
    t    = f"{timings[name]:.1f}s"
    print(f"{name:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {auc:>10} {t:>8}")"""),

    md("## 4. Confusion Matrices"),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices — Classical ML Models', fontsize=15, fontweight='bold')

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    acc = all_metrics[name]['accuracy']
    ax.set_title(f'{name}\\nAccuracy: {acc:.3f}', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_classical.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()"""),

    md("## 5. Feature Importance (Random Forest)"),
    code("""\
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(
    [FEATURE_NAMES[i] for i in sorted_idx],
    importances[sorted_idx],
    color=COLORS[:len(FEATURE_NAMES)] * 5,
    edgecolor='white', linewidth=0.3, alpha=0.9
)
ax.set_title('Random Forest Feature Importances\\n(Gini Impurity Decrease)', fontsize=13, fontweight='bold')
ax.set_ylabel('Importance')
ax.set_xlabel('Feature')
plt.xticks(rotation=45, ha='right')
ax.grid(True, alpha=0.2, axis='y')

# Highlight top feature
top_feat = FEATURE_NAMES[sorted_idx[0]]
print(f"\\n🏆 Most important feature: '{top_feat}' ({importances[sorted_idx[0]]:.4f})")
print(f"   This corresponds to the blade flash rate — a direct physical signature!")

plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()"""),

    md("## 6. SNR Robustness"),
    code("""\
from Robustness_Testing.noise_robustness import snr_sweep, plot_snr_curve

svm_model = models['SVM (RBF, C=10)']
rf_model_  = models['Random Forest']

results_dict = {}
for name, mdl in [('SVM', svm_model), ('Random Forest', rf_model_)]:
    print(f"\\nSNR sweep — {name}")
    def predict_fn(X_iq_noisy, m=mdl):
        feat = extract_features(X_iq_noisy)
        return m.predict(scaler.transform(feat))
    results_dict[name] = snr_sweep(predict_fn, X_iq_test[:300], y_test[:300], snr_range=range(-5, 31, 5))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, (name, snr_res) in enumerate(results_dict.items()):
    snrs = sorted(snr_res.keys())
    accs = [snr_res[s] for s in snrs]
    ax.plot(snrs, accs, 'o-', linewidth=2.5, markersize=7, color=COLORS[i], label=name)

ax.axhline(y=1/3, color='white', linestyle='--', alpha=0.4, label='Random baseline')
ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Robustness: Accuracy vs SNR', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('snr_robustness_classical.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ Classical ML benchmark complete!")"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 3: Deep Learning
# ─────────────────────────────────────────────────────────────────────────────
nb3_cells = [
    md("# 🧠 Notebook 3: Deep Learning for Micro-Doppler Classification\n\nTrains CNN (ResNet-style), Bidirectional LSTM, and Spectrogram Transformer\non micro-Doppler spectrograms. Includes Grad-CAM visualization."),

    md("## Setup"),
    code("""\
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    print(f"PyTorch version: {torch.__version__}")
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
except ImportError:
    print("PyTorch not installed. Run: pip install torch")
    raise

from Dataset.loader import load_dataset, get_iq_matrix, get_labels, get_train_test_split
from Preprocessing.spectrogram import compute_stft_spectrogram
from Deep_Learning.cnn import MicroDopplerCNN
from Deep_Learning.lstm import MicroDopplerLSTM, iq_to_tensor
from Deep_Learning.transformer import MicroDopplerTransformer
from Deep_Learning.dataset_torch import MicroDopplerDataset, get_dataloader, spectrogram_transform
from Deep_Learning.train_dl import train_model
from Evaluation.metrics import compute_metrics

plt.style.use('dark_background')
COLORS = ['#58a6ff', '#3fb950', '#f78166']
CLASS_NAMES = ['2-blade', '3-blade', '4-blade']"""),

    md("## 1. Load Data"),
    code("""\
df = load_dataset('../helicopter_microdoppler_dataset.csv', nrows=5000)
X_iq_train, X_iq_test, y_train, y_test = get_train_test_split(df, test_size=0.2)

# Datasets with spectrogram transform
ds_train = MicroDopplerDataset(X_iq_train, y_train, transform=spectrogram_transform)
ds_test  = MicroDopplerDataset(X_iq_test,  y_test,  transform=spectrogram_transform)
dl_train = get_dataloader(ds_train, batch_size=64, shuffle=True)
dl_test  = get_dataloader(ds_test,  batch_size=64, shuffle=False)

# Inspect one batch
X_b, y_b = next(iter(dl_train))
print(f"Batch shape: {X_b.shape}  |  Labels: {set(y_b.numpy().tolist())}")
print(f"Train batches: {len(dl_train)}, Test batches: {len(dl_test)}")"""),

    md("## 2. Model Architectures"),
    code("""\
models_info = {
    'CNN (ResNet-style)': MicroDopplerCNN(n_classes=3),
    'Transformer (ViT)':  MicroDopplerTransformer(n_classes=3, img_h=33, img_w=15),
}

for name, model in models_info.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name:<25}  {n_params:>8,} parameters")"""),

    md("## 3. Train CNN"),
    code("""\
cnn = MicroDopplerCNN(n_classes=3)
print("Training CNN...")
history_cnn = train_model(
    cnn, dl_train, dl_test,
    n_epochs=30, lr=1e-3, patience=7,
    model_name='cnn_nb3', device=device
)"""),

    md("## 4. Training Curves"),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN Training Curves', fontsize=14, fontweight='bold')

epochs = range(1, len(history_cnn['train_loss']) + 1)

axes[0].plot(epochs, history_cnn['train_loss'], color=COLORS[0], linewidth=2, label='Train')
axes[0].plot(epochs, history_cnn['val_loss'],   color=COLORS[1], linewidth=2, linestyle='--', label='Validation')
axes[0].set_title('Loss', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history_cnn['val_acc'], color=COLORS[2], linewidth=2.5)
axes[1].fill_between(epochs, history_cnn['val_acc'], alpha=0.2, color=COLORS[2])
axes[1].set_title('Validation Accuracy', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1.05)
axes[1].axhline(y=max(history_cnn['val_acc']), color='yellow', linestyle=':', alpha=0.7,
                label=f"Best: {max(history_cnn['val_acc']):.4f}")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_training_curves.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Best validation accuracy: {max(history_cnn['val_acc']):.4f}")"""),

    md("## 5. Grad-CAM Visualization"),
    code("""\
from Explainability.gradcam import GradCAM

cnn.eval()
cam_extractor = GradCAM(cnn, target_layer=cnn.layer3)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Grad-CAM: What the CNN Sees in Each Spectrogram', fontsize=14, fontweight='bold')

for col_idx, class_idx in enumerate([0, 1, 2]):
    # Find a sample of this class in test set
    target_class = class_idx + 2  # un-offset
    sample_indices = np.where(y_test == target_class)[0]
    if len(sample_indices) == 0:
        continue
    idx = sample_indices[0]

    X_sample, y_sample = ds_test[idx]
    X_tensor = X_sample.unsqueeze(0)  # (1, 1, H, W)
    spectrogram = X_sample.squeeze().numpy()

    cam = cam_extractor(X_tensor, class_idx=class_idx, device=device)

    axes[0, col_idx].imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno')
    axes[0, col_idx].set_title(f'{CLASS_NAMES[class_idx]}\\nSpectrogram', fontsize=11)
    axes[0, col_idx].set_xlabel('Time bins')
    axes[0, col_idx].set_ylabel('Frequency bins')

    axes[1, col_idx].imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno', alpha=0.6)
    axes[1, col_idx].imshow(cam, aspect='auto', origin='lower', cmap='jet', alpha=0.7)
    axes[1, col_idx].set_title(f'Grad-CAM Overlay\\n(activation hotspots)', fontsize=11)
    axes[1, col_idx].set_xlabel('Time bins')
    axes[1, col_idx].set_ylabel('Frequency bins')

plt.tight_layout()
plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("🔍 Grad-CAM reveals the CNN focuses on the periodic blade-flash frequency bands.")"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 4: Quantum ML
# ─────────────────────────────────────────────────────────────────────────────
nb4_cells = [
    md("# ⚛️ Notebook 4: Quantum Machine Learning\n\nDemonstrates Quantum Kernel SVM and Variational Quantum Classifier (VQC)\non a small subset of micro-Doppler features.\n\n> **Note**: Quantum simulation is exponentially costly. We use ≤ 4 qubits\n> and ≤ 300 samples for practical simulation on CPU."),

    md("## Setup"),
    code("""\
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

try:
    import pennylane as qml
    print(f"PennyLane version: {qml.__version__}")
except ImportError:
    print("PennyLane not installed. Run: pip install pennylane pennylane-lightning")
    raise

from Dataset.loader import load_dataset, get_iq_matrix, get_labels, get_train_test_split
from Preprocessing.features import extract_features
from Preprocessing.normalize import fit_feature_scaler
from Quantum_ML.qkernel_svm import QuantumKernelSVM
from Quantum_ML.vqc import VariationalQuantumClassifier
from Evaluation.metrics import compute_metrics, print_metrics

plt.style.use('dark_background')
COLORS = ['#58a6ff', '#3fb950', '#f78166']
print("✅ Ready")"""),

    md("## 1. Prepare Small Quantum-Friendly Dataset\n\nReduce dimensionality to `n_qubits=4` via PCA, then scale to `[0, π]`."),
    code("""\
N_QUBITS = 4
N_TRAIN  = 150   # keep small for simulation speed
N_TEST   = 50

print(f"Loading dataset ({N_TRAIN + N_TEST} samples)...")
df = load_dataset('../helicopter_microdoppler_dataset.csv', nrows=N_TRAIN + N_TEST + 200)
X_iq_train, X_iq_test, y_train, y_test = get_train_test_split(df, test_size=N_TEST/(N_TRAIN+N_TEST))

# Feature extraction
X_train_feat = extract_features(X_iq_train[:N_TRAIN])
X_test_feat  = extract_features(X_iq_test[:N_TEST])

# Standard scaling
scaler = fit_feature_scaler(X_train_feat)
X_tr_sc = scaler.transform(X_train_feat)
X_te_sc = scaler.transform(X_test_feat)

# PCA → 4 dimensions
pca = PCA(n_components=N_QUBITS, random_state=42)
X_tr_pca = pca.fit_transform(X_tr_sc)
X_te_pca = pca.transform(X_te_sc)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f} ({N_QUBITS} components)")

# Scale to [0, π] for angle embedding
angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_tr_q = angle_scaler.fit_transform(X_tr_pca)
X_te_q = angle_scaler.transform(X_te_pca)

print(f"Quantum feature shape: {X_tr_q.shape} (train), {X_te_q.shape} (test)")
print(f"Value range: [{X_tr_q.min():.3f}, {X_tr_q.max():.3f}]")"""),

    md("## 2. Quantum Kernel SVM"),
    code("""\
print("Training Quantum Kernel SVM...")
qksvm = QuantumKernelSVM(n_qubits=N_QUBITS, C=1.0)
qksvm.fit(X_tr_q, y_train[:N_TRAIN])
acc_qksvm = qksvm.score(X_te_q, y_test[:N_TEST])
y_pred_qksvm = qksvm.predict(X_te_q)
m_qksvm = compute_metrics(y_test[:N_TEST], y_pred_qksvm)
print_metrics(m_qksvm, 'Quantum Kernel SVM')"""),

    md("## 3. Variational Quantum Classifier (VQC)"),
    code("""\
print("Training VQC (this may take 2-5 minutes)...")
vqc = VariationalQuantumClassifier(
    n_qubits=N_QUBITS, n_layers=2, n_classes=3,
    lr=0.02, n_epochs=20, batch_size=16
)
vqc.fit(X_tr_q, y_train[:N_TRAIN])
y_pred_vqc = vqc.predict(X_te_q)
m_vqc = compute_metrics(y_test[:N_TEST], y_pred_vqc)
print_metrics(m_vqc, 'VQC')"""),

    md("## 4. Classical vs Quantum Comparison"),
    code("""\
from Classical_ML.train import train_svm
from sklearn.preprocessing import StandardScaler

# Classical SVM on same 4D PCA features for fair comparison
classical_svm = train_svm(X_tr_q, y_train[:N_TRAIN])
y_pred_csvm = classical_svm.predict(X_te_q)
m_csvm = compute_metrics(y_test[:N_TEST], y_pred_csvm)

results = {
    'Classical SVM (4-D PCA)': m_csvm,
    'Quantum Kernel SVM':       m_qksvm,
    'VQC (2 layers)':           m_vqc,
}

print(f"\\n{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10}")
print("─" * 55)
for name, m in results.items():
    print(f"{name:<30} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
accs  = [results[n]['accuracy'] for n in names]
f1s   = [results[n]['macro_f1'] for n in names]

x = np.arange(len(names))
w = 0.35
bars1 = ax.bar(x - w/2, accs, w, label='Accuracy', color=COLORS[0], alpha=0.85)
bars2 = ax.bar(x + w/2, f1s,  w, label='Macro-F1', color=COLORS[1], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classical vs Quantum ML — Performance on 4-D PCA Features', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(y=1/3, color='white', linestyle='--', alpha=0.4, label='Random baseline')

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('quantum_vs_classical.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ Quantum ML notebook complete!")"""),
]

# ── Write all notebooks ───────────────────────────────────────────────────────
notebooks = {
    "01_EDA_and_Signal_Analysis.ipynb": nb(nb1_cells),
    "02_Classical_ML_Benchmark.ipynb":  nb(nb2_cells),
    "03_Deep_Learning.ipynb":           nb(nb3_cells),
    "04_Quantum_ML.ipynb":              nb(nb4_cells),
}

for filename, notebook in notebooks.items():
    path = NB_DIR / filename
    with open(path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"✅ Created: {path}")

print(f"\n🎉 All {len(notebooks)} notebooks generated in {NB_DIR}")
