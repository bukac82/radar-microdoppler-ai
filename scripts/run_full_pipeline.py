#!/usr/bin/env python3
"""
Full Pipeline Runner
====================
Runs the complete Radar Micro-Doppler AI pipeline end-to-end:
  1. Load & preprocess a subset of the dataset
  2. Train all Classical ML models
  3. Train Deep Learning CNN (if PyTorch available)
  4. Run robustness (SNR sweep)
  5. Print benchmark comparison table

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --nrows 2000 --dataset extended
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Dataset.loader import load_dataset, get_train_test_split
from Preprocessing.features import extract_features
from Preprocessing.normalize import fit_feature_scaler
from Classical_ML.train import train_svm, train_random_forest, train_knn
from Classical_ML.evaluate import evaluate_model
from Evaluation.metrics import compute_metrics, print_metrics
from Robustness_Testing.noise_robustness import snr_sweep

try:
    import torch
    from Deep_Learning.cnn import MicroDopplerCNN
    from Deep_Learning.dataset_torch import MicroDopplerDataset, get_dataloader, spectrogram_transform
    from Deep_Learning.train_dl import train_model
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       Radar Micro-Doppler AI — Full Pipeline Runner          ║
║   Classical ML  ·  Deep Learning  ·  Robustness Testing     ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_pipeline(nrows: int = 5000, dataset: str = "base", test_size: float = 0.2):
    print(BANNER)
    t_start = time.time()

    # ── 1. Load Data ──────────────────────────────────────────
    print(f"[1/5] Loading dataset ({nrows} rows, {dataset})...")
    extended = dataset == "extended"
    csv_path = ROOT / (
        "helicopter_microdoppler_extended_dataset.csv" if extended
        else "helicopter_microdoppler_dataset.csv"
    )
    if not csv_path.exists():
        print(f"  ✗ Dataset not found at {csv_path}")
        print("    Run: python scripts/download_data.py --source generate")
        sys.exit(1)

    df = load_dataset(csv_path, nrows=nrows, extended=extended)
    X_iq_train, X_iq_test, y_train, y_test = get_train_test_split(
        df, test_size=test_size, extended=extended
    )
    print(f"  Train: {X_iq_train.shape}, Test: {X_iq_test.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ── 2. Feature Extraction ─────────────────────────────────
    print("\n[2/5] Extracting features...")
    X_train_feat = extract_features(X_iq_train)
    X_test_feat  = extract_features(X_iq_test)
    scaler = fit_feature_scaler(X_train_feat)
    X_train_sc = scaler.transform(X_train_feat)
    X_test_sc  = scaler.transform(X_test_feat)
    print(f"  Feature matrix: {X_train_sc.shape} (train), {X_test_sc.shape} (test)")

    # ── 3. Classical ML ───────────────────────────────────────
    print("\n[3/5] Training Classical ML models...")
    results = {}

    for name, train_fn, kwargs in [
        ("SVM (RBF)",       train_svm,           {}),
        ("Random Forest",   train_random_forest, {"n_estimators": 100}),
        ("k-NN (k=7)",      train_knn,           {}),
    ]:
        t0 = time.time()
        model = train_fn(X_train_sc, y_train, **kwargs)
        elapsed = time.time() - t0
        r = evaluate_model(model, X_test_sc, y_test, model_name=name)
        r["train_time_s"] = elapsed
        results[name] = r

    # ── 4. Deep Learning CNN ─────────────────────────────────
    if HAS_TORCH:
        print("\n[4/5] Training Deep Learning CNN (3 epochs smoke-test)...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        ds_tr = MicroDopplerDataset(X_iq_train, y_train, transform=spectrogram_transform)
        ds_te = MicroDopplerDataset(X_iq_test,  y_test,  transform=spectrogram_transform)
        dl_tr = get_dataloader(ds_tr, batch_size=32, shuffle=True)
        dl_te = get_dataloader(ds_te, batch_size=32, shuffle=False)

        cnn = MicroDopplerCNN(n_classes=3)
        t0 = time.time()
        history = train_model(cnn, dl_tr, dl_te, n_epochs=5, model_name="cnn_pipeline", device=device)
        elapsed = time.time() - t0

        # Predict
        cnn.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_b, y_b in dl_te:
                logits = cnn(X_b.to(device))
                preds.extend(logits.argmax(1).cpu().numpy() + 2)  # un-offset
                trues.extend(y_b.numpy() + 2)
        r = compute_metrics(np.array(trues), np.array(preds))
        r["train_time_s"] = elapsed
        results["CNN (ResNet-style)"] = r
    else:
        print("\n[4/5] Skipping Deep Learning (PyTorch not installed)")

    # ── 5. Robustness SNR Sweep ───────────────────────────────
    print("\n[5/5] Running SNR robustness sweep (SVM)...")
    svm_model = train_svm(X_train_sc, y_train)

    def svm_predict_iq(X_iq):
        feat = extract_features(X_iq)
        feat_sc = scaler.transform(feat)
        return svm_model.predict(feat_sc)

    snr_results = snr_sweep(svm_predict_iq, X_iq_test[:200], y_test[:200], snr_range=range(0, 26, 5))

    # ── Summary Table ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Time (s)':>10}")
    print(f"  {'-'*60}")
    for name, m in results.items():
        t = f"{m.get('train_time_s', 0):.1f}s"
        print(f"  {name:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {t:>10}")
    print(f"{'='*65}")

    print(f"\n  SNR Robustness (SVM): {snr_results}")
    print(f"\n✅ Pipeline complete in {time.time() - t_start:.1f}s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=5000, help="Rows to load (default 5000)")
    parser.add_argument("--dataset", choices=["base", "extended"], default="base")
    args = parser.parse_args()
    run_pipeline(nrows=args.nrows, dataset=args.dataset)
