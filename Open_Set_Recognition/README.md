# Open-Set Recognition Module

Detects unknown aerial targets (e.g., drones, birds) that were not seen during training.

## Problem Statement

Standard classifiers assign every input to a known class. Open-set recognition (OSR)
adds the ability to output **"unknown"** for out-of-distribution (OOD) inputs.

## Methods

| File | Method | Description |
|------|--------|-------------|
| `openmax.py` | OpenMax | Replaces softmax with Weibull-calibrated open-set scores (Bendale & Boult, 2016) |
| `energy_ood.py` | Energy Score | Uses the free energy of DNN logits as an OOD score (Liu et al., 2020) |
| `mahalanobis.py` | Mahalanobis Distance | Class-conditional feature-space distance for OOD detection |
| `evaluate_osr.py` | OSR Evaluation | AUROC, FPR@95TPR, OSCR metrics |

## Usage

```python
from Open_Set_Recognition.energy_ood import EnergyOODDetector

detector = EnergyOODDetector(model, threshold=-25.0)
is_unknown = detector.predict(X_test_spec)  # True = unknown class
```
