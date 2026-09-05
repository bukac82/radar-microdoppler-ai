# Evaluation Module

Cross-model benchmarking and unified metrics for the Radar Micro-Doppler AI project.

## Files

| File | Description |
|------|-------------|
| `metrics.py` | Core metrics: accuracy, F1, ROC-AUC, confusion matrix |
| `benchmark.py` | Compare all models (Classical, QML, DL) on the same test set |
| `plots.py` | Visualization: ROC curves, confusion matrices, SNR curves |
| `report.py` | Auto-generate a Markdown/HTML evaluation report |

## Usage

```python
from Evaluation.benchmark import run_benchmark
from Evaluation.plots import plot_roc_curves

results = run_benchmark(models_dict, X_test_feat, X_test_spec, y_test)
plot_roc_curves(results, save_path="Evaluation/results/roc_curves.png")
```
