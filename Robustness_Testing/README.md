# Robustness Testing Module

Evaluates model robustness under realistic signal degradation conditions.

## Test Scenarios

| File | Test | Description |
|------|------|-------------|
| `noise_robustness.py` | SNR sweep | Evaluates accuracy vs. SNR from -5 dB to 30 dB |
| `adversarial.py` | Adversarial attacks | FGSM and PGD attacks on DL models |
| `multipath.py` | Multipath / clutter | Adds multipath reflections to the IQ signal |
| `run_robustness.py` | Full suite runner | Runs all robustness tests and saves results |

## Usage

```python
from Robustness_Testing.noise_robustness import snr_sweep

results = snr_sweep(model, X_iq_test, y_test, snr_range=range(-5, 31, 5))
# Returns dict: {snr_db: accuracy}
```
