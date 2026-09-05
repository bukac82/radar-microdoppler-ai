# Quantum ML Module

Quantum-classical hybrid classifiers for micro-Doppler target recognition.

## Approach

We use **PennyLane** as the quantum computing framework with support for:
- **Quantum Kernel SVM** — encodes classical features into a quantum Hilbert space; kernel matrix computed via quantum circuits
- **Variational Quantum Classifier (VQC)** — parameterized quantum circuit trained end-to-end with gradient descent

## Files

| File | Description |
|------|-------------|
| `qkernel_svm.py` | Quantum kernel construction + SVM training |
| `vqc.py` | Variational Quantum Classifier with angle embedding |
| `circuits.py` | Reusable PennyLane circuit building blocks |
| `train_qml.py` | Unified training entry point |

## Dependencies

```bash
pip install pennylane pennylane-lightning scikit-learn
```

## Usage

```python
from Quantum_ML.qkernel_svm import QuantumKernelSVM

qksvm = QuantumKernelSVM(n_qubits=4)
qksvm.fit(X_train_small, y_train_small)
acc = qksvm.score(X_test_small, y_test_small)
print(f"QK-SVM accuracy: {acc:.4f}")
```

> **Note**: Quantum simulations scale exponentially with qubit count. Use a subset of features (e.g., PCA to 4–8 dims) and a small sample size (≤ 500) for simulation.
