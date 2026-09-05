"""
Quantum Kernel SVM
==================
Computes a quantum kernel matrix using a PennyLane circuit and feeds it
to sklearn's SVC with precomputed kernel.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


class QuantumKernelSVM:
    """Quantum Kernel SVM classifier.

    Encodes features via IQP embedding and measures kernel overlap
    using the quantum state fidelity (|<phi(x)|phi(x')>|^2).

    Args:
        n_qubits: Number of qubits (= number of input features after PCA).
        backend: PennyLane device name.
        C: SVM regularization parameter.
    """

    def __init__(self, n_qubits: int = 4, backend: str = "default.qubit", C: float = 1.0):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required. Run: pip install pennylane")
        self.n_qubits = n_qubits
        self.C = C
        self.dev = qml.device(backend, wires=n_qubits)
        self._build_kernel_circuit()
        self.svm = SVC(kernel="precomputed", C=C, probability=True)
        self.X_train_ = None

    def _build_kernel_circuit(self):
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            # Encode x1
            qml.AngleEmbedding(x1, wires=range(self.n_qubits))
            # Adjoint encode x2 (computes <phi(x1)|phi(x2)>)
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        self._circuit = kernel_circuit

    def _kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Fidelity between two encoded states (first element of probs = |0...0>)."""
        probs = self._circuit(x1, x2)
        return float(probs[0])

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute full kernel matrix K[i, j] = kernel(X1[i], X2[j])."""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_value(X1[i], X2[j])
        return K

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the quantum kernel SVM.

        Args:
            X_train: Feature matrix (n_samples, n_qubits). Should be pre-scaled to [0, π].
            y_train: Integer labels.
        """
        self.X_train_ = X_train
        print(f"Computing quantum kernel matrix ({len(X_train)}x{len(X_train)})...")
        K_train = self._compute_kernel_matrix(X_train, X_train)
        self.svm.fit(K_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("Call fit() first.")
        K_test = self._compute_kernel_matrix(X_test, self.X_train_)
        return self.svm.predict(K_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    if not HAS_PENNYLANE:
        print("PennyLane not installed — skipping smoke test.")
    else:
        rng = np.random.default_rng(0)
        n = 30
        X = rng.uniform(0, np.pi, (n, 4))
        y = rng.integers(2, 5, n)

        qksvm = QuantumKernelSVM(n_qubits=4, C=1.0)
        qksvm.fit(X[:20], y[:20])
        acc = qksvm.score(X[20:], y[20:])
        print(f"QK-SVM smoke-test accuracy: {acc:.3f}")
