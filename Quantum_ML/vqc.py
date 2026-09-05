"""
Variational Quantum Classifier (VQC)
=====================================
A parameterized quantum circuit trained end-to-end via gradient descent.
Uses PennyLane with angle embedding + StronglyEntanglingLayers.
"""

import numpy as np

try:
    import pennylane as qml
    import pennylane.numpy as pnp
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


class VariationalQuantumClassifier:
    """VQC with angle embedding and strongly entangling layers.

    Args:
        n_qubits: Number of qubits (= number of input features after PCA).
        n_layers: Number of variational layers.
        n_classes: Number of output classes (3 for 2-, 3-, 4-blade).
        lr: Learning rate for the Adam optimizer.
        n_epochs: Training epochs.
        batch_size: Mini-batch size.
        backend: PennyLane device.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_classes: int = 3,
        lr: float = 0.01,
        n_epochs: int = 30,
        batch_size: int = 16,
        backend: str = "default.qubit",
    ):
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required. Run: pip install pennylane")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.dev = qml.device(backend, wires=n_qubits)
        self.weights_ = None
        self.label_map_ = None
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(min(self.n_classes, self.n_qubits))]

        self._circuit = circuit

    def _forward(self, weights, X: np.ndarray) -> np.ndarray:
        """Run circuit for a batch and return logit-like outputs."""
        outputs = []
        for x in X:
            out = self._circuit(weights, x)
            outputs.append(out)
        return pnp.array(outputs)

    def _softmax(self, z):
        e = pnp.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _cross_entropy_loss(self, weights, X, y_onehot):
        logits = self._forward(weights, X)
        probs = self._softmax(logits)
        log_probs = pnp.log(probs + 1e-10)
        return -pnp.mean((y_onehot * log_probs).sum(axis=1))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the VQC.

        Args:
            X_train: Feature matrix (n_samples, n_qubits), values in [0, π].
            y_train: Integer labels.
        """
        # Map labels to 0-indexed
        unique_labels = sorted(set(y_train.tolist()))
        self.label_map_ = {v: i for i, v in enumerate(unique_labels)}
        self.inv_label_map_ = {i: v for v, i in self.label_map_.items()}
        y_idx = np.array([self.label_map_[y] for y in y_train])

        # One-hot encode
        n = len(y_idx)
        y_oh = np.zeros((n, self.n_classes))
        y_oh[np.arange(n), y_idx] = 1.0
        y_oh = pnp.array(y_oh)

        # Initialize weights
        rng = np.random.default_rng(42)
        self.weights_ = pnp.array(
            rng.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits, 3)),
            requires_grad=True,
        )

        opt = qml.AdamOptimizer(stepsize=self.lr)
        X_pnp = pnp.array(X_train)

        print(f"Training VQC ({self.n_qubits} qubits, {self.n_layers} layers)...")
        for epoch in range(self.n_epochs):
            # Mini-batch
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start: start + self.batch_size]
                X_b = X_pnp[batch_idx]
                y_b = y_oh[batch_idx]
                self.weights_, loss = opt.step_and_cost(
                    lambda w: self._cross_entropy_loss(w, X_b, y_b),
                    self.weights_,
                )
            if (epoch + 1) % 5 == 0:
                total_loss = self._cross_entropy_loss(self.weights_, X_pnp, y_oh)
                print(f"  Epoch {epoch+1}/{self.n_epochs} — loss: {float(total_loss):.4f}")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        logits = self._forward(self.weights_, pnp.array(X_test))
        class_idx = pnp.argmax(logits, axis=1)
        return np.array([self.inv_label_map_[int(i)] for i in class_idx])

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.predict(X_test)
        return float((y_pred == y_test).mean())


if __name__ == "__main__":
    if not HAS_PENNYLANE:
        print("PennyLane not installed — skipping VQC smoke test.")
    else:
        rng = np.random.default_rng(0)
        X = rng.uniform(0, np.pi, (40, 4))
        y = rng.integers(2, 5, 40)

        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=1, n_epochs=5, batch_size=10)
        vqc.fit(X[:30], y[:30])
        acc = vqc.score(X[30:], y[30:])
        print(f"VQC smoke-test accuracy: {acc:.3f}")
