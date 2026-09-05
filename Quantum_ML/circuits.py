"""
Reusable PennyLane Circuit Building Blocks
==========================================
Angle embedding, strongly entangling layers, and measurement helpers.
"""

try:
    import pennylane as qml
    import pennylane.numpy as pnp
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("[WARNING] PennyLane not installed. Install via: pip install pennylane pennylane-lightning")

import numpy as np


def get_device(n_qubits: int, backend: str = "lightning.qubit"):
    """Return a PennyLane device.

    Args:
        n_qubits: Number of qubits.
        backend: PennyLane device name. 'lightning.qubit' is fastest for CPU simulation.

    Returns:
        pennylane.Device
    """
    if not HAS_PENNYLANE:
        raise ImportError("PennyLane is required. Run: pip install pennylane pennylane-lightning")
    try:
        return qml.device(backend, wires=n_qubits)
    except Exception:
        # Fallback to default.qubit if lightning not available
        return qml.device("default.qubit", wires=n_qubits)


def angle_embedding(x: np.ndarray, n_qubits: int, rotation: str = "RY"):
    """Encode a feature vector into qubit rotations (angle embedding).

    Args:
        x: Feature vector of length n_qubits.
        n_qubits: Number of qubits (must equal len(x)).
        rotation: Gate type ('RX', 'RY', 'RZ').
    """
    if not HAS_PENNYLANE:
        raise ImportError("PennyLane required.")
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation=rotation)


def strongly_entangling_layer(weights: np.ndarray, n_qubits: int, n_layers: int):
    """Apply StronglyEntanglingLayers with trainable weights.

    Args:
        weights: Shape (n_layers, n_qubits, 3).
        n_qubits: Number of qubits.
        n_layers: Number of layers.
    """
    if not HAS_PENNYLANE:
        raise ImportError("PennyLane required.")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))


def iqp_embedding(x: np.ndarray, n_qubits: int):
    """IQP-style feature map: Hadamard + ZZ interactions (used for quantum kernel)."""
    if not HAS_PENNYLANE:
        raise ImportError("PennyLane required.")
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
