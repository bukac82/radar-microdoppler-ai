"""
Energy-Based Out-of-Distribution Detection
==========================================
Uses the free energy of DNN logits as an OOD score.
Reference: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    def compute_energy_scores(model: nn.Module, dataloader, temperature: float = 1.0, device: str = "cpu") -> np.ndarray:
        """Compute energy scores for all samples in a DataLoader.

        Energy score: E(x) = -T * log( sum_c exp(f_c(x) / T) )
        Lower energy = more in-distribution. Higher energy = more OOD.

        Args:
            model: Trained classifier.
            dataloader: DataLoader yielding (X, y) batches.
            temperature: Temperature scaling factor.
            device: Compute device.

        Returns:
            Energy scores array of shape (N,). Higher = more OOD.
        """
        import torch
        model.eval()
        scores = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                # Energy: -T * logsumexp(logits / T)
                energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
                scores.append(energy.cpu().numpy())
        return np.concatenate(scores)

    class EnergyOODDetector:
        """Wrapper for energy-based OOD detection with a threshold.

        Args:
            model: Trained DNN.
            threshold: Energy threshold. Samples with energy > threshold are flagged OOD.
                       Tune this on a validation set to achieve target FPR.
            temperature: Temperature for energy computation.
            device: Compute device.
        """

        def __init__(self, model: nn.Module, threshold: float = -25.0, temperature: float = 1.0, device: str = "cpu"):
            self.model = model
            self.threshold = threshold
            self.temperature = temperature
            self.device = device

        def score(self, X_tensor: "torch.Tensor") -> np.ndarray:
            """Compute energy scores for a batch of inputs."""
            import torch
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X_tensor.to(self.device))
                energy = -self.temperature * torch.logsumexp(
                    logits / self.temperature, dim=1
                )
            return energy.cpu().numpy()

        def predict(self, X_tensor: "torch.Tensor") -> np.ndarray:
            """Return boolean array: True = OOD (unknown), False = known class."""
            scores = self.score(X_tensor)
            return scores > self.threshold

        def calibrate_threshold(self, id_scores: np.ndarray, fpr_target: float = 0.05) -> float:
            """Choose threshold to achieve target FPR on in-distribution validation data.

            Args:
                id_scores: Energy scores of in-distribution validation samples.
                fpr_target: Desired false-positive rate (fraction of ID flagged as OOD).

            Returns:
                Threshold value.
            """
            self.threshold = float(np.percentile(id_scores, (1 - fpr_target) * 100))
            print(f"Calibrated threshold: {self.threshold:.4f} (FPR target: {fpr_target})")
            return self.threshold


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed.")
    else:
        import torch
        from Deep_Learning.cnn import MicroDopplerCNN

        model = MicroDopplerCNN(n_classes=3)
        X_id = torch.randn(20, 1, 33, 15)
        X_ood = torch.randn(20, 1, 33, 15) * 5  # high-amplitude OOD

        detector = EnergyOODDetector(model, threshold=-20.0)
        id_pred = detector.predict(X_id)
        ood_pred = detector.predict(X_ood)
        print(f"ID flagged as OOD: {id_pred.sum()}/{len(id_pred)}")
        print(f"OOD flagged as OOD: {ood_pred.sum()}/{len(ood_pred)}")
