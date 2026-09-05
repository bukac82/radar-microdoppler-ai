"""
Adversarial Robustness Testing
================================
FGSM and PGD adversarial attacks on PyTorch deep learning models.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    def fgsm_attack(
        model: nn.Module,
        X: "torch.Tensor",
        y: "torch.Tensor",
        epsilon: float = 0.01,
        device: str = "cpu",
    ) -> "torch.Tensor":
        """Fast Gradient Sign Method (FGSM) adversarial perturbation.

        Args:
            model: Trained PyTorch model.
            X: Input tensor (B, ...).
            y: True labels (B,).
            epsilon: Perturbation magnitude.
            device: Compute device.

        Returns:
            Adversarial examples tensor of same shape as X.
        """
        import torch
        model.eval()
        X = X.to(device).requires_grad_(True)
        y = y.to(device)
        criterion = nn.CrossEntropyLoss()

        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        X_adv = X + epsilon * X.grad.sign()
        return X_adv.detach()

    def pgd_attack(
        model: nn.Module,
        X: "torch.Tensor",
        y: "torch.Tensor",
        epsilon: float = 0.01,
        alpha: float = 0.002,
        n_steps: int = 10,
        device: str = "cpu",
    ) -> "torch.Tensor":
        """Projected Gradient Descent (PGD) adversarial attack.

        Args:
            model: Trained PyTorch model.
            X: Input tensor (B, ...).
            y: True labels (B,).
            epsilon: L-inf perturbation budget.
            alpha: Step size per iteration.
            n_steps: Number of PGD steps.
            device: Compute device.

        Returns:
            Adversarial examples tensor.
        """
        import torch
        model.eval()
        X_orig = X.clone().to(device)
        X_adv = X.clone().to(device)
        y = y.to(device)
        criterion = nn.CrossEntropyLoss()

        for _ in range(n_steps):
            X_adv.requires_grad_(True)
            logits = model(X_adv)
            loss = criterion(logits, y)
            loss.backward()

            with torch.no_grad():
                X_adv = X_adv + alpha * X_adv.grad.sign()
                # Project back into epsilon-ball
                delta = torch.clamp(X_adv - X_orig, -epsilon, epsilon)
                X_adv = (X_orig + delta).detach()

        return X_adv

    def evaluate_adversarial(
        model: nn.Module,
        X_clean: "torch.Tensor",
        y: "torch.Tensor",
        epsilon_values: list = [0.001, 0.005, 0.01, 0.05],
        attack: str = "fgsm",
        device: str = "cpu",
    ) -> dict:
        """Evaluate adversarial robustness across multiple epsilon values.

        Returns:
            Dict mapping epsilon → accuracy under attack.
        """
        import torch
        model.eval()
        results = {}
        for eps in epsilon_values:
            if attack == "fgsm":
                X_adv = fgsm_attack(model, X_clean, y, epsilon=eps, device=device)
            elif attack == "pgd":
                X_adv = pgd_attack(model, X_clean, y, epsilon=eps, device=device)
            else:
                raise ValueError(f"Unknown attack: {attack}")

            with torch.no_grad():
                logits = model(X_adv.to(device))
                preds = logits.argmax(dim=1)
                acc = (preds == y.to(device)).float().mean().item()
            results[eps] = acc
            print(f"  {attack.upper()} eps={eps:.4f}  →  acc={acc:.4f}")
        return results


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed.")
    else:
        import torch
        from Deep_Learning.cnn import MicroDopplerCNN

        model = MicroDopplerCNN(n_classes=3)
        X = torch.randn(16, 1, 33, 15)
        y = torch.randint(0, 3, (16,))

        results = evaluate_adversarial(model, X, y, epsilon_values=[0.01, 0.05], attack="fgsm")
        print("FGSM adversarial results:", results)
