"""
Grad-CAM for CNN Explainability
=================================
Gradient-weighted Class Activation Maps for the MicroDopplerCNN.
Highlights which frequency-time regions in the spectrogram most influenced the prediction.
"""

import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


if HAS_TORCH:
    class GradCAM:
        """Grad-CAM implementation for 2-D CNN models.

        Args:
            model: Trained PyTorch CNN model.
            target_layer: The convolutional layer to hook into (last conv layer recommended).
        """

        def __init__(self, model: nn.Module, target_layer: nn.Module):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self._register_hooks()

        def _register_hooks(self):
            def forward_hook(module, input, output):
                self.activations = output.detach()

            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()

            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)

        def __call__(self, x: "torch.Tensor", class_idx: int = None, device: str = "cpu") -> np.ndarray:
            """Compute Grad-CAM heatmap.

            Args:
                x: Input tensor (1, 1, H, W) — single spectrogram.
                class_idx: Target class index. If None, uses argmax of model output.
                device: Compute device.

            Returns:
                Heatmap array of shape (H, W), values in [0, 1].
            """
            import torch
            self.model.eval()
            x = x.to(device).requires_grad_(True)

            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

            self.model.zero_grad()
            logits[0, class_idx].backward()

            # Global average pool gradients over spatial dims
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
            cam = F.relu(cam)

            # Upsample to input size
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
            return cam

    def overlay_gradcam(spectrogram: np.ndarray, cam: np.ndarray, model_name: str = "cnn", sample_idx: int = 0):
        """Overlay Grad-CAM heatmap on the spectrogram and save."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Spectrogram")
        axes[0].set_xlabel("Time bins")
        axes[0].set_ylabel("Frequency bins")

        axes[1].imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
        axes[1].imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
        axes[1].set_title("Grad-CAM Overlay")
        axes[1].set_xlabel("Time bins")

        plt.tight_layout()
        out = PLOTS_DIR / f"{model_name}_gradcam_{sample_idx}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Grad-CAM overlay saved to {out}")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed.")
    else:
        import torch
        from Deep_Learning.cnn import MicroDopplerCNN

        model = MicroDopplerCNN(n_classes=3)
        cam_extractor = GradCAM(model, target_layer=model.layer3)
        x = torch.randn(1, 1, 33, 15)
        heatmap = cam_extractor(x, class_idx=0)
        print(f"Grad-CAM heatmap shape: {heatmap.shape}")
        overlay_gradcam(x.squeeze().numpy(), heatmap, sample_idx=0)
