"""
Unified Deep Learning Training Loop
=====================================
Trains any PyTorch model (CNN, LSTM, Transformer) with early stopping,
LR scheduling, and checkpoint saving.
"""

from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


if HAS_TORCH:
    def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        model_name: str = "model",
        device: str = "cpu",
    ) -> dict:
        """Train a PyTorch model with early stopping and cosine LR schedule.

        Args:
            model: PyTorch model (CNN / LSTM / Transformer).
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            n_epochs: Maximum training epochs.
            lr: Initial learning rate.
            patience: Early stopping patience (epochs without val improvement).
            model_name: Prefix for checkpoint file.
            device: 'cpu', 'cuda', or 'mps'.

        Returns:
            Dict with 'train_loss', 'val_loss', 'val_acc' history lists.
        """
        device = torch.device(device)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_acc = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"

        for epoch in range(1, n_epochs + 1):
            # --- Train ---
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
            train_loss /= len(train_loader.dataset)

            # --- Validate ---
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * len(y_batch)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += len(y_batch)
            val_loss /= total
            val_acc = correct / total

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}. Best val_acc={best_val_acc:.4f}")
                    break

        # Restore best weights
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Best model saved at {ckpt_path}")
        return history


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed.")
    else:
        from Deep_Learning.cnn import MicroDopplerCNN
        from Deep_Learning.dataset_torch import MicroDopplerDataset, get_dataloader, spectrogram_transform

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 500)) + 1j * rng.standard_normal((200, 500))
        y = rng.integers(2, 5, 200)

        ds = MicroDopplerDataset(X, y, transform=spectrogram_transform)
        train_dl = get_dataloader(ds, batch_size=16, shuffle=True)
        val_dl = get_dataloader(ds, batch_size=16, shuffle=False)

        model = MicroDopplerCNN(n_classes=3)
        history = train_model(model, train_dl, val_dl, n_epochs=3, model_name="cnn_smoke")
        print(f"Smoke test complete. Final val_acc={history['val_acc'][-1]:.4f}")
