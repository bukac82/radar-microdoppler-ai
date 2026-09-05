"""
Spectrogram Transformer (ViT-inspired)
========================================
Splits a micro-Doppler spectrogram into patches and applies multi-head
self-attention for classification.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class PatchEmbedding(nn.Module):
        """Split spectrogram into patches and project to embedding dim."""

        def __init__(self, img_h: int, img_w: int, patch_h: int, patch_w: int, embed_dim: int):
            super().__init__()
            assert img_h % patch_h == 0 and img_w % patch_w == 0
            self.n_patches = (img_h // patch_h) * (img_w // patch_w)
            self.proj = nn.Conv2d(1, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

        def forward(self, x):
            # x: (B, 1, H, W) → (B, n_patches, embed_dim)
            x = self.proj(x)          # (B, embed_dim, nh, nw)
            x = x.flatten(2)          # (B, embed_dim, n_patches)
            x = x.transpose(1, 2)     # (B, n_patches, embed_dim)
            return x

    class MicroDopplerTransformer(nn.Module):
        """Vision-Transformer-style classifier for micro-Doppler spectrograms.

        Args:
            img_h, img_w: Spectrogram height and width.
            patch_h, patch_w: Patch size.
            embed_dim: Token embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer encoder layers.
            n_classes: Output classes.
            dropout: Dropout rate.
        """

        def __init__(
            self,
            img_h: int = 33,
            img_w: int = 15,
            patch_h: int = 11,
            patch_w: int = 5,
            embed_dim: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            n_classes: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.patch_embed = PatchEmbedding(img_h, img_w, patch_h, patch_w, embed_dim)
            n_patches = self.patch_embed.n_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, n_classes)

        def forward(self, x):
            """Args: x: (B, 1, H, W). Returns: logits (B, n_classes)."""
            B = x.shape[0]
            x = self.patch_embed(x)  # (B, n_patches, embed_dim)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, embed_dim)
            x = x + self.pos_embed
            x = self.transformer(x)
            x = self.norm(x[:, 0])   # CLS token
            return self.head(x)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not installed.")
    else:
        import torch
        model = MicroDopplerTransformer()
        x = torch.randn(4, 1, 33, 15)
        logits = model(x)
        print(f"Transformer output shape: {logits.shape}")  # (4, 3)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
