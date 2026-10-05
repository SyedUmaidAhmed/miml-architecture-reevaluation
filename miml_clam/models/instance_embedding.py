"""Component A: Instance Embedding with LayerNorm + GELU + residual."""

import torch
import torch.nn as nn


class InstanceEmbedding(nn.Module):
    """Embeds raw instance features into a shared latent space.

    Uses LayerNorm (not BatchNorm) for stability with batch_size=1.
    Residual connection when input_dim == embed_dim.
    """

    def __init__(self, input_dim, embed_dim, dropout=0.3, num_layers=2):
        super().__init__()
        self.use_residual = (input_dim == embed_dim) and (num_layers > 1)

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else embed_dim
            layers.extend([
                nn.Linear(in_d, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

        # Projection for residual when dims don't match
        if not self.use_residual and num_layers > 1:
            self.proj = nn.Linear(input_dim, embed_dim)
            self.use_residual = True
        else:
            self.proj = None

    def forward(self, x):
        """x: (B, N, input_dim) -> (B, N, embed_dim)"""
        out = self.net(x)
        if self.use_residual:
            residual = self.proj(x) if self.proj is not None else x
            out = out + residual
        return out
