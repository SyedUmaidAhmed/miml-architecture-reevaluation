"""Component B: Contextual Encoder — single pre-norm Transformer layer."""

import torch
import torch.nn as nn


class ContextualEncoder(nn.Module):
    """Single-layer pre-norm Transformer encoder for instance-instance interactions.

    No CLS token — label-conditional attention (Component C) replaces it.
    """

    def __init__(self, embed_dim=32, nhead=4, ff_dim=64, dropout=0.3, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        """
        x: (B, N, D) — embedded instances
        mask: (B, N) — True where padded
        Returns: (B, N, D) — contextual instance representations
        """
        if mask is not None:
            mask = mask.bool()
        return self.encoder(x, src_key_padding_mask=mask)
