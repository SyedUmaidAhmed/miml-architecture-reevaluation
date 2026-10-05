"""Component C: Label-Conditional Gated Attention (KEY INNOVATION).

Learnable label embeddings condition gated attention to produce
label-specific bag representations. Different labels attend to
different instances.
"""

import torch
import torch.nn as nn


class LabelConditionalGatedAttention(nn.Module):
    """Gated attention conditioned on learnable label embeddings.

    For each label l:
      - Concatenate label embedding with each instance: concat(h_i, q_l)
      - Gated attention: score = W_c * (tanh(W_a * concat) * sigmoid(W_b * concat))
      - Softmax over instances -> label-specific bag representation

    Reuses pattern from CLAM's Attn_Net_Gated.
    """

    def __init__(self, embed_dim, num_labels, attn_dim=16, dropout=0.25,
                 use_gated=True):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.use_gated = use_gated

        # Learnable label embeddings
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)

        # Attention layers (shared across labels, differentiated by label embedding)
        concat_dim = embed_dim * 2  # instance + label embedding
        self.attention_a = nn.Sequential(
            nn.Linear(concat_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        if use_gated:
            self.attention_b = nn.Sequential(
                nn.Linear(concat_dim, attn_dim),
                nn.Sigmoid(),
                nn.Dropout(dropout),
            )
        self.attention_c = nn.Linear(attn_dim, 1)

    def forward(self, h, mask=None):
        """
        h: (B, N, D) — contextual instance representations
        mask: (B, N) — True where padded

        Returns:
            bag_repr: (B, L, D) — label-specific bag representations
            attn_weights: (B, L, N) — attention weights per label
        """
        B, N, D = h.shape
        L = self.num_labels

        # Expand label embeddings: (L, D) -> (B, L, 1, D) -> (B, L, N, D)
        q = self.label_embeddings.unsqueeze(0).unsqueeze(2).expand(B, L, N, D)

        # Expand instances: (B, N, D) -> (B, 1, N, D) -> (B, L, N, D)
        h_exp = h.unsqueeze(1).expand(B, L, N, D)

        # Concatenate: (B, L, N, 2D)
        concat = torch.cat([h_exp, q], dim=-1)

        # Attention scores
        a = self.attention_a(concat)      # (B, L, N, attn_dim)
        if self.use_gated:
            b = self.attention_b(concat)  # (B, L, N, attn_dim)
            a = a * b                     # Gated: tanh * sigmoid
        scores = self.attention_c(a).squeeze(-1)  # (B, L, N)

        # Mask padded positions
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).bool(), float('-inf'))

        # Softmax over instances
        attn_weights = torch.softmax(scores, dim=-1)  # (B, L, N)

        # Weighted sum to get label-specific bag representations
        bag_repr = torch.bmm(attn_weights, h)  # (B, L, D)

        return bag_repr, attn_weights
