"""Component D: Label Correlation Head.

Per-label MLP classifiers + learnable correlation matrix for
inter-label coupling.
"""

import torch
import torch.nn as nn


class LabelCorrelationHead(nn.Module):
    """Per-label classifiers with label correlation coupling.

    Each label gets its own small MLP: embed_dim -> embed_dim//2 -> 1
    A learnable correlation matrix couples logits across labels:
      coupled_logits = z + beta * (C @ z - z)
    """

    def __init__(self, embed_dim, num_labels, dropout=0.2):
        super().__init__()
        self.num_labels = num_labels
        hidden = max(embed_dim // 2, 8)

        # Per-label classifiers
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for _ in range(num_labels)
        ])

        # Learnable correlation matrix (initialized later from data)
        self.correlation = nn.Parameter(torch.zeros(num_labels, num_labels))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def init_correlation_from_data(self, co_matrix):
        """Initialize correlation matrix from label co-occurrence statistics."""
        with torch.no_grad():
            self.correlation.copy_(co_matrix)

    def forward(self, bag_repr, use_correlation=True):
        """
        bag_repr: (B, L, D) — label-specific bag representations

        Returns:
            logits: (B, L) — raw logits
            coupled_logits: (B, L) — correlation-coupled logits
        """
        B, L, D = bag_repr.shape

        # Per-label classification
        logits_list = []
        for l in range(L):
            logit = self.classifiers[l](bag_repr[:, l, :])  # (B, 1)
            logits_list.append(logit)
        logits = torch.cat(logits_list, dim=1)  # (B, L)

        if use_correlation:
            # Linear coupling through correlation matrix
            # coupled = z + beta * (C @ z - z)  (matches eq:coupling in paper)
            Cz = torch.matmul(logits, self.correlation.T)  # (B, L)
            coupled_logits = logits + self.beta * (Cz - logits)
        else:
            coupled_logits = logits

        return logits, coupled_logits
