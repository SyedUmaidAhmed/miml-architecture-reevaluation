"""Neural baselines for fair comparison with MIML-LCGA.

Three models forming an ablation ladder:
  MLP-Pool-ML        → InstanceEmbedding + mean pooling + per-label MLPs
  Transformer-Pool-ML → + ContextualEncoder
  ABMIL-ML           → + single global gated attention + LabelCorrelationHead

All return the same output dict as MIML_CLAM:
  {'logits', 'coupled_logits', 'attn_weights'}
so CurriculumTrainer works unchanged.
"""

import torch
import torch.nn as nn

from .instance_embedding import InstanceEmbedding
from .contextual_encoder import ContextualEncoder
from .label_correlation_head import LabelCorrelationHead


class MLPPoolML(nn.Module):
    """Simplest neural baseline: embed + mean-pool + per-label classifiers.

    No Transformer, no attention, no label correlation.
    """

    def __init__(self, input_dim, num_labels, embed_dim=64, dropout=0.15,
                 embedding_layers=2, **kwargs):
        super().__init__()
        self.num_labels = num_labels

        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers,
        )

        hidden = max(embed_dim // 2, 8)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for _ in range(num_labels)
        ])

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        h = self.instance_embedding(bags)  # (B, N, D)

        # Mask-aware mean pooling
        if mask is not None:
            inv_mask = (~mask.bool()).float().unsqueeze(-1)  # (B, N, 1)
            h_masked = h * inv_mask
            pooled = h_masked.sum(dim=1) / inv_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)  # (B, D)

        # Per-label classification
        logits = torch.cat([clf(pooled) for clf in self.classifiers], dim=1)  # (B, L)

        B, N, _ = bags.shape
        uniform = torch.ones(B, self.num_labels, N, device=bags.device) / N
        if mask is not None:
            uniform = uniform.masked_fill(mask.unsqueeze(1).bool(), 0.0)
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': uniform,
        }


class TransformerPoolML(nn.Module):
    """Embed + Transformer + mean-pool + per-label classifiers.

    Has Transformer but no attention mechanism.
    """

    def __init__(self, input_dim, num_labels, embed_dim=64, nhead=4, ff_dim=128,
                 transformer_layers=1, dropout=0.15, embedding_layers=2, **kwargs):
        super().__init__()
        self.num_labels = num_labels

        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers,
        )
        self.contextual_encoder = ContextualEncoder(
            embed_dim=embed_dim, nhead=nhead, ff_dim=ff_dim,
            dropout=dropout, num_layers=transformer_layers,
        )

        hidden = max(embed_dim // 2, 8)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for _ in range(num_labels)
        ])

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        h = self.instance_embedding(bags)
        h = self.contextual_encoder(h, mask=mask)

        # Mask-aware mean pooling
        if mask is not None:
            inv_mask = (~mask.bool()).float().unsqueeze(-1)
            h_masked = h * inv_mask
            pooled = h_masked.sum(dim=1) / inv_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)

        logits = torch.cat([clf(pooled) for clf in self.classifiers], dim=1)

        B, N, _ = bags.shape
        uniform = torch.ones(B, self.num_labels, N, device=bags.device) / N
        if mask is not None:
            uniform = uniform.masked_fill(mask.unsqueeze(1).bool(), 0.0)
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': uniform,
        }


class ABMILML(nn.Module):
    """ABMIL-style: embed + Transformer + single global gated attention + correlation head.

    Same as MIML-LCGA EXCEPT attention is NOT conditioned on label embeddings.
    All labels share the SAME bag representation (single attention distribution).
    """

    def __init__(self, input_dim, num_labels, embed_dim=64, nhead=4, ff_dim=128,
                 transformer_layers=1, attn_dim=32, dropout=0.15,
                 embedding_layers=2, use_correlation=True,
                 use_gated_attention=True, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.use_correlation = use_correlation

        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers,
        )
        self.contextual_encoder = ContextualEncoder(
            embed_dim=embed_dim, nhead=nhead, ff_dim=ff_dim,
            dropout=dropout, num_layers=transformer_layers,
        )

        # Single global gated attention (NOT label-conditional)
        self.attention_a = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.use_gated = use_gated_attention
        if use_gated_attention:
            self.attention_b = nn.Sequential(
                nn.Linear(embed_dim, attn_dim),
                nn.Sigmoid(),
                nn.Dropout(dropout),
            )
        self.attention_c = nn.Linear(attn_dim, 1)

        # Per-label classifiers + correlation
        self.label_head = LabelCorrelationHead(
            embed_dim=embed_dim, num_labels=num_labels, dropout=dropout,
        )

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        h = self.instance_embedding(bags)          # (B, N, D)
        h = self.contextual_encoder(h, mask=mask)  # (B, N, D)

        # Global gated attention — single distribution for ALL labels
        a = self.attention_a(h)    # (B, N, attn_dim)
        if self.use_gated:
            b = self.attention_b(h)
            a = a * b
        scores = self.attention_c(a).squeeze(-1)  # (B, N)

        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # (B, N)

        # Single bag representation: (B, D)
        bag_repr = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, D)

        # Expand to (B, L, D) — same representation for every label
        bag_repr_expanded = bag_repr.unsqueeze(1).expand(-1, self.num_labels, -1)

        logits, coupled_logits = self.label_head(
            bag_repr_expanded, use_correlation=self.use_correlation)

        # Expand attention to (B, L, N) — same for all labels
        attn_expanded = attn_weights.unsqueeze(1).expand(-1, self.num_labels, -1)

        return {
            'logits': logits,
            'coupled_logits': coupled_logits,
            'attn_weights': attn_expanded,
        }
