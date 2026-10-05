"""MIML-CLAM++: Multi-Instance Multi-Label CLAM.

Full model wiring: Instance Embedding -> Contextual Encoder ->
Label-Conditional Gated Attention -> Label Correlation Head + Instance Clustering.

Architecture:
  A. Instance Embedding: MLP with LayerNorm + GELU + residual
  B. Contextual Encoder: Pre-norm Transformer (instance-instance interactions)
  C. Label-Conditional Gated Attention: Different labels attend to different instances
  D. Label Correlation Head: Per-label classifiers + correlation coupling
  E. Instance Clustering Loss: CLAM-adapted pseudo instance-level supervision
"""

import torch
import torch.nn as nn

from .instance_embedding import InstanceEmbedding
from .contextual_encoder import ContextualEncoder
from .label_conditional_attn import LabelConditionalGatedAttention
from .wsi_components import EfficientLabelConditionalGatedAttention
from .moe_label_attention import MoELabelConditionalGatedAttention
from .label_correlation_head import LabelCorrelationHead
from .causal_correlation_head import CausalCorrelationHead
from .instance_clustering import InstanceClusteringLoss


class MIML_CLAM(nn.Module):
    """Multi-Instance Multi-Label CLAM++ model.

    Shape flow (Scene): [B,9,15] -> embed [B,9,64] -> transformer [B,9,64]
                        -> attention [B,6,9] -> bag_repr [B,6,64] -> logits [B,6]
    """

    def __init__(
        self,
        input_dim,
        num_labels,
        embed_dim=64,
        nhead=4,
        ff_dim=128,
        transformer_layers=1,
        attn_dim=32,
        dropout=0.15,
        k_sample=2,
        use_transformer=True,
        use_correlation=True,
        use_instance_clustering=False,
        use_gated_attention=True,
        embedding_layers=2,
        use_efficient_attn=False,
        num_attn_experts=1,
        moe_load_balance=0.01,
        label_chunk=8,
        use_causal=False,
        causal_learnable_C=False,
        causal_graph_coupling=False,
        causal_graph_reg=0.1,
        causal_beta_init=0.0,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_transformer = use_transformer
        self.use_correlation = use_correlation
        self.use_instance_clustering = use_instance_clustering
        self.use_causal = use_causal

        # Component A: Instance Embedding
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            num_layers=embedding_layers,
        )

        # Component B: Contextual Encoder
        if use_transformer:
            self.contextual_encoder = ContextualEncoder(
                embed_dim=embed_dim,
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout,
                num_layers=transformer_layers,
            )

        # Component C: Label-Conditional Gated Attention.
        # The efficient variant is an exact, memory-bounded reformulation for
        # large-N (WSI) bags — identical parameters, interchangeable state_dict.
        # The mixture-of-experts variant addresses a measured failure: win rate
        # against the neural baselines correlates about -0.91 with label count,
        # because one shared attention map has to serve every label's selection
        # pattern. K experts with learned per-label routing give capacity that
        # grows with K rather than L. num_attn_experts=1 reduces exactly to the
        # shared design, so the comparison isolates K.
        if num_attn_experts and num_attn_experts > 1:
            self.label_attention = MoELabelConditionalGatedAttention(
                embed_dim=embed_dim,
                num_labels=num_labels,
                attn_dim=attn_dim,
                num_experts=num_attn_experts,
                dropout=dropout,
                use_gated=use_gated_attention,
                load_balance_weight=moe_load_balance,
            )
        elif use_efficient_attn:
            self.label_attention = EfficientLabelConditionalGatedAttention(
                embed_dim=embed_dim,
                num_labels=num_labels,
                attn_dim=attn_dim,
                dropout=dropout,
                use_gated=use_gated_attention,
                label_chunk=label_chunk,
            )
        else:
            self.label_attention = LabelConditionalGatedAttention(
                embed_dim=embed_dim,
                num_labels=num_labels,
                attn_dim=attn_dim,
                dropout=dropout,
                use_gated=use_gated_attention,
            )

        # Component D: Label Correlation Head.
        # The causal variant (Direction A) is a drop-in with a deconfounded,
        # prior-shift-invariant coupling matrix and a per-label gate.
        if use_causal:
            self.label_head = CausalCorrelationHead(
                embed_dim=embed_dim,
                num_labels=num_labels,
                dropout=dropout,
                learnable_C=causal_learnable_C,
                use_graph_coupling=causal_graph_coupling,
                graph_reg=causal_graph_reg,
                beta_init=causal_beta_init,
            )
        else:
            self.label_head = LabelCorrelationHead(
                embed_dim=embed_dim,
                num_labels=num_labels,
                dropout=dropout,
            )

        # Component E: Instance Clustering Loss
        if use_instance_clustering:
            self.instance_clustering = InstanceClusteringLoss(
                embed_dim=embed_dim,
                num_labels=num_labels,
                k_sample=k_sample,
            )

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        """
        Args:
            bags: (B, N, input_dim) — padded bags of instances
            mask: (B, N) — True where padded
            bag_labels: (B, L) — bag-level labels (for instance clustering)
            compute_instance_loss: bool — whether to compute instance clustering loss

        Returns dict with:
            logits: (B, L) — raw logits
            coupled_logits: (B, L) — correlation-coupled logits
            attn_weights: (B, L, N) — attention weights per label
            instance_loss: scalar (if compute_instance_loss)
            instance_acc: float (if compute_instance_loss)
        """
        # A: Instance Embedding
        h = self.instance_embedding(bags)  # (B, N, D)

        # B: Contextual Encoder
        if self.use_transformer:
            h = self.contextual_encoder(h, mask=mask)  # (B, N, D)

        # C: Label-Conditional Gated Attention
        bag_repr, attn_weights = self.label_attention(h, mask=mask)  # (B, L, D), (B, L, N)

        # D: Label Correlation Head
        logits, coupled_logits = self.label_head(
            bag_repr, use_correlation=self.use_correlation
        )

        result = {
            'logits': logits,
            'coupled_logits': coupled_logits,
            'attn_weights': attn_weights,
        }

        # Causal head exposes an optional coupling regulariser (graph-anchor term)
        if self.use_causal and getattr(self.label_head, 'use_graph_coupling', False):
            result['coupling_reg'] = self.label_head.last_coupling_reg

        # E: Instance Clustering Loss
        if compute_instance_loss and self.use_instance_clustering and bag_labels is not None:
            inst_loss, inst_acc = self.instance_clustering(
                h, attn_weights.detach(), bag_labels, mask=mask
            )
            result['instance_loss'] = inst_loss
            result['instance_acc'] = inst_acc

        return result

    def get_attention_maps(self, bags, mask=None):
        """Get attention maps for visualization (no grad)."""
        with torch.no_grad():
            result = self.forward(bags, mask=mask)
        return result['attn_weights']
