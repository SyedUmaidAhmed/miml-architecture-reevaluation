"""Causal Correlation Head (Direction A) -- a drop-in for LabelCorrelationHead.

Differences from the original statistical head:

  1. The coupling matrix ``C`` is a *deconfounded* causal coupling (log odds
     ratio / backdoor adjustment), computed by miml_clam.data.causal_stats and
     installed via ``init_causal``. By default it is a FIXED buffer, not a free
     parameter, so training cannot drift it back toward the spurious
     co-occurrence structure -- this is what keeps the invariance argument
     honest.

  2. The scalar gate ``beta`` is replaced by a PER-LABEL gate ``beta_l``. The
     original scalar over a dense, dilutive statistical matrix parked near 0.1
     (coupling near-inert). A per-label gate over a sparse, informative causal
     matrix lets useful edges actually act while leaving others off.

  3. An OPTIONAL cross-label graph-attention refinement (reused from
     graph_miml) that feature-couples the per-label bag representations, with
     its attention bias initialised from the causal matrix and a regulariser
     (exposed as ``last_coupling_reg``) that anchors the learned label graph to
     the causal prior. This is the Claim-2 (metric-wins) lever and is off by
     default so the causal contribution can be isolated in ablation.

Returns ``(logits, coupled_logits)`` exactly like LabelCorrelationHead, so
MIML_CLAM.forward is unchanged. The coupling regulariser (if any) is stashed on
``self.last_coupling_reg`` for the trainer to pick up.
"""

import torch
import torch.nn as nn


class CausalCorrelationHead(nn.Module):
    def __init__(self, embed_dim, num_labels, dropout=0.2,
                 learnable_C=False, use_graph_coupling=False,
                 graph_reg=0.1, graph_nhead=2, beta_init=0.0):
        super().__init__()
        self.num_labels = num_labels
        self.learnable_C = learnable_C
        self.use_graph_coupling = use_graph_coupling
        self.graph_reg = graph_reg
        hidden = max(embed_dim // 2, 8)

        # Per-label classifiers (identical to LabelCorrelationHead)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for _ in range(num_labels)
        ])

        # Deconfounded causal coupling matrix: fixed buffer by default.
        if learnable_C:
            self.correlation = nn.Parameter(torch.zeros(num_labels, num_labels))
        else:
            self.register_buffer('correlation', torch.zeros(num_labels, num_labels))
        # Frozen copy of the causal prior (regulariser target / provenance).
        self.register_buffer('causal_prior', torch.zeros(num_labels, num_labels))

        # Per-label gate. Default init 0 (coupled == logits at init; grows only
        # where coupling helps); set beta_init=0.1 to match the original head's
        # scalar default for a fair load-bearing comparison.
        self.beta = nn.Parameter(torch.full((num_labels,), float(beta_init)))

        # Optional graph-attention feature coupling (Claim 2).
        if use_graph_coupling:
            from graph_miml.models.label_graph_attention import CrossLabelSelfAttention
            self.graph = CrossLabelSelfAttention(
                embed_dim, num_labels, nhead=graph_nhead, dropout=dropout)

        self.last_coupling_reg = torch.tensor(0.0)

    # ------------------------------------------------------------------ init
    def init_causal(self, causal_matrix):
        """Install the deconfounded causal coupling matrix (L x L).

        causal_matrix is expected already deconfounded + scaled (see
        causal_stats.causal_coupling). Stored as the coupling C and as the
        frozen causal prior; also seeds the graph-attention bias if enabled.
        """
        C = causal_matrix.to(self.correlation.device).float()
        with torch.no_grad():
            self.correlation.copy_(C)
            self.causal_prior.copy_(C)
            if self.use_graph_coupling:
                self.graph.init_from_cooccurrence(C)

    # Back-compat alias so _init_model can treat both heads the same way if
    # desired; here it routes to the causal installer.
    def init_correlation_from_data(self, causal_matrix):
        self.init_causal(causal_matrix)

    # --------------------------------------------------------------- forward
    def forward(self, bag_repr, use_correlation=True):
        """
        bag_repr: (B, L, D) per-label bag representations.
        Returns (logits, coupled_logits), both (B, L).
        """
        B, L, D = bag_repr.shape
        reg = bag_repr.new_zeros(())

        # Optional feature-level graph coupling before classification.
        if self.use_graph_coupling:
            bag_repr = self.graph(bag_repr)                       # (B, L, D)
            # anchor the learned per-head label graph to the causal prior
            learned = self.graph.attn_bias.mean(dim=0)           # (L, L)
            reg = ((learned - self.causal_prior) ** 2).mean()

        # Per-label classification
        logits = torch.cat(
            [self.classifiers[l](bag_repr[:, l, :]) for l in range(L)], dim=1)  # (B, L)

        if use_correlation:
            Cz = torch.matmul(logits, self.correlation.T)        # (B, L)
            gate = self.beta.unsqueeze(0)                        # (1, L) per-label
            coupled_logits = logits + gate * (Cz - logits)
        else:
            coupled_logits = logits

        self.last_coupling_reg = self.graph_reg * reg
        return logits, coupled_logits
