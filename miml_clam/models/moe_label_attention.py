"""Mixture-of-experts label-conditional gated attention.

The diagnosis this is built from
--------------------------------
Measured on the 13 benchmarks, single-model versus single-model, MIML-LCGA's
win rate against the neural baselines trends downward with label count
(Pearson r=-0.50, p=0.085 -- suggestive, not significant at p<0.05; see
scripts/compute_moe_win_counts.py, which is the authoritative source for this
number): it takes 5/5 metrics on Scene (L=5) and 4/5 on Reuters (L=7), and
then collapses to 0-2/5 on every dataset with L >= 14. An earlier version of
this docstring quoted r=-0.91, from a label-count lookup that had the wrong
L for 3 of 13 datasets (Protein Haloarcula/Geobacter/Azotobacter) -- see
paper_tmlr/GAPS.md item G2 for the correction.

The mechanism is the shared attention map. In `LabelConditionalGatedAttention`
the parameters W_a, W_b and w_c are shared across all labels, and label-specific
behaviour has to emerge entirely from the label embedding q_l entering a
concatenation. One fixed-capacity map must therefore serve every label's
selection pattern at once. At L=5 that is easy. At L=52 the map is being asked
to encode 52 different notions of "which instances matter", and it saturates.

The manuscript claims this sharing as a virtue (attention parameters do not grow
with L). The data says it is a liability precisely where the claimed saving is
largest.

The technique
-------------
Interpolate between the two extremes instead of picking one.

    shared (LCGA)          K = 1 expert           saturates as L grows
    this module            1 < K < L experts      capacity grows with K, not L
    per-label (CLAM)       K = L experts          O(L) parameters, overfits small data

Each expert is a full gated-attention map. A label is routed across experts by a
softmax gate computed from its own embedding, so labels that want similar
instance-selection behaviour can share an expert while dissimilar labels can pull
apart. Routing is learned, not clustered in advance.

Setting `num_experts=1` recovers the original module exactly (the gate is a
softmax over one logit, hence identically 1), which keeps the ablation honest:
any gain must come from K > 1 and nothing else.

Parameter cost is K x the single-map cost, independent of L. So on Protein
Pyrococcus (L=52), K=4 costs 4/52 of what per-label branches would.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELabelConditionalGatedAttention(nn.Module):
    """Gated attention with K expert maps and learned per-label routing.

    Returns the same contract as LabelConditionalGatedAttention:
        bag_reprs   [B, L, D]
        attn        [B, L, N]
    """

    def __init__(self, embed_dim, num_labels, attn_dim=32, num_experts=4,
                 use_gated=True, dropout=0.0, router_temp=1.0,
                 load_balance_weight=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.attn_dim = attn_dim
        self.num_experts = num_experts
        self.use_gated = use_gated
        self.router_temp = router_temp
        self.load_balance_weight = load_balance_weight

        # Label embeddings, as in the original module.
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)

        # K expert attention maps. Each is the original (W_a, W_b, w_c) triple,
        # consuming the [h ; q] concatenation. Held as single batched tensors so
        # every expert is evaluated in one matmul.
        # Biases are present so that K=1 reproduces LabelConditionalGatedAttention
        # exactly: that module uses nn.Linear (with bias) for the a and b branches,
        # and those biases sit inside tanh/sigmoid so they are not absorbable.
        # (The w_c bias is omitted deliberately — a constant added to every
        # instance score cancels in the softmax.)
        cat_dim = 2 * embed_dim
        self.W_a = nn.Parameter(torch.empty(num_experts, cat_dim, attn_dim))
        self.W_b = nn.Parameter(torch.empty(num_experts, cat_dim, attn_dim))
        self.b_a = nn.Parameter(torch.zeros(num_experts, attn_dim))
        self.b_b = nn.Parameter(torch.zeros(num_experts, attn_dim))
        self.w_c = nn.Parameter(torch.empty(num_experts, attn_dim))
        for p in (self.W_a, self.W_b):
            nn.init.xavier_uniform_(p)
        nn.init.normal_(self.w_c, std=0.02)

        # Router: label embedding -> distribution over experts.
        self.router = nn.Linear(embed_dim, num_experts)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._last_router_probs = None

    def routing_probs(self):
        """[L, K] soft assignment of labels to experts."""
        return F.softmax(self.router(self.label_embeddings) / self.router_temp, dim=-1)

    def load_balance_loss(self):
        """Discourage collapse onto a single expert.

        Without this the router can send every label to one expert, which
        silently reduces the module to the original shared-attention design --
        exactly the failure mode we are trying to escape. Penalising the squared
        deviation of mean expert usage from uniform keeps all K in play.
        """
        if self._last_router_probs is None or self.num_experts == 1:
            return torch.zeros((), device=self.label_embeddings.device)
        usage = self._last_router_probs.mean(dim=0)          # [K]
        uniform = 1.0 / self.num_experts
        return self.load_balance_weight * ((usage - uniform) ** 2).sum() * self.num_experts

    def forward(self, H, mask=None):
        """H: [B, N, D]; mask: [B, N] — **True where PADDED**.

        This matches `LabelConditionalGatedAttention` and the collate function
        (`mask: 1=padded, 0=real`). The polarity is the opposite of the usual
        "True = keep" convention, so it is masked_fill(mask, -inf), not ~mask.
        """
        B, N, D = H.shape
        L, K, A = self.num_labels, self.num_experts, self.attn_dim

        R = self.routing_probs()                              # [L, K]
        self._last_router_probs = R

        # Concatenate every instance with every label embedding: [B, L, N, 2D]
        h = H.unsqueeze(1).expand(B, L, N, D)
        q = self.label_embeddings.view(1, L, 1, D).expand(B, L, N, D)
        cat = torch.cat([h, q], dim=-1)

        # Every expert scores every (label, instance) pair: [B, L, N, K, A]
        a = torch.einsum('blnc,kca->blnka', cat, self.W_a) + self.b_a
        if self.use_gated:
            b = torch.einsum('blnc,kca->blnka', cat, self.W_b) + self.b_b
            gated = torch.tanh(a) * torch.sigmoid(b)
        else:
            gated = torch.tanh(a)
        gated = self.dropout(gated)

        # Per-expert scalar scores, then mix by the label's routing weights.
        scores_k = torch.einsum('blnka,ka->blnk', gated, self.w_c)   # [B, L, N, K]
        scores = torch.einsum('blnk,lk->bln', scores_k, R)           # [B, L, N]

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)   # a fully-masked bag yields zeros
        bag_reprs = torch.einsum('bln,bnd->bld', attn, H)
        return bag_reprs, attn


def expert_parameter_count(embed_dim, attn_dim, num_experts, num_labels):
    """Attention-block parameters, for the accuracy-per-parameter comparison.

    The manuscript's Remark 2 argues parameter efficiency arithmetically and
    never measures accuracy against a per-label design. These are the three
    points needed to turn that into an empirical trade-off curve.
    """
    per_map = (2 * embed_dim) * attn_dim * 2 + attn_dim
    return {
        'shared_lcga': per_map,
        'moe': per_map * num_experts + embed_dim * num_experts + num_experts,
        'per_label': per_map * num_labels,
    }
