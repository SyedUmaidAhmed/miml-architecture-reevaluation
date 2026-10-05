"""WSI-MIL baselines adapted to the multi-instance multi-label (MIML) setting.

The canonical whole-slide-image MIL methods are single-label (binary / multi-
class). Here they are adapted to multi-label slide prediction by replacing the
head with independent per-label classifiers, and made to return the shared
output dict ``{'logits', 'coupled_logits', 'attn_weights'}`` so ``CurriculumTrainer``
and ``run_cv`` work unchanged. Baselines without a correlation head set
``coupled_logits = logits``; those without per-label attention emit a uniform
``attn_weights`` (same convention as ``baselines.MLPPoolML``).

  ABMIL (gated-attention MIL)  -> use ``baselines.ABMILML`` (already in the repo).
  TransMIL                     -> ``TransMILMultiLabel``  (this file).
  DSMIL  (dual-stream MIL)     -> ``DSMILMultiLabel``     (this file).
  CLAM   (clustering-att. MIL) -> ``CLAMMultiLabel``      (this file).
  DTFD-MIL (double-tier)       -> ``DTFDMultiLabel``      (this file).
  M4 (multi-task MoE)          -> ``M4MultiLabel``        (this file).
  Double-Tier Attention ML     -> ``DoubleTierMultiLabel``(this file).

These are faithful-to-the-core-idea adaptations sized for the WSI scaffold;
for a camera-ready table they can be swapped for the official implementations
(see github.com/lingxitong/MIL_BASELINE) behind the same interface.
"""

import torch
import torch.nn as nn

from .instance_embedding import InstanceEmbedding
from .instance_clustering import InstanceClusteringLoss


def _uniform_attn(B, L, N, mask, device):
    """Uniform per-label attention over real instances (interface filler)."""
    attn = torch.ones(B, L, N, device=device)
    if mask is not None:
        attn = attn.masked_fill(mask.unsqueeze(1).bool(), 0.0)
    return attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)


class TransMILMultiLabel(nn.Module):
    """TransMIL-style baseline: class-token Transformer aggregation, multi-label.

    Captures TransMIL's core idea — self-attention as "correlated MIL" plus a
    class-token readout. The PPEG positional conv and Nyström attention of the
    original are omitted (PPEG needs a square patch grid; padding-masked SDPA
    keeps memory linear in N regardless).
    """

    def __init__(self, input_dim, num_labels, embed_dim=192, nhead=6, ff_dim=384,
                 transformer_layers=2, dropout=0.25, embedding_layers=2, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(embed_dim)

        hidden = max(embed_dim // 2, 8)
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden, 1))
            for _ in range(num_labels)])

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        B, N, _ = bags.shape
        h = self.instance_embedding(bags)                       # (B, N, D)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], 1)  # (B, N+1, D)

        enc_mask = None
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=mask.dtype, device=mask.device)
            enc_mask = torch.cat([cls_mask, mask], dim=1).bool()

        h = self.encoder(h, src_key_padding_mask=enc_mask)
        pooled = self.norm(h[:, 0])                              # CLS token (B, D)
        logits = torch.cat([clf(pooled) for clf in self.classifiers], dim=1)

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': _uniform_attn(B, self.num_labels, N, mask, bags.device),
        }


class DSMILMultiLabel(nn.Module):
    """DSMIL-style dual-stream baseline, multi-label.

    Stream 1 (instance): a per-patch classifier; the top-scoring "critical"
    patch per label gives a max-pooled score. Stream 2 (bag): every patch
    attends to that label's critical patch, producing an attention-pooled
    score. The bag logit is the mean of the two streams.
    """

    def __init__(self, input_dim, num_labels, embed_dim=192, dropout=0.25,
                 embedding_layers=2, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers)
        self.inst_classifier = nn.Linear(embed_dim, num_labels)   # instance stream
        self.q = nn.Linear(embed_dim, embed_dim)                  # query projection
        self.v = nn.Linear(embed_dim, embed_dim)                  # value projection
        self.dropout = nn.Dropout(dropout)
        self.bag_classifier = nn.Linear(embed_dim, num_labels)    # bag stream

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        B, N, _ = bags.shape
        L, D = self.num_labels, self.embed_dim
        h = self.instance_embedding(bags)                         # (B, N, D)

        inst_logits = self.inst_classifier(h)                     # (B, N, L)
        if mask is not None:
            inst_logits = inst_logits.masked_fill(
                mask.unsqueeze(-1).bool(), float('-inf'))
        max_inst, crit_idx = inst_logits.max(dim=1)               # (B, L), (B, L)

        Q = self.q(h)                                             # (B, N, D)
        V = self.dropout(self.v(h))                               # (B, N, D)
        # Critical-patch query per label — gather without an (B,L,N,D) blow-up.
        crit_q = torch.gather(Q, 1, crit_idx.unsqueeze(-1).expand(B, L, D))  # (B,L,D)

        attn_scores = torch.einsum('bld,bnd->bln', crit_q, Q) / (D ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(1).bool(), float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)                 # (B, L, N)
        bag_repr = torch.bmm(attn, V)                             # (B, L, D)

        bag_logits = (torch.einsum('bld,ld->bl', bag_repr, self.bag_classifier.weight)
                      + self.bag_classifier.bias)                 # (B, L)
        logits = 0.5 * (max_inst + bag_logits)

        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': attn,
        }


class _GatedAttnPool(nn.Module):
    """One gated-attention pooling head: (B, N, D) + mask -> pooled (B, D), attn (B, N).

    Gated attention (tanh * sigmoid) as in ABMIL. A row whose instances are all
    padded pools to a zero vector instead of NaN.
    """

    def __init__(self, embed_dim, attn_dim=128, dropout=0.25):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh(),
                               nn.Dropout(dropout))
        self.b = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid(),
                               nn.Dropout(dropout))
        self.c = nn.Linear(attn_dim, 1)

    def forward(self, h, mask=None):
        scores = self.c(self.a(h) * self.b(h)).squeeze(-1)         # (B, N)
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float('-inf'))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1))     # all-padded -> 0
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)        # (B, D)
        return pooled, attn


class CLAMMultiLabel(nn.Module):
    """CLAM-style baseline, multi-label.

    Keeps CLAM's two signatures: one gated-attention branch per label (CLAM's
    per-class attention branches) and CLAM's instance-level clustering auxiliary
    supervision (top-/bottom-attended instances as pseudo-labels). Instance
    clustering is intrinsic to CLAM and is always on; the trainer adds its loss
    in Phase 2 through the standard ``compute_instance_loss`` path.
    """

    def __init__(self, input_dim, num_labels, embed_dim=192, attn_dim=128,
                 dropout=0.25, embedding_layers=2, k_sample=8, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.use_instance_clustering = True            # CLAM always uses it
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers)
        self.branches = nn.ModuleList(
            [_GatedAttnPool(embed_dim, attn_dim, dropout) for _ in range(num_labels)])
        self.classifiers = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(num_labels)])
        self.instance_clustering = InstanceClusteringLoss(
            embed_dim=embed_dim, num_labels=num_labels, k_sample=k_sample)

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        h = self.instance_embedding(bags)                          # (B, N, D)
        pooled, attns = [], []
        for l in range(self.num_labels):
            p, a = self.branches[l](h, mask=mask)
            pooled.append(p)
            attns.append(a)
        logits = torch.cat(
            [self.classifiers[l](pooled[l]) for l in range(self.num_labels)], dim=1)
        attn_weights = torch.stack(attns, dim=1)                   # (B, L, N)

        result = {'logits': logits, 'coupled_logits': logits,
                  'attn_weights': attn_weights}
        if compute_instance_loss and bag_labels is not None:
            inst_loss, inst_acc = self.instance_clustering(
                h, attn_weights.detach(), bag_labels, mask=mask)
            result['instance_loss'] = inst_loss
            result['instance_acc'] = inst_acc
        return result


class DTFDMultiLabel(nn.Module):
    """DTFD-MIL-style baseline, multi-label.

    Instances are randomly partitioned into ``num_pseudo_bags`` pseudo-bags; a
    tier-1 gated-attention head pools each pseudo-bag, and a tier-2 head pools
    the resulting pseudo-bag features into the final bag representation.
    """

    def __init__(self, input_dim, num_labels, embed_dim=192, attn_dim=128,
                 dropout=0.25, embedding_layers=2, num_pseudo_bags=4, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.num_pseudo_bags = num_pseudo_bags
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers)
        self.tier1 = _GatedAttnPool(embed_dim, attn_dim, dropout)   # patches -> pseudo-bag
        self.tier2 = _GatedAttnPool(embed_dim, attn_dim, dropout)   # pseudo-bags -> bag
        hidden = max(embed_dim // 2, 8)
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden, 1))
            for _ in range(num_labels)])

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        B, N, _ = bags.shape
        h = self.instance_embedding(bags)                          # (B, N, D)
        if mask is None:
            mask = torch.zeros(B, N, dtype=torch.bool, device=bags.device)
        mask = mask.bool()

        # Random pseudo-bag partition (one permutation shared across the batch).
        perm = torch.randperm(N, device=bags.device)
        chunks_h = torch.chunk(h[:, perm, :], self.num_pseudo_bags, dim=1)
        chunks_m = torch.chunk(mask[:, perm], self.num_pseudo_bags, dim=1)

        pb_feats, pb_mask = [], []
        for ch_h, ch_m in zip(chunks_h, chunks_m):
            feat, _ = self.tier1(ch_h, mask=ch_m)                  # (B, D)
            pb_feats.append(feat)
            pb_mask.append(ch_m.all(dim=1))                        # (B,) all-padded?
        pb = torch.stack(pb_feats, dim=1)                          # (B, M, D)
        pbm = torch.stack(pb_mask, dim=1)                          # (B, M)

        bag_repr, _ = self.tier2(pb, mask=pbm)                     # (B, D)
        logits = torch.cat([clf(bag_repr) for clf in self.classifiers], dim=1)
        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': _uniform_attn(B, self.num_labels, N, mask, bags.device),
        }


class M4MultiLabel(nn.Module):
    """M4-style baseline: shared gated-attention expert pool + per-label gating.

    Captures the Multi-Proxy Multi-Gate Mixture-of-Experts pattern: a pool of
    ``num_experts`` shared gated-attention experts each produce a candidate bag
    representation, and a per-label linear-softmax gating network mixes those
    M representations into the label-specific bag representation. Distinct from
    CLAM's per-label branches (full per-label attention modules, no sharing) and
    from MIML-LCGA (a single shared attention head conditioned on label
    embeddings, no expert mixture).
    """

    def __init__(self, input_dim, num_labels, embed_dim=192, attn_dim=128,
                 dropout=0.25, embedding_layers=2, num_experts=4, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.num_experts = num_experts
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers)
        self.experts = nn.ModuleList(
            [_GatedAttnPool(embed_dim, attn_dim, dropout) for _ in range(num_experts)])
        # per-label gating networks: bag-level context -> M expert weights
        self.gates = nn.ModuleList(
            [nn.Linear(embed_dim, num_experts) for _ in range(num_labels)])
        self.classifiers = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(num_labels)])

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        B, N, _ = bags.shape
        h = self.instance_embedding(bags)                          # (B, N, D)
        # mean-pooled context over real patches drives the gating
        if mask is not None:
            inv = (~mask.bool()).float().unsqueeze(-1)
            ctx = (h * inv).sum(1) / inv.sum(1).clamp(min=1)
        else:
            ctx = h.mean(dim=1)                                    # (B, D)
        expert_reps = torch.stack(
            [e(h, mask=mask)[0] for e in self.experts], dim=1)     # (B, M, D)
        logits = []
        for l in range(self.num_labels):
            gate = torch.softmax(self.gates[l](ctx), dim=-1)       # (B, M)
            bag_l = (gate.unsqueeze(-1) * expert_reps).sum(1)      # (B, D)
            logits.append(self.classifiers[l](bag_l))
        logits = torch.cat(logits, dim=1)                          # (B, L)
        # M4 does not naturally produce per-label patch attention -> uniform filler
        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': _uniform_attn(B, self.num_labels, N, mask, bags.device),
        }


class DoubleTierMultiLabel(nn.Module):
    """Double-Tier Attention multi-label baseline.

    Two-tier per-label attention. Instances are randomly partitioned into
    ``num_pseudo_bags`` pseudo-bags. Tier-1: a per-label gated-attention head
    pools each pseudo-bag (one attention per (label, pseudo-bag)). Tier-2: a
    per-label gated-attention head pools across the M pseudo-bag features.
    Distinct from DTFD-MIL (which has a single, shared tier-2 head) and from
    MIML-LCGA (which has no pseudo-bag structure).
    """

    def __init__(self, input_dim, num_labels, embed_dim=192, attn_dim=128,
                 dropout=0.25, embedding_layers=2, num_pseudo_bags=4, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.num_pseudo_bags = num_pseudo_bags
        self.instance_embedding = InstanceEmbedding(
            input_dim=input_dim, embed_dim=embed_dim,
            dropout=dropout, num_layers=embedding_layers)
        self.tier1 = nn.ModuleList(
            [_GatedAttnPool(embed_dim, attn_dim, dropout) for _ in range(num_labels)])
        self.tier2 = nn.ModuleList(
            [_GatedAttnPool(embed_dim, attn_dim, dropout) for _ in range(num_labels)])
        self.classifiers = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(num_labels)])

    def forward(self, bags, mask=None, bag_labels=None, compute_instance_loss=False):
        B, N, _ = bags.shape
        h = self.instance_embedding(bags)                          # (B, N, D)
        if mask is None:
            mask = torch.zeros(B, N, dtype=torch.bool, device=bags.device)
        mask = mask.bool()

        perm = torch.randperm(N, device=bags.device)
        chunks_h = torch.chunk(h[:, perm, :], self.num_pseudo_bags, dim=1)
        chunks_m = torch.chunk(mask[:, perm], self.num_pseudo_bags, dim=1)

        logits = []
        for l in range(self.num_labels):
            pb_feats, pb_mask = [], []
            for ch_h, ch_m in zip(chunks_h, chunks_m):
                f, _ = self.tier1[l](ch_h, mask=ch_m)              # (B, D)
                pb_feats.append(f)
                pb_mask.append(ch_m.all(dim=1))
            pb = torch.stack(pb_feats, dim=1)                      # (B, M, D)
            pbm = torch.stack(pb_mask, dim=1)                      # (B, M)
            bag_l, _ = self.tier2[l](pb, mask=pbm)                 # (B, D)
            logits.append(self.classifiers[l](bag_l))
        logits = torch.cat(logits, dim=1)                          # (B, L)
        return {
            'logits': logits,
            'coupled_logits': logits,
            'attn_weights': _uniform_attn(B, self.num_labels, N, mask, bags.device),
        }
