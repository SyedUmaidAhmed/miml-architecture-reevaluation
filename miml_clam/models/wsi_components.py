"""Large-N components for MIML-LCGA on whole-slide images.

WSI bags hold thousands of patches. The standard ``LabelConditionalGatedAttention``
materialises a ``(B, L, N, 2D)`` concatenation tensor — gigabytes at WSI scale
(e.g. B=4, L=15, N=10000, D=64 -> ~1.5 GB just for that one tensor).

``EfficientLabelConditionalGatedAttention`` computes the **identical** function
without that tensor. The attention input is the concatenation ``[h_i ; q_l]``,
and a linear layer applied to a concatenation is separable::

    W [h_i ; q_l] = W_h h_i + W_q q_l

so the instance term ``W_h h_i`` is computed once for all labels, the label term
``W_q q_l`` once for all instances, and the two are combined by broadcasting
inside small per-label-chunk tensors of shape ``(B, chunk, N, attn_dim)``. Peak
memory is bounded by ``label_chunk`` regardless of the number of labels, and
``attn_dim`` (typically 32) replaces ``2D`` (typically 128).

It shares the exact parameter layout of ``LabelConditionalGatedAttention``
(``label_embeddings``, ``attention_a.0``, ``attention_b.0``, ``attention_c``), so
it is a drop-in replacement: ``state_dict``s are interchangeable in both
directions and the forward pass is mathematically exact, not an approximation.
"""

import torch
import torch.nn as nn


class EfficientLabelConditionalGatedAttention(nn.Module):
    """Memory-efficient, exact reformulation of LabelConditionalGatedAttention.

    Args mirror ``LabelConditionalGatedAttention``; ``label_chunk`` controls how
    many labels are processed at once (peak memory ~ B * label_chunk * N * attn).
    """

    def __init__(self, embed_dim, num_labels, attn_dim=16, dropout=0.25,
                 use_gated=True, label_chunk=8):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.use_gated = use_gated
        self.label_chunk = max(1, label_chunk)

        # Identical parameter layout to LabelConditionalGatedAttention.
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)

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
        """h: (B, N, D); mask: (B, N) True where padded.

        Returns (bag_repr (B, L, D), attn_weights (B, L, N)).
        """
        B, N, D = h.shape

        # A linear on [h ; q] splits into an instance half (cols :D) and a
        # label half (cols D:). Compute each half once.
        lin_a = self.attention_a[0]
        act_a, drop_a = self.attention_a[1], self.attention_a[2]
        h_a = h @ lin_a.weight[:, :D].t()                       # (B, N, attn)
        q_a = self.label_embeddings @ lin_a.weight[:, D:].t()   # (L, attn)

        if self.use_gated:
            lin_b = self.attention_b[0]
            act_b, drop_b = self.attention_b[1], self.attention_b[2]
            h_b = h @ lin_b.weight[:, :D].t()
            q_b = self.label_embeddings @ lin_b.weight[:, D:].t()

        mask_b = mask.bool().unsqueeze(1) if mask is not None else None

        bag_chunks, attn_chunks = [], []
        for s in range(0, self.num_labels, self.label_chunk):
            e = min(s + self.label_chunk, self.num_labels)

            # (B,1,N,attn) + (1,Lc,1,attn) + (attn,) -> (B,Lc,N,attn)
            pre_a = h_a.unsqueeze(1) + q_a[s:e].unsqueeze(0).unsqueeze(2) + lin_a.bias
            a = drop_a(act_a(pre_a))
            if self.use_gated:
                pre_b = h_b.unsqueeze(1) + q_b[s:e].unsqueeze(0).unsqueeze(2) + lin_b.bias
                a = a * drop_b(act_b(pre_b))           # gated: tanh * sigmoid

            scores = self.attention_c(a).squeeze(-1)   # (B, Lc, N)
            if mask_b is not None:
                scores = scores.masked_fill(mask_b, float('-inf'))
            attn = torch.softmax(scores, dim=-1)       # (B, Lc, N)
            bag_chunks.append(torch.bmm(attn, h))      # (B, Lc, D)
            attn_chunks.append(attn)

        return torch.cat(bag_chunks, dim=1), torch.cat(attn_chunks, dim=1)
