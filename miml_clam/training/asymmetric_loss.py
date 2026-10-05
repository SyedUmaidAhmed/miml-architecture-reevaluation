"""Asymmetric Loss for Multi-Label Classification.

Adapted from Ben-Baruch et al. (2020) "Asymmetric Loss For Multi-Label Classification".
Key insight: In multi-label, most labels per sample are negative. Standard BCE wastes
gradient capacity on easy negatives. ASL applies:
  - Probability shifting: clip low-confidence negatives to zero
  - Asymmetric focusing: gamma_neg >> gamma_pos to focus on hard negatives
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Args:
        gamma_neg: focusing parameter for negatives (higher = more down-weighting of easy negatives)
        gamma_pos: focusing parameter for positives (0 = no focusing, keep all positive gradients)
        clip: probability margin for hard thresholding negatives (probability shifting)
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, L) raw logits
            targets: (B, L) labels in [0, 1] (supports soft labels from mixup/label smoothing)

        Returns:
            scalar loss
        """
        # Probabilities
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos

        # Asymmetric Clipping (probability shifting for negatives)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE terms
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg

        # Asymmetric Focusing — use detach() for the weighting factor
        # so gradients flow only through the CE terms, not the weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos.detach() * targets
            pt1 = xs_neg.detach() * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()
