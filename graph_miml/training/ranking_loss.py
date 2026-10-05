"""Innovation 3: Pairwise Ranking Loss.

Directly optimizes ranking metrics (OneError, Coverage, RankingLoss, AveragePrecision).
For each sample, penalizes when a negative label is ranked above a positive label.
Uses smooth approximation (soft margin) for differentiability.
"""

import torch
import torch.nn as nn


class PairwiseRankingLoss(nn.Module):
    """Pairwise ranking loss for multi-label learning.

    For each sample, considers all (positive, negative) label pairs.
    Penalizes when the score of a negative label exceeds the positive label's score.

    Loss = (1/|Y_i| * |Y_i_bar|) * sum_{(p,n)} max(0, margin - (s_p - s_n))

    Uses soft margin (log-sigmoid) for smoother gradients.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, L) — raw logits (before sigmoid)
            labels: (B, L) — binary labels {0, 1}

        Returns:
            scalar ranking loss
        """
        B, L = logits.shape

        # Masks for positive and negative labels
        pos_mask = (labels > 0.5)  # (B, L)
        neg_mask = ~pos_mask       # (B, L)

        total_loss = torch.tensor(0.0, device=logits.device)
        num_valid = 0

        for b in range(B):
            pos_idx = pos_mask[b].nonzero(as_tuple=True)[0]
            neg_idx = neg_mask[b].nonzero(as_tuple=True)[0]

            n_pos = pos_idx.shape[0]
            n_neg = neg_idx.shape[0]

            if n_pos == 0 or n_neg == 0:
                continue

            # Scores for positive and negative labels
            pos_scores = logits[b, pos_idx]  # (n_pos,)
            neg_scores = logits[b, neg_idx]  # (n_neg,)

            # All pairwise differences: pos_i - neg_j
            # (n_pos, 1) - (1, n_neg) -> (n_pos, n_neg)
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)

            # Soft margin ranking loss: log(1 + exp(margin - diff))
            pair_loss = torch.nn.functional.softplus(self.margin - diff)

            # Normalize by number of pairs
            total_loss = total_loss + pair_loss.sum() / (n_pos * n_neg)
            num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        return total_loss
