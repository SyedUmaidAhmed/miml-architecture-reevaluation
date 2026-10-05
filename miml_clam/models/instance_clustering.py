"""Component E: Instance Clustering Loss (adapted from CLAM for multi-label).

For each label: top-k attended -> pseudo-positive, bottom-k -> pseudo-negative.
Per-label binary instance classifiers provide pseudo instance-level supervision.
"""

import torch
import torch.nn as nn


class InstanceClusteringLoss(nn.Module):
    """Multi-label instance clustering loss adapted from CLAM.

    For each label, uses attention weights to identify:
    - Top-k attended instances (pseudo-positives for that label)
    - Bottom-k attended instances (pseudo-negatives for that label)
    Then trains per-label binary classifiers on these pseudo-labels.
    """

    def __init__(self, embed_dim, num_labels, k_sample=3):
        super().__init__()
        self.k_sample = k_sample
        self.num_labels = num_labels

        self.instance_classifiers = nn.ModuleList([
            nn.Linear(embed_dim, 2) for _ in range(num_labels)
        ])
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, h, attn_weights, bag_labels, mask=None):
        """
        h: (B, N, D) — contextual instance representations
        attn_weights: (B, L, N) — attention weights per label
        bag_labels: (B, L) — bag-level binary labels
        mask: (B, N) — True where padded

        Returns:
            inst_loss: scalar — average instance clustering loss
            inst_acc: float — instance classification accuracy
        """
        B, N, D = h.shape
        device = h.device
        total_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_count = 0

        for b in range(B):
            for l in range(self.num_labels):
                if bag_labels[b, l] < 0.5:
                    continue

                if mask is not None:
                    valid_mask = ~mask[b].bool()
                    num_valid = valid_mask.sum().item()
                else:
                    num_valid = N
                    valid_mask = torch.ones(N, dtype=torch.bool, device=device)

                k = min(self.k_sample, num_valid // 2)
                if k < 1:
                    continue

                a = attn_weights[b, l]
                valid_indices = torch.where(valid_mask)[0]
                a_valid = a[valid_indices]

                top_k_idx = torch.topk(a_valid, k)[1]
                bot_k_idx = torch.topk(-a_valid, k)[1]

                top_instances = h[b][valid_indices[top_k_idx]]
                bot_instances = h[b][valid_indices[bot_k_idx]]

                p_targets = torch.ones(k, dtype=torch.long, device=device)
                n_targets = torch.zeros(k, dtype=torch.long, device=device)

                all_instances = torch.cat([top_instances, bot_instances], dim=0)
                all_targets = torch.cat([p_targets, n_targets], dim=0)

                logits = self.instance_classifiers[l](all_instances)
                loss = self.loss_fn(logits, all_targets)
                total_loss = total_loss + loss

                preds = logits.argmax(dim=1)
                total_correct += (preds == all_targets).sum().item()
                total_count += 2 * k

        num_pairs = max((bag_labels > 0.5).sum().item(), 1)
        inst_loss = total_loss / num_pairs
        inst_acc = total_correct / max(total_count, 1)

        return inst_loss, inst_acc
