"""Bag-level Mixup augmentation for MIML."""

import torch
import numpy as np


def bag_mixup(bags, labels, mask, alpha=0.2, fixed_size=True):
    """Apply bag-level mixup augmentation.

    For fixed-size bags (Scene): interpolate instance features, soft labels.
    For variable-length bags (Reuters): concatenate instances, soft labels.

    Args:
        bags: (B, N, D)
        labels: (B, L)
        mask: (B, N) — True where padded
        alpha: Beta distribution parameter
        fixed_size: If True, all bags have same # instances (interpolate mode)

    Returns:
        mixed_bags, mixed_labels, mixed_mask
    """
    B = bags.size(0)
    if B < 2:
        return bags, labels, mask

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    lam = max(lam, 1 - lam)
    indices = torch.randperm(B, device=bags.device)

    if fixed_size:
        mixed_bags = lam * bags + (1 - lam) * bags[indices]
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        mixed_mask = mask
    else:
        mixed_bags_list = []
        mixed_mask_list = []
        mixed_labels = lam * labels + (1 - lam) * labels[indices]

        for i in range(B):
            j = indices[i].item()
            valid_i = (~mask[i].bool()).sum().item()
            valid_j = (~mask[j].bool()).sum().item()

            bag_i = bags[i, :valid_i]
            bag_j = bags[j, :valid_j]

            combined = torch.cat([bag_i, bag_j], dim=0)
            mixed_bags_list.append(combined)
            mixed_mask_list.append(torch.zeros(combined.shape[0], device=bags.device))

        max_len = max(b.shape[0] for b in mixed_bags_list)
        D = bags.shape[2]
        mixed_bags = torch.zeros(B, max_len, D, device=bags.device)
        mixed_mask = torch.ones(B, max_len, device=bags.device)

        for i, (b, m) in enumerate(zip(mixed_bags_list, mixed_mask_list)):
            n = b.shape[0]
            mixed_bags[i, :n] = b
            mixed_mask[i, :n] = 0

    return mixed_bags, mixed_labels, mixed_mask
