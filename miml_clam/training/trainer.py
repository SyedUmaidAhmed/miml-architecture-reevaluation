"""Curriculum training for MIML-CLAM++.

Two-phase training:
  Phase 1: Independent label training (BCE on raw logits, cosine annealing)
  Phase 2: Joint training (BCE on coupled logits, SWA, early stopping on val AP)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np

from .augmentation import bag_mixup
from ..metrics.miml_metrics import AllFive


class CurriculumTrainer:
    """Two-phase curriculum trainer with SWA and early stopping."""

    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.label_smooth = config.get('label_smoothing', 0.0)
        self.bce_loss = nn.BCEWithLogitsLoss()

        # --- Claim-2 levers (all off by default; default path is unchanged) ---
        # Asymmetric loss for imbalanced/high-L datasets (replaces the Phase-2
        # main objective on coupled logits).
        if config.get('use_asl', False):
            from .asymmetric_loss import AsymmetricLoss
            self.loss_fn = AsymmetricLoss(
                gamma_neg=config.get('asl_gamma_neg', 4),
                gamma_pos=config.get('asl_gamma_pos', 1),
                clip=config.get('asl_clip', 0.05))
        # Pairwise ranking loss directly targets OneError/Coverage/RankingLoss/AP.
        self.lambda_rank = config.get('lambda_rank', 0.0)
        if self.lambda_rank > 0:
            from graph_miml.training.ranking_loss import PairwiseRankingLoss
            self.rank_loss = PairwiseRankingLoss(margin=config.get('rank_margin', 1.0))
        # Weight on the causal head's graph-coupling anchor regulariser.
        self.lambda_coupling_reg = config.get('lambda_coupling_reg', 1.0)

    def _smooth_labels(self, labels):
        if self.label_smooth > 0:
            return labels * (1 - self.label_smooth) + (1 - labels) * self.label_smooth
        return labels

    def train_phase1(self, train_loader, val_loader, num_epochs=80, lr=1e-3):
        """Phase 1: Independent label training on raw logits."""
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=self.config.get('weight_decay', 5e-4)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(num_epochs // 3, 10), T_mult=1
        )

        best_ap = 0.0
        best_state = None

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for bags, labels, mask in train_loader:
                bags, labels, mask = bags.to(self.device), labels.to(self.device), mask.to(self.device)

                if self.config.get('use_mixup', False):
                    fixed = self.config.get('fixed_bag_size', True)
                    bags, labels, mask = bag_mixup(bags, labels, mask, alpha=0.2, fixed_size=fixed)

                smooth_labels = self._smooth_labels(labels)
                optimizer.zero_grad()

                result = self.model(bags, mask=mask, bag_labels=labels)
                loss = self.bce_loss(result['logits'], smooth_labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.get('gradient_clip', 1.0))
                optimizer.step()
                scheduler.step(epoch + num_batches / len(train_loader))

                epoch_loss += loss.item()
                num_batches += 1

            val_metrics = self.evaluate(val_loader, use_raw_logits=True)
            val_ap = val_metrics['AveragePrecision']
            if val_ap > best_ap:
                best_ap = val_ap
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Phase1 Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/max(num_batches,1):.4f} | "
                      f"Val AP: {val_ap:.4f} | Best: {best_ap:.4f}", flush=True)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return best_ap

    def train_phase2(self, train_loader, val_loader, max_epochs=120, lr=3e-4,
                     patience=25, inst_warmup=5):
        """Phase 2: Joint training with SWA and early stopping."""
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=self.config.get('weight_decay', 5e-4)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        # SWA
        swa_model = AveragedModel(self.model)
        swa_start = self.config.get('swa_start', 30)
        swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.5, anneal_epochs=5)
        swa_active = False

        bag_weight = self.config.get('bag_weight', 0.9)
        inst_weight = self.config.get('inst_weight', 0.05)

        best_score = float('-inf')
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            compute_inst = (epoch >= inst_warmup) and getattr(self.model, 'use_instance_clustering', False)

            for bags, labels, mask in train_loader:
                bags, labels, mask = bags.to(self.device), labels.to(self.device), mask.to(self.device)
                optimizer.zero_grad()

                # Instance clustering on clean data
                inst_loss_val = None
                if compute_inst:
                    result_inst = self.model(bags, mask=mask, bag_labels=labels, compute_instance_loss=True)
                    if 'instance_loss' in result_inst:
                        inst_loss_val = result_inst['instance_loss']

                # Mixup (optional)
                mixed_bags, mixed_labels, mixed_mask = bags, labels, mask
                if self.config.get('use_mixup', False):
                    fixed = self.config.get('fixed_bag_size', True)
                    mixed_bags, mixed_labels, mixed_mask = bag_mixup(
                        bags, labels, mask, alpha=0.2, fixed_size=fixed)

                result = self.model(mixed_bags, mask=mixed_mask, bag_labels=mixed_labels)

                # Main loss on coupled logits
                total_loss = bag_weight * self.loss_fn(result['coupled_logits'], mixed_labels)

                if inst_loss_val is not None:
                    total_loss = total_loss + inst_weight * inst_loss_val

                # Pairwise ranking loss on coupled logits (Claim 2 ranking lever).
                # Uses hard {0,1} labels, so skip when mixup produced soft labels.
                if self.lambda_rank > 0 and not self.config.get('use_mixup', False):
                    total_loss = total_loss + self.lambda_rank * self.rank_loss(
                        result['coupled_logits'], mixed_labels)

                # Causal graph-coupling anchor regulariser (if the causal head
                # with graph coupling is active).
                if 'coupling_reg' in result:
                    total_loss = total_loss + self.lambda_coupling_reg * result['coupling_reg']

                # Mixture-of-experts attention: keep the router from collapsing
                # every label onto one expert, which would silently reduce the
                # module back to the shared-attention design it is meant to
                # replace. No-op unless MoE attention is active.
                attn = getattr(self.model, 'label_attention', None)
                if attn is not None and hasattr(attn, 'load_balance_loss'):
                    total_loss = total_loss + attn.load_balance_loss()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.get('gradient_clip', 1.0))
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            # SWA
            if epoch >= swa_start:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                swa_active = True
            else:
                scheduler.step()

            # Validation
            eval_model = swa_model.module if swa_active else self.model
            val_metrics = self._evaluate_model(eval_model, val_loader)
            val_ap = val_metrics['AveragePrecision']

            if (epoch + 1) % 5 == 0:
                swa_str = " [SWA]" if swa_active else ""
                print(f"  Phase2 Epoch {epoch+1}/{max_epochs} | Loss: {epoch_loss/max(num_batches,1):.4f} | "
                      f"Val AP: {val_ap:.4f}{swa_str}", flush=True)

            if val_ap > best_score:
                best_score = val_ap
                patience_counter = 0
                state_dict = swa_model.module.state_dict() if swa_active else self.model.state_dict()
                best_state = {k: v.cpu().clone() for k, v in state_dict.items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stop at epoch {epoch+1} (best AP: {best_score:.4f})", flush=True)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return best_score

    def _evaluate_model(self, model, data_loader, use_raw_logits=False):
        """Evaluate a model on data loader."""
        model.eval()
        all_labels, all_raw, all_coupled = [], [], []

        with torch.no_grad():
            for bags, labels, mask in data_loader:
                bags, mask = bags.to(self.device), mask.to(self.device)
                result = model(bags, mask=mask)
                all_raw.append(torch.sigmoid(result['logits']).cpu().numpy())
                all_coupled.append(torch.sigmoid(result['coupled_logits']).cpu().numpy())
                all_labels.append(labels.numpy())

        all_labels = np.concatenate(all_labels)
        all_raw = np.concatenate(all_raw)
        all_coupled = np.concatenate(all_coupled)

        metrics_raw = AllFive(all_labels, all_raw)
        metrics_coupled = AllFive(all_labels, all_coupled)

        if use_raw_logits:
            return metrics_raw
        return metrics_coupled if metrics_coupled['AveragePrecision'] >= metrics_raw['AveragePrecision'] else metrics_raw

    @torch.no_grad()
    def evaluate(self, data_loader, use_raw_logits=False):
        return self._evaluate_model(self.model, data_loader, use_raw_logits)

    @torch.no_grad()
    def get_predictions(self, data_loader):
        """Get raw predictions and labels (for threshold optimization)."""
        self.model.eval()
        all_labels, all_raw, all_coupled = [], [], []

        for bags, labels, mask in data_loader:
            bags, mask = bags.to(self.device), mask.to(self.device)
            result = self.model(bags, mask=mask)
            all_raw.append(torch.sigmoid(result['logits']).cpu().numpy())
            all_coupled.append(torch.sigmoid(result['coupled_logits']).cpu().numpy())
            all_labels.append(labels.numpy())

        return (np.concatenate(all_labels),
                np.concatenate(all_raw),
                np.concatenate(all_coupled))
