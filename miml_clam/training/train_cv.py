"""Cross-Validation training loop for MIML-CLAM++.

Supports:
  - Standard format datasets (scene.mat, reuters_MIML.mat) with 10CV.mat splits
  - DD format datasets (MSCV2, letter_frost, letter_carroll) with random K-fold
  - Per-label threshold optimization
  - Multi-seed ensemble (average probabilities across seeds)
"""

import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from ..data.miml_dataset_cv import MIMLDatasetCV, collate_fn, get_cv_splits
from ..data.flat_knn_dataset import FlatKNNBagDataset
from ..data.miml_dataset_dd import MIMLDatasetDD, collate_fn_dd
from ..data.wsi_dataset import WSIDataset
from ..models.miml_clam import MIML_CLAM
from ..metrics.miml_metrics import (
    AllFive, AllFiveWithThresholds, optimize_thresholds, AveragePrecision as AP_fn
)
from .trainer import CurriculumTrainer


def create_model(config, device='cpu', model_class=None):
    """Create model from config dict.

    Args:
        config: Hyperparameter dict.
        device: Target device.
        model_class: Model class to instantiate. None defaults to MIML_CLAM.
    """
    if model_class is None:
        model_class = MIML_CLAM
    model = model_class(
        input_dim=config['input_dim'],
        num_labels=config['num_labels'],
        embed_dim=config['embed_dim'],
        nhead=config['nhead'],
        ff_dim=config.get('ff_dim', config['embed_dim'] * 2),
        transformer_layers=config.get('transformer_layers', 1),
        attn_dim=config.get('attn_dim', config['embed_dim'] // 2),
        dropout=config['dropout'],
        k_sample=config['k_sample'],
        use_transformer=config.get('use_transformer', True),
        use_correlation=config.get('use_correlation', True),
        use_instance_clustering=config.get('use_instance_clustering', False),
        use_gated_attention=config.get('use_gated_attention', True),
        embedding_layers=config.get('embedding_layers', 2),
        use_efficient_attn=config.get('use_efficient_attn', False),
        num_attn_experts=config.get('num_attn_experts', 1),
        moe_load_balance=config.get('moe_load_balance', 0.01),
        label_chunk=config.get('label_chunk', 8),
        use_causal=config.get('use_causal', False),
        causal_learnable_C=config.get('causal_learnable_C', False),
        causal_graph_coupling=config.get('causal_graph_coupling', False),
        causal_graph_reg=config.get('causal_graph_reg', 0.1),
        causal_beta_init=config.get('causal_beta_init', 0.0),
    )
    return model.to(device)


def _train_label_matrix(dataset, actual_train_idx):
    """Stack the (N_train, L) {0,1} label matrix for the training indices."""
    return torch.stack([dataset.labels[i] for i in actual_train_idx])


def _init_model(model, config, dataset, actual_train_idx, device):
    """Initialize the correlation head from training-set label statistics.

    Statistical head: raw co-occurrence P(j|i) (unchanged legacy path).
    Causal head: a deconfounded coupling (log odds ratio / lift / backdoor)
    computed from the *training* labels only, so no test information leaks.
    """
    if not hasattr(model, 'label_head'):
        return

    if config.get('use_causal', False):
        from ..data.causal_stats import causal_coupling, make_strata
        Y_train = _train_label_matrix(dataset, actual_train_idx)
        method = config.get('causal_method', 'odds_ratio')
        strata = None
        if method == 'backdoor':
            strata = make_strata(Y_train, n_strata=config.get('causal_strata', 4),
                                 seed=0)
        C = causal_coupling(
            Y_train, method=method,
            correction=config.get('causal_correction', 0.5),
            strata=strata, normalize=config.get('causal_normalize', True))
        model.label_head.init_causal(C.to(device))
    elif config.get('use_correlation', True):
        co_matrix = dataset.get_label_cooccurrence(actual_train_idx)
        model.label_head.init_correlation_from_data(co_matrix.to(device))


def _emit_probs(prob_sink, fold_idx, seed, val_labels, val_raw, val_coupled,
                test_labels, test_raw, test_coupled):
    """Hand one (fold, seed) worth of probabilities to an optional observer.

    No-op when prob_sink is None, which is the default everywhere. Arrays are
    passed as copies so a sink cannot perturb the metrics computed downstream.
    """
    if prob_sink is None:
        return
    prob_sink({
        'fold': int(fold_idx),
        'seed': None if seed is None else int(seed),
        'val_labels': np.asarray(val_labels).copy(),
        'val_raw': np.asarray(val_raw).copy(),
        'val_coupled': np.asarray(val_coupled).copy(),
        'test_labels': np.asarray(test_labels).copy(),
        'test_raw': np.asarray(test_raw).copy(),
        'test_coupled': np.asarray(test_coupled).copy(),
    })


def run_fold(fold_idx, dataset_orig, train_idx, test_idx, config, device='cpu',
             model_class=None, prob_sink=None):
    """Run a single fold of cross-validation.

    Supports threshold optimization and multi-seed ensemble.

    Args:
        prob_sink: Optional callable invoked once per (fold, seed) with a dict of
            validation/test labels and raw/coupled probabilities. Purely an
            observer — it must not mutate its argument and does not affect any
            metric computed here.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Fold {fold_idx + 1}", flush=True)
    print(f"{'='*60}", flush=True)

    dataset = copy.deepcopy(dataset_orig)
    dataset.normalize_features(train_idx)

    # Split train into train/val (90/10)
    np.random.seed(fold_idx)
    train_arr = np.array(train_idx)
    np.random.shuffle(train_arr)
    val_size = max(len(train_arr) // 10, 1)
    val_idx = train_arr[:val_size].tolist()
    actual_train_idx = train_arr[val_size:].tolist()
    if hasattr(dataset, 'prepare_fold'):
        # Leak-free k-NN bagging: neighbours come from the fitting split only.
        dataset.prepare_fold(actual_train_idx)

    collate = collate_fn_dd if config.get('dataset_format', 'standard') == 'dd' else collate_fn

    sampler = dataset.get_balanced_sampler(actual_train_idx)
    train_loader = DataLoader(
        Subset(dataset, actual_train_idx), batch_size=config['batch_size'],
        sampler=sampler, collate_fn=collate, num_workers=0, drop_last=False)
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate, num_workers=0)

    # Multi-seed ensemble
    # A single-element seed list routes here too, so that the one model it trains
    # is explicitly seeded. For len(seeds) == 1 the ensemble path is arithmetically
    # identical to the single-model path below (mean over one array is that array).
    ensemble_seeds = config.get('ensemble_seeds', None)
    if ensemble_seeds:
        return _run_fold_ensemble(
            fold_idx, dataset, actual_train_idx, val_idx, test_idx,
            train_loader, val_loader, test_loader, config, device, ensemble_seeds,
            model_class=model_class, prob_sink=prob_sink)

    # Single model training
    model = create_model(config, device, model_class=model_class)
    _init_model(model, config, dataset, actual_train_idx, device)

    trainer = CurriculumTrainer(model, config, device)

    print("Phase 1: Independent label training...", flush=True)
    trainer.train_phase1(
        train_loader, val_loader,
        num_epochs=config.get('phase1_epochs', 80),
        lr=config.get('phase1_lr', 1e-3))

    print("Phase 2: Joint training...", flush=True)
    trainer.train_phase2(
        train_loader, val_loader,
        max_epochs=config.get('phase2_max_epochs', 120),
        lr=config.get('phase2_lr', 3e-4),
        patience=config.get('early_stop_patience', 25))

    use_threshold_opt = config.get('use_threshold_opt', False)

    if use_threshold_opt or prob_sink is not None:
        val_labels, val_raw, val_coupled = trainer.get_predictions(val_loader)
        test_labels, test_raw, test_coupled = trainer.get_predictions(test_loader)
        _emit_probs(prob_sink, fold_idx, None, val_labels, val_raw, val_coupled,
                    test_labels, test_raw, test_coupled)

    if use_threshold_opt:
        # Pick raw or coupled based on val AP
        val_ap_raw = AP_fn(val_labels, val_raw)
        val_ap_coupled = AP_fn(val_labels, val_coupled)

        if val_ap_coupled >= val_ap_raw:
            val_probs, test_probs = val_coupled, test_coupled
            source = "coupled"
        else:
            val_probs, test_probs = val_raw, test_raw
            source = "raw"

        thresholds = optimize_thresholds(val_labels, val_probs, metric='hamming')
        print(f"  Optimized thresholds ({source}): {thresholds.round(3)}", flush=True)
        test_metrics = AllFiveWithThresholds(test_labels, test_probs, thresholds)
    else:
        test_metrics = trainer.evaluate(test_loader)

    print(f"\nFold {fold_idx + 1} Results:", flush=True)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    return test_metrics


def _run_fold_ensemble(fold_idx, dataset, actual_train_idx, val_idx, test_idx,
                       train_loader, val_loader, test_loader, config, device, seeds,
                       model_class=None, prob_sink=None):
    """Run multiple seeds and average predictions for ensemble."""
    print(f"  Running {len(seeds)}-seed ensemble...", flush=True)

    all_test_probs = []
    all_val_probs = []
    test_labels = None
    val_labels = None

    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_model(config, device, model_class=model_class)
        _init_model(model, config, dataset, actual_train_idx, device)
        trainer = CurriculumTrainer(model, config, device)

        print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)}):", flush=True)
        trainer.train_phase1(
            train_loader, val_loader,
            num_epochs=config.get('phase1_epochs', 80),
            lr=config.get('phase1_lr', 1e-3))
        trainer.train_phase2(
            train_loader, val_loader,
            max_epochs=config.get('phase2_max_epochs', 120),
            lr=config.get('phase2_lr', 3e-4),
            patience=config.get('early_stop_patience', 25))

        vl, vr, vc = trainer.get_predictions(val_loader)
        tl, tr, tc = trainer.get_predictions(test_loader)
        _emit_probs(prob_sink, fold_idx, seed, vl, vr, vc, tl, tr, tc)

        if AP_fn(vl, vc) >= AP_fn(vl, vr):
            all_val_probs.append(vc)
            all_test_probs.append(tc)
        else:
            all_val_probs.append(vr)
            all_test_probs.append(tr)

        val_labels = vl
        test_labels = tl

    # Average probabilities across seeds
    val_probs = np.mean(all_val_probs, axis=0)
    test_probs = np.mean(all_test_probs, axis=0)

    use_threshold_opt = config.get('use_threshold_opt', False)
    if use_threshold_opt:
        thresholds = optimize_thresholds(val_labels, val_probs, metric='hamming')
        print(f"  Ensemble thresholds: {thresholds.round(3)}", flush=True)
        test_metrics = AllFiveWithThresholds(test_labels, test_probs, thresholds)
    else:
        test_metrics = AllFive(test_labels, test_probs)

    print(f"\nFold {fold_idx + 1} Ensemble Results:", flush=True)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    return test_metrics


def run_cv(config, device='cpu', num_folds=None, model_class=None, prob_sink=None):
    """Run cross-validation.

    Supports both standard (.mat with 10CV.mat) and DD format datasets.

    prob_sink is an optional observer callback forwarded to each fold; see
    run_fold. Leaving it None reproduces the original behaviour exactly.
    """
    dataset_format = config.get('dataset_format', 'standard')

    if dataset_format == 'wsi':
        dataset = WSIDataset(
            feature_dir=config['feature_dir'],
            label_csv=config['label_csv'],
            slide_id_col=config.get('slide_id_col'),
            label_cols=config.get('label_cols'),
            feature_key=config.get('feature_key', 'features'),
            max_patches=config.get('max_patches'),
        )
    elif dataset_format == 'dd':
        dataset = MIMLDatasetDD(mat_path=config['data_path'])
    elif dataset_format == 'flat_knn':
        dataset = FlatKNNBagDataset(
            mat_path=config['data_path'],
            bag_key=config.get('bag_key', 'bags'),
            label_key=config.get('label_key', 'labels'),
            k_neighbors=config.get('knn_bag_k', 4),
        )
    else:
        dataset = MIMLDatasetCV(
            mat_path=config['data_path'],
            bag_key=config.get('bag_key', 'bags'),
            label_key=config.get('label_key', 'labels'),
        )

    if config.get('legacy_std_clamp', False):
        dataset.legacy_std_clamp = True
    config['input_dim'] = dataset.input_dim
    config['num_labels'] = dataset.num_labels

    print(f"Dataset: {len(dataset)} samples, {dataset.num_labels} labels, "
          f"{dataset.input_dim}-dim features", flush=True)

    # CV splits
    if 'cv_mat_path' in config and config['cv_mat_path']:
        splits = get_cv_splits(config['cv_mat_path'], len(dataset), 10)
    else:
        k = num_folds or 5
        indices = np.arange(len(dataset))
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(dataset) // k
        splits = []
        for i in range(k):
            test_idx = indices[i * fold_size:(i + 1) * fold_size].tolist()
            train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]]).tolist()
            splits.append((train_idx, test_idx))

    if num_folds is not None:
        splits = splits[:num_folds]

    all_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_metrics = run_fold(fold_idx, dataset, train_idx, test_idx, config, device,
                               model_class=model_class, prob_sink=prob_sink)
        all_metrics.append(fold_metrics)

    # Aggregate results
    n = len(splits)
    print(f"\n{'='*60}", flush=True)
    print(f"{n}-Fold CV Results (mean +/- std)", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    for name in all_metrics[0].keys():
        values = [m[name] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        results[name] = (mean, std)
        print(f"  {name}: {mean:.4f} +/- {std:.4f}", flush=True)

    return results, all_metrics
