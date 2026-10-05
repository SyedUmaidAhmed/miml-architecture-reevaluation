#!/usr/bin/env python3
"""Run neural baseline experiments for fair comparison with MIML-LCGA.

All models use identical hyperparameters, CV splits, and training protocol.

Usage:
    python run_baseline_experiments.py --model all --dataset scene --mode single --folds 2  # quick test
    python run_baseline_experiments.py --model all --dataset all --mode single
    python run_baseline_experiments.py --model all --dataset all --mode ensemble
    python run_baseline_experiments.py --model all --dataset all --mode both
    python run_baseline_experiments.py --model abmil_ml --dataset scene --mode single
"""

import torch
import numpy as np
import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miml_clam.models.miml_clam import MIML_CLAM
from miml_clam.models.baselines import MLPPoolML, TransformerPoolML, ABMILML
from miml_clam.models.wsi_baselines import CLAMMultiLabel
from miml_clam.training.train_cv import run_cv

# Benchmarks live in ./data (download data.zip from the GitHub release);
# override with the MIML_DATA_DIR environment variable.
DATA_DIR = os.environ.get('MIML_DATA_DIR',
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

MODEL_REGISTRY = {
    'miml_lcga': MIML_CLAM,
    'abmil_ml': ABMILML,
    # One independent gated-attention branch per label (CLAM's design). This is
    # the control for the label-conditional attention claim: ABMIL-ML shows only
    # that per-label attention helps at all, whereas this shows whether
    # conditioning ONE shared attention on a label embedding beats the obvious
    # alternative of instantiating L separate heads. Defined in wsi_baselines.py
    # and already used by the WSI track as 'clam'; the same class, same contract.
    'clam_multilabel_ml': CLAMMultiLabel,
    'transformer_pool_ml': TransformerPoolML,
    'mlp_pool_ml': MLPPoolML,
}

DATASET_CONFIGS = {
    'scene': {
        'data_path': os.path.join(DATA_DIR, 'scene.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, '10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'scene',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': True, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'reuters': {
        'data_path': os.path.join(DATA_DIR, 'reuters_MIML.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, '10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'reuters',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'mscv2': {
        'data_path': os.path.join(DATA_DIR, 'MSCV2.mat'),
        'cv_mat_path': None, 'dataset': 'mscv2',
        'dataset_format': 'dd',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'default_folds': 5,
    },
    'letter_frost': {
        'data_path': os.path.join(DATA_DIR, 'letter_frost.mat'),
        'cv_mat_path': None, 'dataset': 'letter_frost',
        'dataset_format': 'dd',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'default_folds': 5,
    },
    'letter_carroll': {
        'data_path': os.path.join(DATA_DIR, 'letter_carroll.mat'),
        'cv_mat_path': None, 'dataset': 'letter_carroll',
        'dataset_format': 'dd',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'default_folds': 5,
    },
    'yeast': {
        'data_path': os.path.join(DATA_DIR, 'yeast.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'yeast_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'yeast',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': True, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'birdsong': {
        'data_path': os.path.join(DATA_DIR, 'birdsong.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'birdsong_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'birdsong',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'protein_haloarcula': {
        'data_path': os.path.join(DATA_DIR, 'protein_haloarcula.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_haloarcula_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'protein_haloarcula',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'protein_pyrococcus': {
        'data_path': os.path.join(DATA_DIR, 'protein_pyrococcus.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_pyrococcus_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'protein_pyrococcus',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'protein_geobacter': {
        'data_path': os.path.join(DATA_DIR, 'protein_geobacter.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_geobacter_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'protein_geobacter',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'protein_azotobacter': {
        'data_path': os.path.join(DATA_DIR, 'protein_azotobacter.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_azotobacter_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'protein_azotobacter',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': False, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'corel5k': {
        'data_path': os.path.join(DATA_DIR, 'corel5k.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'corel5k_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'corel5k',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': False,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': True, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'emotions': {
        'data_path': os.path.join(DATA_DIR, 'emotions.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'emotions_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'emotions',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': True, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'medical': {
        'data_path': os.path.join(DATA_DIR, 'medical.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'medical_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'medical',
        'dataset_format': 'standard',
        'embed_dim': 64, 'nhead': 4, 'ff_dim': 128, 'transformer_layers': 1,
        'attn_dim': 32, 'k_sample': 2, 'embedding_layers': 2,
        'use_transformer': True, 'use_correlation': True,
        'use_instance_clustering': False, 'use_gated_attention': True,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3,
        'phase2_max_epochs': 120, 'phase2_lr': 3e-4,
        'early_stop_patience': 25, 'bag_weight': 0.9, 'inst_weight': 0.05,
        'swa_start': 30, 'label_smoothing': 0.05,
        'fixed_bag_size': True, 'use_mixup': False,
        'dropout': 0.15, 'use_threshold_opt': True,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
}


def run_single(model_name, model_class, config, device, num_folds):
    """Run single-model (no ensemble) experiment."""
    cfg = config.copy()
    cfg.pop('ensemble_seeds', None)
    results, per_fold = run_cv(cfg, device, num_folds=num_folds,
                               model_class=model_class)
    return results, per_fold


def run_ensemble(model_name, model_class, config, device, num_folds):
    """Run ensemble experiment."""
    cfg = config.copy()
    if 'ensemble_seeds' not in cfg:
        cfg['ensemble_seeds'] = [42, 123, 456, 789, 1024]
    results, per_fold = run_cv(cfg, device, num_folds=num_folds,
                               model_class=model_class)
    return results, per_fold


def main():
    parser = argparse.ArgumentParser(description='Neural Baseline Experiments')
    parser.add_argument('--model', type=str, default='all',
                        choices=list(MODEL_REGISTRY.keys()) + ['all'],
                        help='Model to run (default: all)')
    parser.add_argument('--dataset', type=str, default='scene',
                        choices=list(DATASET_CONFIGS.keys()) + ['all'],
                        help='Dataset to run (default: scene)')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'ensemble', 'both'],
                        help='single, ensemble, or both')
    parser.add_argument('--folds', type=int, default=None,
                        help='Override number of CV folds')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == 'all' else [args.model]
    datasets_to_run = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]

    modes = []
    if args.mode in ('single', 'both'):
        modes.append('single')
    if args.mode in ('ensemble', 'both'):
        modes.append('ensemble')

    all_results = {}

    for dataset_name in datasets_to_run:
        config = DATASET_CONFIGS[dataset_name].copy()

        if args.data_dir:
            for key in ['data_path', 'cv_mat_path']:
                if config.get(key):
                    filename = os.path.basename(config[key])
                    config[key] = os.path.join(args.data_dir, filename)

        num_folds = args.folds or config.get('default_folds', 10)

        for model_name in models_to_run:
            model_class = MODEL_REGISTRY[model_name]

            for mode in modes:
                key = f"{dataset_name}/{model_name}/{mode}"
                print(f"\n{'#'*70}")
                print(f"# {key} ({num_folds}-fold CV)")
                print(f"{'#'*70}")

                torch.manual_seed(42)
                np.random.seed(42)

                if mode == 'single':
                    results, per_fold = run_single(
                        model_name, model_class, config, device, num_folds)
                else:
                    results, per_fold = run_ensemble(
                        model_name, model_class, config, device, num_folds)

                all_results[key] = {
                    'summary': {k: [float(v[0]), float(v[1])] for k, v in results.items()},
                    'per_fold': [
                        {k: float(v) for k, v in fold.items()} for fold in per_fold
                    ],
                    'model': model_name,
                    'dataset': dataset_name,
                    'mode': mode,
                    'num_folds': num_folds,
                }

                # Print summary
                print(f"\n  Summary ({key}):")
                for metric in ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision']:
                    if metric in results:
                        print(f"    {metric}: {results[metric][0]:.4f} +/- {results[metric][1]:.4f}")

    # Save
    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/baselines_{ts}.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")
    for dataset_name in datasets_to_run:
        print(f"\n  Dataset: {dataset_name}")
        print(f"  {'Model':<25} {'Mode':<10} {'HL':>8} {'OE':>8} {'Cov':>8} {'RL':>8} {'AP':>8}")
        print(f"  {'-'*85}")
        for model_name in models_to_run:
            for mode in modes:
                key = f"{dataset_name}/{model_name}/{mode}"
                if key in all_results:
                    s = all_results[key]['summary']
                    print(f"  {model_name:<25} {mode:<10} "
                          f"{s['HammingLoss'][0]:>8.4f} "
                          f"{s['OneError'][0]:>8.4f} "
                          f"{s['Coverage'][0]:>8.4f} "
                          f"{s['RankingLoss'][0]:>8.4f} "
                          f"{s['AveragePrecision'][0]:>8.4f}")


if __name__ == '__main__':
    main()
