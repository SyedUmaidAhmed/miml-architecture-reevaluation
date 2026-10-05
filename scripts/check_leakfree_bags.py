#!/usr/bin/env python3
"""Prove the per-fold k-NN bags (FlatKNNBagDataset) contain no held-out rows.

For every fold of emotions / medical / yeast, reproduce run_fold's split
(normalise on the fold-train, seed with the fold index, hold out the first
10% as validation), rebuild the bags, and check:

  1. every bag has k+1 = 5 instances and instance 0 is the sample itself;
  2. every non-self instance of every bag is a row of the fitting split
     (so no validation/test feature enters any bag, fitting or held-out);
  3. normalize_features is idempotent (it always restarts from the raw matrix).

Also reports how the SHIPPED .mat bags violate (2), as the before/after record.

Usage: python scripts/check_leakfree_bags.py [--folds N]
"""
import argparse
import copy
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from miml_clam.data import FlatKNNBagDataset, get_cv_splits  # noqa: E402
import run_baseline_experiments as rbe  # noqa: E402

DATASETS = ["emotions", "medical", "yeast"]


def fold_split(train_idx, fold_idx):
    arr = np.array(train_idx)
    np.random.seed(fold_idx)
    np.random.shuffle(arr)
    v = max(len(arr) // 10, 1)
    return arr[:v], arr[v:]


def row_index(X):
    key = defaultdict(set)
    for i, r in enumerate(X):
        key[r.tobytes()].add(i)
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()
    failures = 0
    for name in DATASETS:
        cfg = rbe.DATASET_CONFIGS[name]
        ds0 = FlatKNNBagDataset(cfg["data_path"])
        splits = get_cv_splits(cfg["cv_mat_path"], len(ds0), 10)[:args.folds]

        # Before: how often do shipped training bags contain a test-fold sample?
        raw = loadmat(cfg["data_path"])["bags"]
        key_raw = row_index(ds0.X_raw)
        tr0, te0 = splits[0]
        te0 = set(te0)
        leaky = sum(1 for i in tr0
                    if any(key_raw[np.asarray(r, dtype=np.float64).tobytes()] <= te0
                           for r in raw[i, 0][1:]))
        print(f"{name}: shipped bags, fold 1: {leaky}/{len(tr0)} training bags "
              f"contain a test-fold sample")

        for fi, (tr, te) in enumerate(splits):
            ds = copy.deepcopy(ds0)
            ds.normalize_features(tr)
            val, fit = fold_split(tr, fi)
            ds.prepare_fold(fit)
            X = ds.X.copy()
            key = row_index(X)
            fitset = set(fit.tolist())
            bad = 0
            for i in range(len(X)):
                b = ds.bags[i].numpy()
                if b.shape[0] != ds.k_neighbors + 1 or not np.array_equal(b[0], X[i]):
                    bad += 1
                    continue
                bad += sum(1 for r in b[1:] if not (key[r.tobytes()] & fitset))
            ds.normalize_features(tr)
            if not np.allclose(ds.X, X):
                bad += 1
            failures += bad
            print(f"  fold {fi + 1}: violations={bad}")
    print("PASS" if failures == 0 else f"FAIL ({failures} violations)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
