#!/usr/bin/env python3
"""External, non-neural baselines for the eight datasets that currently lack one.

Why this exists
---------------
The manuscript compares against published results on five datasets, but for
Yeast, Birdsong, the four protein-function sets, Emotions and Medical it
compares "exclusively against our neural ablation baselines" — i.e. against
ablations of the proposed model. Eight of the thirteen headline wins come from
exactly those eight datasets. A reviewer will observe that MIML-kNN is a short
lazy learner that could have been run on all of them.

Three baselines, all deliberately simple:

  miml_knn    MIML-kNN (Zhang 2010, "A k-nearest neighbor based multi-instance
              multi-label learning algorithm"). Bag distance is the average
              Hausdorff distance; each query's neighbourhood is the union of its
              k nearest REFERENCES and the training bags that CITE it; the
              label-count vector over that neighbourhood is mapped to per-label
              scores by a least-squares weight matrix fitted on the training
              bags. This is the citer/reference algorithm the MIML literature
              means by "MIML-kNN".

  mlknn_hausdorff  ML-kNN (Zhang & Zhou 2007) Bayesian posteriors over the same
              bag-level Hausdorff distances — the simpler multi-label-kNN
              degeneration, kept because it is the other thing a reviewer may
              mean. Note these are two DIFFERENT papers and algorithms; do not
              cite Zhang & Zhou 2007 for the MIML-kNN row.

  br_bagmean  Binary relevance: one logistic regression per label over
              bag-mean features. The obvious "did you try the simple thing"
              control.

All produce real-valued per-label scores, so all five ranking metrics are
computable, not just Hamming Loss.

Pairing. These use the SAME cross-validation folds AND the same 90/10
train/validation split as every neural experiment — the split is reconstructed
exactly as train_cv.py:run_fold builds it (np.random.seed(fold_idx), shuffle,
first 10% held out). That matters: an earlier version held out an unshuffled 20%
prefix, which trained the external baselines on a different and smaller subset
than the neural models they are compared against (~73% overlap), quietly
handicapping the baseline. k is selected on validation, never on test.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from miml_clam.data import MIMLDatasetCV, MIMLDatasetDD, FlatKNNBagDataset, get_cv_splits  # noqa: E402
from miml_clam.metrics.miml_metrics import AllFive, optimize_thresholds  # noqa: E402
from miml_clam.metrics.miml_metrics import AllFiveWithThresholds  # noqa: E402
import run_baseline_experiments as rbe  # noqa: E402

K_GRID = [3, 5, 7, 9, 11, 15, 21]
SMOOTH = 1.0  # Laplace smoothing, as in the original ML-kNN


def average_hausdorff(A, B):
    """Average Hausdorff distance between two instance sets (reference version).

    avgH(A,B) = [ sum_a min_b d(a,b) + sum_b min_a d(a,b) ] / (|A| + |B|)

    Kept as the readable definition and as the oracle the fast path is tested
    against; ``bag_distance_matrix`` computes the same thing in bulk.
    """
    d = np.sqrt(np.maximum(
        ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T), 0.0))
    return (d.min(1).sum() + d.min(0).sum()) / (len(A) + len(B))


def _stack(bags):
    """Concatenate instances and record where each bag starts."""
    X = np.ascontiguousarray(np.concatenate(bags, axis=0), dtype=np.float64)
    sizes = np.array([len(b) for b in bags])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return X, starts, sizes


def bag_distance_matrix(bags_a, bags_b, chunk_bytes=256 << 20):
    """Average Hausdorff distances for every (a, b) pair, in bulk.

    The pairwise Python loop this replaces costs one small NumPy call per bag
    pair, which dominates everything else at these bag sizes (a few instances
    each) -- millions of calls per fold. Here every instance is stacked into one
    matrix, the instance-by-instance distances are computed with a single BLAS
    matmul per row chunk, and the per-bag min/sum reductions are done with
    ``reduceat`` over the bag boundaries. Same arithmetic, ~1-2 orders of
    magnitude faster.
    """
    Xa, sa, na = _stack(bags_a)
    Xb, sb, nb = _stack(bags_b)
    sq_a, sq_b = (Xa ** 2).sum(1), (Xb ** 2).sum(1)

    T_ab = np.zeros((len(bags_a), len(bags_b)))   # sum_a min_b
    T_ba = np.zeros((len(bags_b), len(bags_a)))   # sum_b min_a

    rows = max(1, int(chunk_bytes / max(1, 8 * len(Xb))))
    for lo in range(0, len(Xa), rows):
        hi = min(lo + rows, len(Xa))
        d = sq_a[lo:hi, None] + sq_b[None, :] - 2.0 * (Xa[lo:hi] @ Xb.T)
        np.maximum(d, 0.0, out=d)
        np.sqrt(d, out=d)

        # min over the instances of each bag in b -> (chunk, n_bags_b)
        min_over_b = np.minimum.reduceat(d, sb, axis=1)
        # min over the instances of each bag in a, restricted to this row chunk
        a_starts = sa[(sa >= lo) & (sa < hi)] - lo
        if a_starts.size:
            min_over_a = np.minimum.reduceat(d, a_starts, axis=0)
            first = int(np.searchsorted(sa, lo, side='left'))
            # sum over the instances of each bag in b -> contributes to T_ba
            T_ba[:, first:first + len(a_starts)] += np.add.reduceat(
                min_over_a, sb, axis=1).T
            # sum over the instances of each bag in a -> contributes to T_ab
            T_ab[first:first + len(a_starts), :] += np.add.reduceat(
                min_over_b, a_starts, axis=0)

    denom = na[:, None] + nb[None, :]
    return (T_ab + T_ba.T) / denom


def mimlknn_neighbourhood(D_query_ref, D_ref_ref, k, k_citers):
    """Reference + citer neighbourhood indicator, Wang & Zucker's citation rule.

    references: the k training bags nearest the query.
    citers:     training bags j whose own k_citers-nearest training neighbours
                would include the query, i.e. d(j, query) is closer than j's
                k_citers-th nearest training bag.

    Only query FEATURES are used; no query label is touched.
    """
    n_q, n_ref = D_query_ref.shape
    ind = np.zeros((n_q, n_ref), dtype=np.float64)

    # references
    ref_idx = np.argpartition(D_query_ref, min(k, n_ref - 1), axis=1)[:, :k]
    np.add.at(ind, (np.repeat(np.arange(n_q), ref_idx.shape[1]), ref_idx.ravel()), 1.0)

    # citers: threshold per training bag = its k_citers-th nearest OTHER training bag
    Dm = D_ref_ref.copy()
    np.fill_diagonal(Dm, np.inf)
    kc = min(k_citers, n_ref - 1)
    radius = np.partition(Dm, kc - 1, axis=1)[:, kc - 1]      # (n_ref,)
    ind += (D_query_ref <= radius[None, :]).astype(np.float64)
    return ind


def mimlknn_fit(D_fit, Y_fit, k, k_citers, ridge=1e-6):
    """Least-squares map from label-count vectors to label scores (Zhang 2010).

    Each training bag's counting vector is built leave-one-out, then W solves
    min_W || Phi W - Y ||^2 with a small ridge for conditioning.
    """
    ind = mimlknn_neighbourhood(D_fit, D_fit, k, k_citers)
    np.fill_diagonal(ind, 0.0)                    # leave-one-out
    Phi = ind @ Y_fit                             # (n, L) label counts
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    W = np.linalg.solve(A, Phi.T @ Y_fit)
    return W


def mimlknn_predict(D_query_fit, D_fit, Y_fit, W, k, k_citers):
    ind = mimlknn_neighbourhood(D_query_fit, D_fit, k, k_citers)
    return (ind @ Y_fit) @ W


def mlknn_fit(D_tr, Y_tr, k):
    """Prior and likelihood tables from the training set (leave-one-out)."""
    n, L = Y_tr.shape
    prior = (SMOOTH + Y_tr.sum(0)) / (2 * SMOOTH + n)
    # exclude self by masking the diagonal
    Dm = D_tr.copy()
    np.fill_diagonal(Dm, np.inf)
    nn = np.argsort(Dm, axis=1)[:, :k]
    counts = Y_tr[nn].sum(1).astype(int)            # (n, L) neighbours carrying label l
    c1 = np.zeros((L, k + 1))
    c0 = np.zeros((L, k + 1))
    for i in range(n):
        for l in range(L):
            (c1 if Y_tr[i, l] == 1 else c0)[l, counts[i, l]] += 1
    lik1 = (SMOOTH + c1) / (SMOOTH * (k + 1) + c1.sum(1, keepdims=True))
    lik0 = (SMOOTH + c0) / (SMOOTH * (k + 1) + c0.sum(1, keepdims=True))
    return prior, lik1, lik0


def mlknn_predict(D_te_tr, Y_tr, prior, lik1, lik0, k):
    nn = np.argsort(D_te_tr, axis=1)[:, :k]
    counts = Y_tr[nn].sum(1).astype(int)
    L = Y_tr.shape[1]
    scores = np.empty((len(D_te_tr), L))
    for l in range(L):
        p1 = prior[l] * lik1[l, counts[:, l]]
        p0 = (1 - prior[l]) * lik0[l, counts[:, l]]
        scores[:, l] = p1 / np.maximum(p1 + p0, 1e-12)
    return scores


def br_bagmean(bags_tr, Y_tr, bags_te):
    from sklearn.linear_model import LogisticRegression
    Xtr = np.stack([b.mean(0) for b in bags_tr])
    Xte = np.stack([b.mean(0) for b in bags_te])
    out = np.zeros((len(Xte), Y_tr.shape[1]))
    for l in range(Y_tr.shape[1]):
        y = Y_tr[:, l]
        if y.min() == y.max():          # degenerate label in this fold
            out[:, l] = float(y[0])
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, y)
        out[:, l] = clf.predict_proba(Xte)[:, 1]
    return out


# Flat multi-label datasets whose shipped .mat bags were built transductively
# (see miml_clam/data/flat_knn_dataset.py). --leakfree rebuilds them per fold.
KNN_BAGGED = {"emotions", "medical", "yeast"}
LEAKFREE = False
LEGACY_CLAMP = False


def load_dataset(name):
    """Mirror run_cv's dispatch exactly, so folds and preprocessing match."""
    cfg = dict(rbe.DATASET_CONFIGS[name])
    if LEAKFREE and name in KNN_BAGGED:
        cfg['dataset_format'] = 'flat_knn'
    if cfg.get('dataset_format') == 'dd':
        ds = MIMLDatasetDD(mat_path=cfg['data_path'])
    elif cfg.get('dataset_format') == 'flat_knn':
        ds = FlatKNNBagDataset(mat_path=cfg['data_path'],
                               bag_key=cfg.get('bag_key', 'bags'),
                               label_key=cfg.get('label_key', 'labels'),
                               k_neighbors=cfg.get('knn_bag_k', 4))
    else:
        ds = MIMLDatasetCV(mat_path=cfg['data_path'],
                           bag_key=cfg.get('bag_key', 'bags'),
                           label_key=cfg.get('label_key', 'labels'))
    if LEGACY_CLAMP:
        ds.legacy_std_clamp = True
    return ds, cfg


def build_splits(ds, cfg, num_folds=None):
    """Identical fold construction to run_cv, including the np.random.seed(42)
    fallback for DD datasets that ship no 10CV index file."""
    if cfg.get('cv_mat_path'):
        splits = get_cv_splits(cfg['cv_mat_path'], len(ds), 10)
    else:
        k = num_folds or cfg.get('default_folds', 5)
        idx = np.arange(len(ds))
        np.random.seed(42)
        np.random.shuffle(idx)
        fold = len(ds) // k
        splits = [(np.concatenate([idx[:i * fold], idx[(i + 1) * fold:]]).tolist(),
                   idx[i * fold:(i + 1) * fold].tolist()) for i in range(k)]
    return splits[:num_folds] if num_folds is not None else splits


def run_dataset(name, out_dir, num_folds=None):
    ds, cfg = load_dataset(name)
    splits = build_splits(ds, cfg, num_folds)
    n_folds = len(splits)
    print(f"\n=== {name}: {len(ds.bags)} bags, L={ds.num_labels}, {n_folds} folds ===")

    per_fold = {"miml_knn": [], "mlknn_hausdorff": [], "br_bagmean": []}
    chosen = {"miml_knn": [], "mlknn_hausdorff": []}

    # normalize_features mutates ds.bags in place; restore the raw bags before
    # every fold so each fold is scaled from its own training rows only (as
    # train_cv.run_fold does via deepcopy). FlatKNNBagDataset always restarts
    # from its raw matrix, so it needs no restore.
    raw_bags = [b.clone() if hasattr(b, "clone") else np.array(b, copy=True) for b in ds.bags]
    for fi, (train_idx, test_idx) in enumerate(splits):
        t0 = time.time()
        ds.bags = [b.clone() if hasattr(b, "clone") else np.array(b, copy=True) for b in raw_bags]
        # Mirror train_cv.run_fold exactly: normalise on the full fold-train, then
        # seed with the fold index, shuffle, and hold out the first 10%. Matching
        # this is what makes the comparison paired -- the external baselines must
        # see the same training rows as the neural models.
        ds.normalize_features(train_idx)
        tr = np.array(train_idx)
        np.random.seed(fi)
        np.random.shuffle(tr)
        val_size = max(len(tr) // 10, 1)
        val_idx, fit_idx = tr[:val_size].tolist(), tr[val_size:].tolist()
        if hasattr(ds, "prepare_fold"):
            ds.prepare_fold(fit_idx)

        bags = [b.numpy() if hasattr(b, "numpy") else np.asarray(b) for b in ds.bags]
        Y = np.stack([(l.numpy() if hasattr(l, "numpy") else np.asarray(l))
                      for l in ds.labels])
        Y = (Y > 0).astype(int)

        B_fit = [bags[i] for i in fit_idx]
        Y_fit = Y[fit_idx]
        Y_val, Y_te = Y[val_idx], Y[test_idx]

        D_fit = bag_distance_matrix(B_fit, B_fit)
        D_val = bag_distance_matrix([bags[i] for i in val_idx], B_fit)
        D_te = bag_distance_matrix([bags[i] for i in test_idx], B_fit)
        grid = [k for k in K_GRID if k < len(B_fit)] or [1]

        # --- MIML-kNN (references + citers, least-squares scores) -----------
        best = (-np.inf, grid[0])
        for k in grid:
            W = mimlknn_fit(D_fit, Y_fit, k, k)
            s = mimlknn_predict(D_val, D_fit, Y_fit, W, k, k)
            ap = AllFive(Y_val, s)["AveragePrecision"]
            if ap > best[0]:
                best = (ap, k)
        k_miml = best[1]
        chosen["miml_knn"].append(k_miml)
        W = mimlknn_fit(D_fit, Y_fit, k_miml, k_miml)
        s_val = mimlknn_predict(D_val, D_fit, Y_fit, W, k_miml, k_miml)
        s_te = mimlknn_predict(D_te, D_fit, Y_fit, W, k_miml, k_miml)
        th = optimize_thresholds(Y_val, s_val, metric="hamming")
        per_fold["miml_knn"].append(AllFiveWithThresholds(Y_te, s_te, th))

        # --- ML-kNN Bayesian posteriors over the same distances -------------
        best = (-np.inf, grid[0])
        for k in grid:
            prior, l1, l0 = mlknn_fit(D_fit, Y_fit, k)
            s = mlknn_predict(D_val, Y_fit, prior, l1, l0, k)
            ap = AllFive(Y_val, s)["AveragePrecision"]
            if ap > best[0]:
                best = (ap, k)
        k_ml = best[1]
        chosen["mlknn_hausdorff"].append(k_ml)
        prior, l1, l0 = mlknn_fit(D_fit, Y_fit, k_ml)
        s_val = mlknn_predict(D_val, Y_fit, prior, l1, l0, k_ml)
        s_te = mlknn_predict(D_te, Y_fit, prior, l1, l0, k_ml)
        th = optimize_thresholds(Y_val, s_val, metric="hamming")
        per_fold["mlknn_hausdorff"].append(AllFiveWithThresholds(Y_te, s_te, th))

        # --- binary relevance over bag-mean features ------------------------
        b_val = br_bagmean(B_fit, Y_fit, [bags[i] for i in val_idx])
        b_te = br_bagmean(B_fit, Y_fit, [bags[i] for i in test_idx])
        th_b = optimize_thresholds(Y_val, b_val, metric="hamming")
        per_fold["br_bagmean"].append(AllFiveWithThresholds(Y_te, b_te, th_b))

        print(f"  fold {fi}: k={k_miml}/{k_ml}  "
              f"MIMLkNN AP={per_fold['miml_knn'][-1]['AveragePrecision']:.4f}  "
              f"MLkNN AP={per_fold['mlknn_hausdorff'][-1]['AveragePrecision']:.4f}  "
              f"BR AP={per_fold['br_bagmean'][-1]['AveragePrecision']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for model, folds in per_fold.items():
        keys = folds[0].keys()
        entry = {
            "model": model, "dataset": name, "mode": "single",
            "num_folds": n_folds,
            "summary": {k: [float(np.mean([f[k] for f in folds])),
                            float(np.std([f[k] for f in folds]))] for k in keys},
            "per_fold": folds,
        }
        if model in chosen:
            entry["selected_k_per_fold"] = chosen[model]
        payload[f"{name}/{model}/single"] = entry
    path = os.path.join(out_dir, f"external_baselines_{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  -> {path}")
    return payload


ORPHAN_DATASETS = ["yeast", "birdsong", "protein_haloarcula", "protein_pyrococcus",
                   "protein_geobacter", "protein_azotobacter", "emotions", "medical"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    help="'all' = the eight datasets with no external baseline")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--leakfree", action="store_true",
                    help="rebuild k-NN bags per fold for emotions/medical/yeast")
    ap.add_argument("--legacy-clamp", action="store_true",
                    help="old 1e-6 std clamp (Medical isolation study only)")
    args = ap.parse_args()
    global LEAKFREE, LEGACY_CLAMP
    LEAKFREE = args.leakfree
    LEGACY_CLAMP = args.legacy_clamp

    names = ORPHAN_DATASETS if args.dataset == "all" else args.dataset.split(",")
    for name in names:
        try:
            run_dataset(name, args.out_dir, args.folds)
        except Exception as exc:  # keep going; one dataset failing is not fatal
            print(f"  !! {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
