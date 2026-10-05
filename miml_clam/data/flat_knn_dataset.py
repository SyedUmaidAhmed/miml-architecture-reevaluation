"""Leak-free k-NN bagging for flat multi-label datasets (Emotions, Medical, Yeast).

The shipped emotions.mat / medical.mat / yeast.mat were built by
MIML_CLAM_Enhanced/convert_new_datasets.py, which fits the scaler and the
nearest-neighbour index on the WHOLE dataset before any CV split. Every test
sample therefore sits inside several training bags, and every test bag contains
labelled training samples' neighbours chosen with the test rows in the index.
That is transductive leakage.

This loader recovers the flat feature matrix (instance 0 of every stored bag is
the sample itself; verified 2026-10-04: the stored bags are reproduced exactly
from it) and rebuilds the bags inside each fold:

  * standardisation statistics come from the fold's training rows only;
  * the neighbour pool is the fitting split only (training minus validation);
  * a fitting bag is the sample plus its k nearest OTHER fitting samples;
  * a validation or test bag is the sample plus its k nearest fitting samples.

No validation or test feature ever enters a fitting bag, and no bag sees a label
other than its own. Bags keep k+1 = 5 instances, as in the original framing.

Usage contract (shared by train_cv.run_fold and scripts/run_miml_knn.py):
    ds.normalize_features(train_idx)   # idempotent: always from the raw matrix
    ds.prepare_fold(fit_idx)           # rebuilds every bag; call before reading ds.bags
"""

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors

from .miml_dataset_cv import MIMLDatasetCV


class FlatKNNBagDataset(MIMLDatasetCV):

    def __init__(self, mat_path, bag_key='bags', label_key='labels', k_neighbors=4):
        # Deliberately skip MIMLDatasetCV.__init__: the stored bags are leaky and
        # must never reach a model, so only the flat matrix is kept.
        torch.utils.data.Dataset.__init__(self)
        mat = loadmat(mat_path)
        raw_bags = mat[bag_key]
        self.k_neighbors = int(k_neighbors)
        self.X_raw = np.stack([np.asarray(raw_bags[i, 0][0], dtype=np.float64)
                               for i in range(raw_bags.shape[0])])
        y = (np.asarray(mat[label_key]) > 0).astype(np.float32)
        self.labels = [torch.tensor(row) for row in y]
        self.num_labels = y.shape[1]
        self.input_dim = self.X_raw.shape[1]
        self.X = self.X_raw.astype(np.float32)
        # Singleton bags until prepare_fold runs; never the leaky stored bags.
        self.bags = [torch.tensor(self.X[i:i + 1]) for i in range(len(self.X))]
        self.num_instances = [1] * len(self.bags)
        self.fold_ready = False

    def normalize_features(self, train_indices):
        tr = self.X_raw[np.asarray(train_indices)]
        self.feat_mean = tr.mean(axis=0)
        # Same ddof as torch.std in the parent class.
        std = tr.std(axis=0, ddof=1)
        # Constant-on-train features get scale 1, as in MIMLDatasetCV.
        if getattr(self, 'legacy_std_clamp', False):
            # Pre-2026-10-04 behaviour, kept only for the Medical isolation study.
            self.feat_std = np.clip(std, 1e-6, None)
        else:
            self.feat_std = np.where(std < 1e-6, 1.0, std)
        self.X = ((self.X_raw - self.feat_mean) / self.feat_std).astype(np.float32)
        self.fold_ready = False

    def prepare_fold(self, fit_indices):
        fit = np.asarray(fit_indices)
        k = self.k_neighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(self.X[fit])
        _, nbr = nn.kneighbors(self.X)
        nbr = fit[nbr]                      # positions in the pool -> dataset indices
        in_fit = np.zeros(len(self.X), dtype=bool)
        in_fit[fit] = True
        bags = []
        for i in range(len(self.X)):
            cand = nbr[i]
            if in_fit[i]:
                # Exclude the sample itself; with exact duplicates it may not be
                # returned first, in which case drop the farthest candidate.
                cand = cand[cand != i][:k] if (cand == i).any() else cand[:k]
            else:
                cand = cand[:k]
            bags.append(torch.tensor(self.X[np.concatenate([[i], cand])]))
        self.bags = bags
        self.num_instances = [b.shape[0] for b in bags]
        self.fold_ready = True

    def __getitem__(self, idx):
        if not self.fold_ready:
            raise RuntimeError("FlatKNNBagDataset: call prepare_fold(fit_idx) "
                               "before reading bags")
        return self.bags[idx], self.labels[idx]
