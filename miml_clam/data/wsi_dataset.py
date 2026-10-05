"""Dataset loader for whole-slide-image (WSI) MIML.

Each whole-slide image is a "bag" of patch feature vectors pre-extracted by a
patch encoder (ResNet50 / UNI / CONCH ...). Features are stored one file per
slide, following the CLAM / UNI2-h convention:

    <slide_id>.h5
        - features : (n_patches, feat_dim) float32
        - coords   : (n_patches, 2)        int      (optional, unused here)

    <slide_id>.pt
        a feature tensor, or a dict with a 'features' key.

Slide-level multi-labels come from a CSV whose first column (or ``slide_id_col``)
is the slide id and whose remaining columns (or ``label_cols``) are 0/1 — or
{-1,+1} — label indicators:

    slide_id,grade_high,subtype_A,TP53_mut,...
    TCGA-XX-....,1,0,1,...

Unlike the .mat loaders, features are NOT held in memory: WSI bags are large
(thousands of patches), so each slide is lazy-loaded from disk in ``__getitem__``.
Only the slide-level label matrix (small) stays resident. The interface matches
``MIMLDatasetCV`` so ``train_cv.run_cv`` and ``CurriculumTrainer`` work unchanged;
batch with ``collate_fn_wsi`` (the standard padding collate).
"""

import os
import csv
import glob
from collections import Counter

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .miml_dataset_cv import collate_fn

# WSI bags are variable-length just like .mat bags — the standard padding
# collate (pads to batch max, mask: 1=padded, 0=real) works as-is.
collate_fn_wsi = collate_fn


def _load_label_csv(csv_path, slide_id_col, label_cols):
    """Parse the slide-level label CSV. Returns (slide_ids, labels, label_names)."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]

    id_idx = 0 if slide_id_col is None else header.index(slide_id_col)
    if label_cols is None:
        label_idx = [i for i in range(len(header)) if i != id_idx]
        label_names = [header[i] for i in label_idx]
    else:
        label_idx = [header.index(c) for c in label_cols]
        label_names = list(label_cols)

    slide_ids, labels = [], []
    for r in rows:
        slide_ids.append(r[id_idx])
        # Accept 0/1 or {-1,+1}: anything > 0 is a positive label.
        labels.append([1.0 if float(r[i]) > 0 else 0.0 for i in label_idx])
    return slide_ids, labels, label_names


class WSIDataset(Dataset):
    """Whole-slide-image MIML dataset over pre-extracted patch features.

    Args:
        feature_dir: directory of per-slide ``.h5`` / ``.pt`` feature files.
        label_csv: path to the slide-level multi-label CSV.
        slide_id_col: name of the slide-id column (default: first column).
        label_cols: names of the label columns (default: all but the id column).
        feature_key: dataset/key holding features inside ``.h5`` / ``.pt``.
        max_patches: if set, cap each bag to this many patches (random subsample
            in train mode, evenly-spaced in eval mode). ``None`` uses all patches.
    """

    def __init__(self, feature_dir, label_csv, slide_id_col=None, label_cols=None,
                 feature_key='features', max_patches=None, file_ext=None):
        super().__init__()
        if not os.path.isdir(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
        self.feature_dir = feature_dir
        self.feature_key = feature_key
        self.max_patches = max_patches
        self.training = True            # toggles random vs deterministic subsample
        self._normalized = False
        self.feat_mean = None
        self.feat_std = None

        slide_ids, labels, self.label_names = _load_label_csv(
            label_csv, slide_id_col, label_cols)

        # Resolve each slide_id to a feature file; drop slides with no file.
        self.slide_ids, self.slide_paths, self.labels = [], [], []
        for sid, lab in zip(slide_ids, labels):
            path = self._resolve_path(sid, file_ext)
            if path is None:
                continue
            self.slide_ids.append(sid)
            self.slide_paths.append(path)
            self.labels.append(torch.tensor(lab, dtype=torch.float32))

        if not self.slide_paths:
            raise RuntimeError(
                f"No feature files matched the {len(slide_ids)} slide IDs from "
                f"{label_csv} under {feature_dir}")
        n_missing = len(slide_ids) - len(self.slide_paths)
        if n_missing:
            print(f"[WSIDataset] {n_missing}/{len(slide_ids)} slides had no "
                  f"feature file and were dropped.", flush=True)

        self.num_labels = len(self.label_names)
        self.input_dim = self._load_raw_features(self.slide_paths[0]).shape[1]

    # ------------------------------------------------------------------ I/O
    def _resolve_path(self, slide_id, file_ext):
        """Map a slide id to a feature file path, probing known extensions."""
        base = os.path.join(self.feature_dir, slide_id)
        candidates = []
        if file_ext:
            candidates.append(base + file_ext)
        candidates += [base + '.h5', base + '.pt',
                       os.path.join(self.feature_dir, slide_id)]
        for c in candidates:
            if os.path.isfile(c):
                return c
        hits = glob.glob(base + '.*')
        return hits[0] if hits else None

    def _load_raw_features(self, path):
        """Load a slide's patch features as a (n_patches, feat_dim) float tensor."""
        if path.endswith('.pt'):
            obj = torch.load(path, map_location='cpu')
            feats = obj['features'] if isinstance(obj, dict) else obj
            return torch.as_tensor(feats, dtype=torch.float32)
        try:
            import h5py
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "Reading .h5 WSI features requires h5py — `pip install h5py`.") from e
        with h5py.File(path, 'r') as f:
            key = self.feature_key if self.feature_key in f else 'features'
            feats = f[key][:]
        return torch.as_tensor(feats, dtype=torch.float32)

    # ------------------------------------------------------------- sampling
    def _subsample(self, feats):
        n = feats.shape[0]
        if self.max_patches is None or n <= self.max_patches:
            return feats
        if self.training:
            sel = torch.randperm(n)[:self.max_patches]
        else:
            sel = torch.linspace(0, n - 1, self.max_patches).round().long()
        return feats[sel]

    def normalize_features(self, train_indices, max_patches_for_stats=512):
        """Streaming per-feature standardization from training slides only.

        Computes mean/std in one disk pass (sampling patches per slide to bound
        cost) and applies them on the fly in ``__getitem__``.
        """
        n = 0
        s = torch.zeros(self.input_dim, dtype=torch.float64)
        sq = torch.zeros(self.input_dim, dtype=torch.float64)
        for i in train_indices:
            feats = self._load_raw_features(self.slide_paths[i])
            if feats.shape[0] > max_patches_for_stats:
                sel = torch.randperm(feats.shape[0])[:max_patches_for_stats]
                feats = feats[sel]
            f64 = feats.double()
            s += f64.sum(0)
            sq += (f64 * f64).sum(0)
            n += f64.shape[0]
        mean = s / max(n, 1)
        var = (sq / max(n, 1)) - mean * mean
        self.feat_mean = mean.float()
        self.feat_std = var.clamp(min=1e-12).sqrt().float().clamp(min=1e-6)
        self._normalized = True

    # ------------------------------------------------------------ Dataset API
    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        feats = self._subsample(self._load_raw_features(self.slide_paths[idx]))
        if self._normalized:
            feats = (feats - self.feat_mean) / self.feat_std
        return feats, self.labels[idx]

    def train(self):
        """Random patch subsampling (training)."""
        self.training = True
        return self

    def eval(self):
        """Deterministic patch subsampling (validation / test)."""
        self.training = False
        return self

    # ---------------------------------------- interface parity with .mat loaders
    def get_label_cooccurrence(self, indices=None):
        """Label co-occurrence matrix C[i,j] = P(label_j=1 | label_i=1)."""
        if indices is None:
            indices = range(len(self))
        labels = torch.stack([self.labels[i] for i in indices])
        co = labels.T @ labels
        counts = labels.sum(dim=0).clamp(min=1)
        co = co / counts.unsqueeze(1)
        co.fill_diagonal_(0)
        return co

    def get_balanced_sampler(self, indices):
        """WeightedRandomSampler balancing label-combination frequency."""
        labels = torch.stack([self.labels[i] for i in indices])
        combo = [''.join(str(int(x)) for x in row) for row in labels.numpy()]
        counts = Counter(combo)
        weights = torch.tensor([1.0 / counts[s] for s in combo], dtype=torch.float64)
        return WeightedRandomSampler(weights, num_samples=len(weights),
                                     replacement=True)
