#!/usr/bin/env python3
"""Medical 2x2 isolation study (PREREGISTERED.md, Addendum A).

{shipped transductive bags, per-fold leak-free bags} x {legacy 1e-6 std clamp,
unit scale for constant-on-train features}, LCGA + BCE, one seed per process.
The (leak-free, unit) cell is the campaign itself and is not re-run here.

Usage: python scripts/dmkd_medical_isolation.py --seed 42
Writes results/dmkd_campaign/isolation/lcga_bce__<cond>__seed<s>.json
"""
import argparse
import copy
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from miml_clam.training.train_cv import run_cv  # noqa: E402
import run_baseline_experiments as rbe  # noqa: E402

OUT = os.path.join(ROOT, "results", "dmkd_campaign", "isolation")
CONDS = {
    "leaky_clamp": {"dataset_format": "standard", "legacy_std_clamp": True},
    "leaky_unit": {"dataset_format": "standard"},
    "free_clamp": {"dataset_format": "flat_knn", "legacy_std_clamp": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dataset", default="medical")
    ap.add_argument("--conds", default=",".join(CONDS))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    for cond in args.conds.split(","):
        over = CONDS[cond]
        tag = "" if args.dataset == "medical" else f"{args.dataset}__"
        path = os.path.join(OUT, f"lcga_bce__{tag}{cond}__seed{args.seed}.json")
        if os.path.exists(path):
            continue
        cfg = copy.deepcopy(rbe.DATASET_CONFIGS[args.dataset])
        cfg.update({"num_attn_experts": 1, "ensemble_seeds": [args.seed], **over})
        t0 = time.time()
        summary, per_fold = run_cv(cfg, device="cpu", num_folds=None,
                                   model_class=rbe.MODEL_REGISTRY["miml_lcga"])
        rec = {"dataset": args.dataset, "arch": "miml_lcga", "objective": "bce",
               "condition": cond, "seed": args.seed, "overrides": over,
               "summary": {k: [float(v[0]), float(v[1])] for k, v in summary.items()},
               "per_fold": [{k: float(v) for k, v in f.items()} for f in per_fold],
               "elapsed_s": round(time.time() - t0)}
        with open(path + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=2)
        os.replace(path + ".tmp", path)
        print(f"{cond} seed={args.seed} AP={summary['AveragePrecision'][0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
