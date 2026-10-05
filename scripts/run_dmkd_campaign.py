#!/usr/bin/env python3
"""Objective-transfer x architecture campaign for the DMKD re-evaluation.

Written 2026-10-04. The design, grid, seeds and decision rules are fixed in
results/dmkd_campaign/PREREGISTERED.md, written before any run of this script.
Read it before changing anything here.

Question: on MIML benchmarks, does the training objective (BCE vs ranking loss +
ASL) or multi-seed ensembling matter more than the attention architecture? The
2026-09 MoE controls showed that, for MIML-LCGA, the objective -- not the
experts -- produced the apparent gain. This campaign tests whether that holds
across architectures, under one protocol:

  * every (model, objective, seed) unit is an explicitly seeded single model
    per fold (a one-element ensemble_seeds list), all 13 benchmarks, default
    fold count, no per-dataset or per-model tuning;
  * emotions / medical / yeast are rebuilt per fold with leak-free k-NN
    bagging (miml_clam/data/flat_knn_dataset.py); the shipped .mat bags were
    built transductively and are not used;
  * per-(fold, seed) validation and test probabilities are dumped so the
    5-seed ensemble of EVERY model can be scored offline, like-for-like.

Every unit writes its own JSON under results/dmkd_campaign/ after each dataset,
so a crash costs one dataset and re-running resumes.

Usage:
    python scripts/run_dmkd_campaign.py --list
    python scripts/run_dmkd_campaign.py --smoke      # 2 folds of emotions per config
    python scripts/run_dmkd_campaign.py --threads 2  # one worker; start several
                                                     # in parallel, they claim
                                                     # units without overlap
"""
import argparse
import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from miml_clam.training.train_cv import run_cv  # noqa: E402
import run_baseline_experiments as rbe  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "dmkd_campaign")
MOE_DIR = os.path.join(ROOT, "results", "moe_method")
# 2026-10-04 (before any real run): imports disabled. CPU training proved ~2.7x
# faster than MPS for these models, so the MoE arm is re-run fresh and every
# cell of the campaign comes from the same device and code path.
IMPORT_MOE = False

DATASETS = ["scene", "reuters", "mscv2", "letter_frost", "letter_carroll",
            "yeast", "birdsong", "protein_haloarcula", "protein_pyrococcus",
            "protein_geobacter", "protein_azotobacter", "emotions", "medical"]
KNN_BAGGED = {"emotions", "medical", "yeast"}
SEEDS = [42, 123, 456, 789, 1024]

OBJECTIVES = {
    "bce": {},
    "rankasl": {"lambda_rank": 0.5, "use_asl": True},
}

# architecture name -> (MODEL_REGISTRY key, config overrides)
ARCHS = {
    "miml_lcga":          ("miml_lcga", {"num_attn_experts": 1}),
    "abmil_ml":           ("abmil_ml", {}),
    "transformer_pool_ml": ("transformer_pool_ml", {}),
    "mlp_pool_ml":        ("mlp_pool_ml", {}),
    "clam_multilabel_ml": ("clam_multilabel_ml", {}),
    # MoE arm (MIML-LCGA variants), carried over from the 2026-09 controls
    "moe_k4":             ("miml_lcga", {"num_attn_experts": 4}),
    "lcga_attn128":       ("miml_lcga", {"num_attn_experts": 1, "attn_dim": 128}),
}

CORE = ["miml_lcga", "abmil_ml", "transformer_pool_ml", "mlp_pool_ml"]

# The 2026-09 MoE-arm JSONs used the identical code path and protocol; their
# cells on the 10 natively bag-structured datasets are imported (with a
# provenance tag) instead of re-run. Their emotions/medical/yeast cells used the
# leaky bags and are never imported.
MOE_IMPORT = {
    ("moe_k4", "bce"): "moe_k4_bce",
    ("moe_k4", "rankasl"): "moe_k4_rankasl",
    ("lcga_attn128", "rankasl"): "k1_attn128_rankasl",
}

# Priority order (fixed in PREREGISTERED.md):
#   block 1: core 4 architectures x 2 objectives, seeds 42/123/456   (24 units)
#   block 2: same, seeds 789/1024                                   (16 units)
#   block 3: independent per-label attention (CLAM-ML) x 2 obj x 5  (10 units)
#   block 4: MoE arm: K=4 x 2 obj, param-matched K=1 x rank+ASL     (15 units)
QUEUE = (
    [(a, o, s) for s in SEEDS[:3] for a in CORE for o in OBJECTIVES]
    + [(a, o, s) for s in SEEDS[3:] for a in CORE for o in OBJECTIVES]
    + [("clam_multilabel_ml", o, s) for s in SEEDS for o in OBJECTIVES]
    + [(a, o, s) for s in SEEDS
       for a, o in (("moe_k4", "bce"), ("moe_k4", "rankasl"), ("lcga_attn128", "rankasl"))]
)


def unit_name(arch, obj):
    return f"{arch}__{obj}"


def out_path(arch, obj, seed):
    return os.path.join(OUT_DIR, f"{unit_name(arch, obj)}__seed{seed}.json")


def probs_path(arch, obj, seed, ds):
    return os.path.join(OUT_DIR, "preds", f"{ds}__{unit_name(arch, obj)}__seed{seed}.npz")


def log(msg):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {msg}", flush=True)


class ProbCollector:
    def __init__(self):
        self.records = []

    def __call__(self, payload):
        self.records.append(payload)

    def save(self, path):
        flat = {}
        for rec in self.records:
            tag = f"fold{rec['fold']}"
            for key in ("val_labels", "val_raw", "val_coupled",
                        "test_labels", "test_raw", "test_coupled"):
                flat[f"{tag}__{key}"] = rec[key].astype(np.float32)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **flat)
        os.replace(tmp, path)


def build_config(ds, arch, obj, seed):
    cfg = copy.deepcopy(rbe.DATASET_CONFIGS[ds])
    cfg.update(ARCHS[arch][1])
    cfg.update(OBJECTIVES[obj])
    cfg["ensemble_seeds"] = [seed]
    if ds in KNN_BAGGED:
        cfg["dataset_format"] = "flat_knn"
    return cfg


def save_json(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=2)
    os.replace(tmp, path)


def try_import(arch, obj, seed, ds):
    """Return a cell from results/moe_method if it is importable, else None."""
    src_name = MOE_IMPORT.get((arch, obj))
    if src_name is None or ds in KNN_BAGGED:
        return None
    src = os.path.join(MOE_DIR, f"{src_name}_seed{seed}.json")
    if seed == 42 and src_name == "moe_k4_rankasl":
        src = os.path.join(ROOT, "results", "promoted_config_test.json")
    if not os.path.exists(src):
        return None
    data = json.load(open(src))
    for key, cell in data.items():
        if (isinstance(cell, dict) and cell.get("dataset") == ds
                and "error" not in cell and "summary" in cell):
            out = dict(cell)
            out["_imported_from"] = f"{os.path.relpath(src, ROOT)}::{key}"
            return out
    return None


def run_unit(arch, obj, seed, device, datasets, num_folds=None, out_dir=None):
    path = out_path(arch, obj, seed) if out_dir is None else os.path.join(
        out_dir, f"{unit_name(arch, obj)}__seed{seed}.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    model_key, _ = ARCHS[arch]
    model_class = rbe.MODEL_REGISTRY[model_key]
    for ds in datasets:
        key = f"{ds}/{unit_name(arch, obj)}/single"
        if key in results and "error" not in results[key]:
            continue
        imported = try_import(arch, obj, seed, ds) if (out_dir is None and IMPORT_MOE) else None
        if imported is not None:
            results[key] = imported
            log(f"{unit_name(arch, obj)} seed={seed} {ds}: imported "
                f"({imported['_imported_from']})")
            save_json(path, results)
            continue
        cfg = build_config(ds, arch, obj, seed)
        collector = ProbCollector()
        t0 = time.time()
        try:
            summary, per_fold = run_cv(cfg, device=device, num_folds=num_folds,
                                       model_class=model_class, prob_sink=collector)
            ppath = (probs_path(arch, obj, seed, ds) if out_dir is None else
                     os.path.join(out_dir, "preds",
                                  f"{ds}__{unit_name(arch, obj)}__seed{seed}.npz"))
            collector.save(ppath)
            results[key] = {
                "arch": arch, "model_class": model_key, "objective": obj,
                "dataset": ds, "mode": "single", "seed": seed,
                "dataset_format": cfg.get("dataset_format", "standard"),
                "device": str(device), "torch_threads": torch.get_num_threads(),
                "config_overrides": {**ARCHS[arch][1], **OBJECTIVES[obj],
                                     "ensemble_seeds": [seed]},
                "num_folds": len(per_fold),
                "summary": {k: [float(v[0]), float(v[1])] for k, v in summary.items()},
                "per_fold": [{k: float(v) for k, v in f.items()} for f in per_fold],
                "preds": os.path.relpath(ppath, ROOT),
                "elapsed_s": round(time.time() - t0),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            log(f"{unit_name(arch, obj)} seed={seed} {ds}: "
                f"AP={summary['AveragePrecision'][0]:.4f} ({time.time()-t0:.0f}s)")
        except Exception as exc:  # keep the queue alive; record the failure
            results[key] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"{unit_name(arch, obj)} seed={seed} {ds}: FAILED "
                f"{type(exc).__name__}: {exc}")
        save_json(path, results)


def unit_complete(arch, obj, seed):
    p = out_path(arch, obj, seed)
    if not os.path.exists(p):
        return False
    d = json.load(open(p))
    return all(f"{ds}/{unit_name(arch, obj)}/single" in d
               and "error" not in d[f"{ds}/{unit_name(arch, obj)}/single"]
               for ds in DATASETS)


def claim_path(arch, obj, seed):
    return os.path.join(OUT_DIR, "claims", f"{unit_name(arch, obj)}__seed{seed}.claim")


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim(arch, obj, seed):
    """Atomically claim a unit for this worker; reclaim it if its owner died."""
    path = claim_path(arch, obj, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = int(open(path).read().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and _alive(owner):
                return False
            try:
                os.remove(path)          # stale claim from a dead worker
            except FileNotFoundError:
                pass
    return False


def release(arch, obj, seed):
    try:
        os.remove(claim_path(arch, obj, seed))
    except FileNotFoundError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 folds of emotions + mscv2 per architecture/objective, "
                         "written to results/dmkd_campaign_smoke/")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; ~2.7x faster than mps for these models)")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch intra-op threads for this worker")
    args = ap.parse_args()

    device = args.device

    if args.list:
        total = done_all = 0
        for arch, obj, seed in QUEUE:
            p = out_path(arch, obj, seed)
            done = 0
            if os.path.exists(p):
                d = json.load(open(p))
                done = sum(1 for v in d.values() if "error" not in v)
            total += 1
            done_all += done == len(DATASETS)
            print(f"{unit_name(arch, obj):<32} seed={seed:<5} {done:>2}/{len(DATASETS)}")
        print(f"complete units: {done_all}/{total}")
        return 0

    if args.smoke:
        smoke_dir = os.path.join(ROOT, "results", "dmkd_campaign_smoke")
        shutil.rmtree(smoke_dir, ignore_errors=True)
        for arch in ARCHS:
            for obj in OBJECTIVES:
                t0 = time.time()
                run_unit(arch, obj, 42, device, ["emotions", "mscv2"],
                         num_folds=2, out_dir=smoke_dir)
                log(f"SMOKE {unit_name(arch, obj)} done in {time.time()-t0:.0f}s")
        log("smoke finished")
        return 0

    if args.threads:
        torch.set_num_threads(args.threads)
    log(f"worker start pid={os.getpid()} device={device} "
        f"threads={torch.get_num_threads()} units={len(QUEUE)}")
    for arch, obj, seed in QUEUE:
        if unit_complete(arch, obj, seed):
            continue
        if not claim(arch, obj, seed):
            continue
        try:
            log(f"=== {unit_name(arch, obj)} seed={seed} ===")
            run_unit(arch, obj, seed, device, DATASETS)
        finally:
            release(arch, obj, seed)
    log("worker finished (no unclaimed incomplete units left)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
