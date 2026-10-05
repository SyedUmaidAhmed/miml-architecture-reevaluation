#!/usr/bin/env python3
"""Pre-registered analysis A0-A6 for the DMKD campaign.

Implements results/dmkd_campaign/PREREGISTERED.md exactly; read it first.
Writes results/dmkd_campaign/analysis.json (every number the manuscript may
quote) and prints a human-readable summary. Incomplete cells are reported as
missing, never imputed; a test only runs on datasets complete for every arm it
compares, and the dataset count n is always stored next to the p-value.

  A0  sanity: single-seed metrics recomputed from the probability dumps must
      equal the per_fold metrics in the unit JSON (validates the ensemble path)
  A1  objective effect per architecture (Wilcoxon over datasets, Holm over archs)
  A2  architecture effect per objective (Friedman over datasets; average ranks)
  A3  5-seed ensemble vs seed-mean single model, per arch x objective (Holm)
  A4  variance decomposition on the core four (dataset fixed effects)
  A5  MoE arm (K4 vs K1 per objective; K4 vs param-matched K1), Holm over three
  A6  neural seed-mean singles vs external non-neural baselines, per metric

Usage:
    python scripts/dmkd_analysis.py [--dir results/dmkd_campaign] [--ext-dir results/dmkd_campaign/external]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from miml_clam.metrics.miml_metrics import (  # noqa: E402
    AllFive, AllFiveWithThresholds, AveragePrecision, optimize_thresholds)
import run_baseline_experiments as rbe  # noqa: E402

METRICS = ["HammingLoss", "OneError", "Coverage", "RankingLoss", "AveragePrecision"]
HIGHER_BETTER = {"AveragePrecision"}
CORE = ["miml_lcga", "abmil_ml", "transformer_pool_ml", "mlp_pool_ml"]
FIVE = CORE + ["clam_multilabel_ml"]
OBJS = ["bce", "rankasl"]
SEEDS = [42, 123, 456, 789, 1024]
DATASETS = ["scene", "reuters", "mscv2", "letter_frost", "letter_carroll",
            "yeast", "birdsong", "protein_haloarcula", "protein_pyrococcus",
            "protein_geobacter", "protein_azotobacter", "emotions", "medical"]


def holm(pvals):
    """Holm step-down adjusted p-values, same order as input (None passes through)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    adj = [None] * len(pvals)
    running = 0.0
    m = len(order)
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * pvals[i]))
        adj[i] = running
    return adj


def load_units(d):
    """cells[(arch, obj, seed, ds)] = cell dict (summary/per_fold/preds...)."""
    cells = {}
    for path in glob.glob(os.path.join(d, "*__seed*.json")):
        stem = os.path.basename(path)[:-5]
        unit, seed = stem.rsplit("__seed", 1)
        arch, obj = unit.split("__")
        for cell in json.load(open(path)).values():
            if isinstance(cell, dict) and "summary" in cell and "error" not in cell:
                cells[(arch, obj, int(seed), cell["dataset"])] = cell
    return cells


def seed_table(cells, arch, obj, metric):
    """{ds: (seed-mean, seed-sd, n_seeds)} over complete seeds only."""
    out = {}
    for ds in DATASETS:
        vals = [cells[(arch, obj, s, ds)]["summary"][metric][0]
                for s in SEEDS if (arch, obj, s, ds) in cells]
        if len(vals) == len(SEEDS):
            out[ds] = (float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals))
    return out


def paired_test(a, b, metric):
    """Wilcoxon over datasets present in both; returns dict with n, p, median diff
    oriented so that positive = a better."""
    common = [ds for ds in DATASETS if ds in a and ds in b]
    if len(common) < 6:
        return {"n": len(common), "p": None, "median_diff": None, "datasets": common}
    sign = 1.0 if metric in HIGHER_BETTER else -1.0
    diff = np.array([sign * (a[ds][0] - b[ds][0]) for ds in common])
    p = float(wilcoxon(diff, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
    return {"n": len(common), "p": p, "median_diff": float(np.median(diff)),
            "wins_a": int((diff > 0).sum()), "wins_b": int((diff < 0).sum()),
            "datasets": common}


# ---------------------------------------------------------------- A0 / A3 --
def fold_arrays(npz, fold):
    g = lambda k: npz[f"fold{fold}__{k}"]
    return (g("val_labels"), g("val_raw"), g("val_coupled"),
            g("test_labels"), g("test_raw"), g("test_coupled"))


def pick_head(vl, vr, vc, tr, tc):
    """Same rule as train_cv._run_fold_ensemble: coupled iff val AP >= raw."""
    if AveragePrecision(vl, vc) >= AveragePrecision(vl, vr):
        return vc, tc
    return vr, tr


def score(vl, vp, tl, tp, use_thr):
    if use_thr:
        return AllFiveWithThresholds(tl, tp, optimize_thresholds(vl, vp, metric="hamming"))
    return AllFive(tl, tp)


def ensemble_cell(cells, arch, obj, ds):
    """CV-mean metrics of the 5-seed ensemble, or None if any seed is missing."""
    keys = [(arch, obj, s, ds) for s in SEEDS]
    if not all(k in cells and cells[k].get("preds") for k in keys):
        return None
    npzs = [np.load(os.path.join(ROOT, cells[k]["preds"])) for k in keys]
    n_folds = cells[keys[0]]["num_folds"]
    use_thr = rbe.DATASET_CONFIGS[ds].get("use_threshold_opt", False)
    per_fold = []
    for f in range(n_folds):
        vps, tps = [], []
        for z in npzs:
            vl, vr, vc, tl, tr, tc = fold_arrays(z, f)
            vp, tp = pick_head(vl, vr, vc, tr, tc)
            vps.append(vp)
            tps.append(tp)
        per_fold.append(score(vl, np.mean(vps, 0), tl, np.mean(tps, 0), use_thr))
    return {m: float(np.mean([pf[m] for pf in per_fold])) for m in METRICS}


def a0_sanity(cells, limit=None):
    """Recompute single-seed per-fold metrics from dumps; return max |diff|."""
    worst, checked = 0.0, 0
    for (arch, obj, seed, ds), cell in list(cells.items())[:limit]:
        if not cell.get("preds"):
            continue
        z = np.load(os.path.join(ROOT, cell["preds"]))
        use_thr = rbe.DATASET_CONFIGS[ds].get("use_threshold_opt", False)
        for f, pf in enumerate(cell["per_fold"]):
            vl, vr, vc, tl, tr, tc = fold_arrays(z, f)
            vp, tp = pick_head(vl, vr, vc, tr, tc)
            m = score(vl, vp, tl, tp, use_thr)
            for k in METRICS:
                worst = max(worst, abs(m[k] - pf[k]))
        checked += 1
    return {"cells_checked": checked, "max_abs_diff": worst}


# ------------------------------------------------------------------- A4 ----
def variance_decomposition(cells, metric="AveragePrecision"):
    """Balanced 4 arch x 2 obj x 5 seed design within each complete dataset.
    Dataset fixed effects removed; SS split into arch, obj, arch:obj, seed."""
    ds_ok = [ds for ds in DATASETS
             if all((a, o, s, ds) in cells for a in CORE for o in OBJS for s in SEEDS)]
    if not ds_ok:
        return {"n_datasets": 0}
    ss = defaultdict(float)
    df = defaultdict(float)
    for ds in ds_ok:
        Y = np.array([[[cells[(a, o, s, ds)]["summary"][metric][0] for s in SEEDS]
                       for o in OBJS] for a in CORE])            # [A, O, S]
        mu = Y.mean()
        A, O, S = Y.shape
        a_eff = Y.mean(axis=(1, 2)) - mu
        o_eff = Y.mean(axis=(0, 2)) - mu
        ao = Y.mean(axis=2) - mu - a_eff[:, None] - o_eff[None, :]
        resid = Y - Y.mean(axis=2, keepdims=True)
        ss["architecture"] += O * S * float((a_eff ** 2).sum())
        ss["objective"] += A * S * float((o_eff ** 2).sum())
        ss["architecture_x_objective"] += S * float((ao ** 2).sum())
        ss["seed_residual"] += float((resid ** 2).sum())
        df["architecture"] += A - 1
        df["objective"] += O - 1
        df["architecture_x_objective"] += (A - 1) * (O - 1)
        df["seed_residual"] += A * O * (S - 1)
    total = sum(ss.values())
    # ADDITIONAL (not pre-registered; added 2026-10-05 after an internal review
    # pointed out that SS shares ignore degrees of freedom): ANOVA mean squares
    # pooled over datasets (datasets as blocks) and method-of-moments variance
    # components from the expected mean squares of the balanced crossed design,
    # negative estimates truncated at 0.
    A_, O_, S_ = len(CORE), len(OBJS), len(SEEDS)
    ms = {k: ss[k] / df[k] for k in ss if df[k] > 0}
    vc = {
        "seed_residual": ms["seed_residual"],
        "architecture_x_objective": max(0.0, (ms["architecture_x_objective"] - ms["seed_residual"]) / S_),
        "architecture": max(0.0, (ms["architecture"] - ms["architecture_x_objective"]) / (O_ * S_)),
        "objective": max(0.0, (ms["objective"] - ms["architecture_x_objective"]) / (A_ * S_)),
    }
    vtot = sum(vc.values())
    return {"n_datasets": len(ds_ok), "datasets": ds_ok, "metric": metric,
            "sum_sq": dict(ss), "df": dict(df), "mean_sq": ms,
            "share": {k: v / total for k, v in ss.items()} if total else {},
            "var_components": vc,
            "vc_share": {k: v / vtot for k, v in vc.items()} if vtot else {}}


def per_dataset_effects(cells, ens, metric="AveragePrecision"):
    rows = {}
    for ds in DATASETS:
        r = {}
        for o in OBJS:
            means = {a: seed_table(cells, a, o, metric).get(ds) for a in CORE}
            if all(means.values()):
                v = [m[0] for m in means.values()]
                r[f"arch_spread_{o}"] = max(v) - min(v)
        for a in CORE:
            b = seed_table(cells, a, "bce", metric).get(ds)
            k = seed_table(cells, a, "rankasl", metric).get(ds)
            if b and k:
                r.setdefault("objective_gap", {})[a] = k[0] - b[0]
            for o in OBJS:
                s = seed_table(cells, a, o, metric).get(ds)
                e = ens.get((a, o, ds))
                if s and e:
                    r.setdefault("ensemble_gain", {})[f"{a}/{o}"] = e[metric] - s[0]
        rows[ds] = r
    return rows


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "dmkd_campaign"))
    ap.add_argument("--ext-dir", default=os.path.join(ROOT, "results", "dmkd_campaign", "external"))
    ap.add_argument("--no-ensemble", action="store_true")
    args = ap.parse_args()

    cells = load_units(args.dir)
    out = {"n_cells": len(cells)}
    out["A0_sanity"] = a0_sanity(cells)

    # A1
    a1 = {}
    for metric in METRICS:
        tests = {a: paired_test(seed_table(cells, a, "rankasl", metric),
                                seed_table(cells, a, "bce", metric), metric) for a in FIVE}
        adj = holm([tests[a]["p"] for a in FIVE])
        for a, p in zip(FIVE, adj):
            tests[a]["p_holm"] = p
        a1[metric] = tests
    out["A1_objective"] = a1

    # A2
    a2 = {}
    for o in OBJS:
        for metric in METRICS:
            tabs = {a: seed_table(cells, a, o, metric) for a in FIVE}
            common = [ds for ds in DATASETS if all(ds in tabs[a] for a in FIVE)]
            res = {"n": len(common), "archs": FIVE}
            if len(common) >= 6:
                M = np.array([[tabs[a][ds][0] for a in FIVE] for ds in common])
                R = np.array([rankdata(-row if metric in HIGHER_BETTER else row) for row in M])
                res["avg_rank"] = dict(zip(FIVE, R.mean(0).round(4).tolist()))
                res["friedman_p"] = float(friedmanchisquare(*M.T).pvalue)
                if res["friedman_p"] < 0.05:
                    # Nemenyi post-hoc (pre-registered: only when Friedman rejects).
                    # CD = q_alpha * sqrt(k(k+1)/(6N)); q_0.05 from Demsar (2006), Table 5.
                    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}
                    k, N = len(FIVE), len(common)
                    cd = q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
                    ranks = dict(zip(FIVE, R.mean(0)))
                    pairs = []
                    for i, x1 in enumerate(FIVE):
                        for x2 in FIVE[i + 1:]:
                            if abs(ranks[x1] - ranks[x2]) > cd:
                                better, worse = (x1, x2) if ranks[x1] < ranks[x2] else (x2, x1)
                                pairs.append([better, worse, float(abs(ranks[x1] - ranks[x2]))])
                    res["nemenyi_cd"] = float(cd)
                    res["nemenyi_significant_pairs"] = pairs
            a2[f"{o}/{metric}"] = res
    out["A2_architecture"] = a2

    # A3
    ens = {}
    if not args.no_ensemble:
        for a in FIVE:
            for o in OBJS:
                for ds in DATASETS:
                    e = ensemble_cell(cells, a, o, ds)
                    if e:
                        ens[(a, o, ds)] = e
    a3 = {}
    for metric in METRICS:
        tests, keys = {}, []
        for a in FIVE:
            for o in OBJS:
                single = seed_table(cells, a, o, metric)
                etab = {ds: (ens[(a, o, ds)][metric], 0.0, 5)
                        for ds in DATASETS if (a, o, ds) in ens}
                tests[f"{a}/{o}"] = paired_test(etab, single, metric)
                keys.append(f"{a}/{o}")
        adj = holm([tests[k]["p"] for k in keys])
        for k, p in zip(keys, adj):
            tests[k]["p_holm"] = p
        a3[metric] = tests
    out["A3_ensemble"] = a3
    out["ensemble_cells"] = {f"{a}/{o}/{ds}": v for (a, o, ds), v in ens.items()}

    # A4
    out["A4_variance"] = {m: variance_decomposition(cells, m) for m in METRICS}
    out["A4_per_dataset"] = per_dataset_effects(cells, ens)

    # A5
    a5 = {}
    for metric in METRICS:
        t = {
            "K4_vs_K1_bce": paired_test(seed_table(cells, "moe_k4", "bce", metric),
                                        seed_table(cells, "miml_lcga", "bce", metric), metric),
            "K4_vs_K1_rankasl": paired_test(seed_table(cells, "moe_k4", "rankasl", metric),
                                            seed_table(cells, "miml_lcga", "rankasl", metric), metric),
            "K4_vs_attn128_rankasl": paired_test(seed_table(cells, "moe_k4", "rankasl", metric),
                                                 seed_table(cells, "lcga_attn128", "rankasl", metric), metric),
        }
        adj = holm([v["p"] for v in t.values()])
        for k, p in zip(t, adj):
            t[k]["p_holm"] = p
        a5[metric] = t
    out["A5_moe"] = a5

    # A6
    a6 = {}
    for path in glob.glob(os.path.join(args.ext_dir, "external_baselines_*.json")):
        for key, cell in json.load(open(path)).items():
            if not isinstance(cell, dict) or "summary" not in cell:
                continue
            ds, model = cell["dataset"], cell["model"]
            for metric in METRICS:
                ev = cell["summary"][metric][0]
                for a in FIVE:
                    for o in OBJS:
                        s = seed_table(cells, a, o, metric).get(ds)
                        if s:
                            better = (s[0] > ev) if metric in HIGHER_BETTER else (s[0] < ev)
                            a6.setdefault(ds, {}).setdefault(f"{a}/{o}", {}).setdefault(
                                model, {})[metric] = {"neural": s[0], "external": ev,
                                                      "neural_better": bool(better)}
    out["A6_external"] = a6

    path = os.path.join(args.dir, "analysis.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(path + ".tmp", path)

    # ---- summary -----------------------------------------------------------
    print(f"cells loaded: {len(cells)}   A0: {out['A0_sanity']}")
    m = "AveragePrecision"
    print("\nA1 objective (rank+ASL vs BCE), AP:")
    for a in FIVE:
        t = a1[m][a]
        print(f"  {a:<22} n={t['n']:>2} p={t['p']} p_holm={t.get('p_holm')} "
              f"median dAP={t['median_diff']}")
    print("\nA2 architecture, AP:")
    for o in OBJS:
        r = a2[f"{o}/{m}"]
        print(f"  {o:<8} n={r['n']:>2} friedman_p={r.get('friedman_p')} ranks={r.get('avg_rank')}")
    print("\nA3 ensemble vs single, AP:")
    for k, t in a3[m].items():
        print(f"  {k:<32} n={t['n']:>2} p_holm={t.get('p_holm')} median gain={t['median_diff']}")
    print("\nA4 variance shares, AP:", out["A4_variance"][m].get("share"),
          "n_datasets=", out["A4_variance"][m]["n_datasets"])
    print("\nA5 MoE, AP:")
    for k, t in a5[m].items():
        print(f"  {k:<24} n={t['n']:>2} p_holm={t.get('p_holm')} median dAP={t['median_diff']}")
    print(f"\nwritten {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
