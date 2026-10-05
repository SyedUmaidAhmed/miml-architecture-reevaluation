#!/usr/bin/env python3
"""Robustness analyses added after the second referee round (post hoc).

1. A4 uncertainty: percentile bootstrap over datasets (2000 resamples) of the
   pooled sum-of-squares shares and variance-component shares for Average
   Precision, and the split of the within-configuration seed term into the seed
   main effect (4 df per dataset) and seed-by-factor interactions.
2. A1 on the ten datasets outside the pre-study objective screen (Scene, MSCV2
   and Protein Pyrococcus excluded): Wilcoxon p, Holm over the five architectures.
3. A5 interval: 95% bootstrap interval (over datasets) of the mean paired
   difference in seed-mean Average Precision, LCGA-MoE minus LCGA, per objective,
   and LCGA-MoE minus LCGA-wide.

Writes results/dmkd_campaign/robustness.json and paper_prai/tables/gen_robust_numbers.tex.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import dmkd_analysis as A  # noqa: E402

CAMP = os.path.join(ROOT, "results", "dmkd_campaign")
RNG = np.random.default_rng(20261005)
B = 2000


def per_dataset_ss(cells, metric):
    """Per dataset: SS and df for arch, obj, arch:obj, seed main, seed interactions."""
    out = {}
    for ds in A.DATASETS:
        Y = np.array([[[cells[(a, o, s, ds)]["summary"][metric][0] for s in A.SEEDS]
                       for o in A.OBJS] for a in A.CORE])
        Aa, O, S = Y.shape
        mu = Y.mean()
        a_e = Y.mean((1, 2)) - mu
        o_e = Y.mean((0, 2)) - mu
        s_e = Y.mean((0, 1)) - mu
        ao = Y.mean(2) - mu - a_e[:, None] - o_e[None, :]
        within = Y - Y.mean(2, keepdims=True)
        ss = {"arch": O * S * (a_e ** 2).sum(), "obj": Aa * S * (o_e ** 2).sum(),
              "int": S * (ao ** 2).sum(), "seed_total": (within ** 2).sum(),
              "seed_main": Aa * O * (s_e ** 2).sum()}
        ss["seed_inter"] = ss["seed_total"] - ss["seed_main"]
        out[ds] = ss
    return out


def shares(ss_by_ds, dss):
    Aa, O, S = len(A.CORE), len(A.OBJS), len(A.SEEDS)
    tot = {k: sum(ss_by_ds[d][k] for d in dss) for k in ("arch", "obj", "int", "seed_total")}
    n = len(dss)
    df = {"arch": (Aa - 1) * n, "obj": (O - 1) * n, "int": (Aa - 1) * (O - 1) * n,
          "seed_total": Aa * O * (S - 1) * n}
    ssum = sum(tot.values())
    ss_share = {k: v / ssum for k, v in tot.items()}
    ms = {k: tot[k] / df[k] for k in tot}
    vc = {"seed_total": ms["seed_total"],
          "int": max(0.0, (ms["int"] - ms["seed_total"]) / S),
          "arch": max(0.0, (ms["arch"] - ms["int"]) / (O * S)),
          "obj": max(0.0, (ms["obj"] - ms["int"]) / (Aa * S))}
    vsum = sum(vc.values())
    return ss_share, {k: v / vsum for k, v in vc.items()}


def main():
    cells = A.load_units(CAMP)
    out = {}
    ss = per_dataset_ss(cells, "AveragePrecision")
    dss = list(A.DATASETS)
    point_ss, point_vc = shares(ss, dss)
    seed_main_frac = sum(ss[d]["seed_main"] for d in dss) / sum(ss[d]["seed_total"] for d in dss)
    boots_ss, boots_vc = {k: [] for k in point_ss}, {k: [] for k in point_vc}
    for _ in range(B):
        samp = list(RNG.choice(dss, size=len(dss), replace=True))
        s1, v1 = shares(ss, samp)
        for k in s1:
            boots_ss[k].append(s1[k])
        for k in v1:
            boots_vc[k].append(v1[k])
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out["A4_AP"] = {"ss_share": point_ss, "vc_share": point_vc,
                    "ss_ci": {k: ci(v) for k, v in boots_ss.items()},
                    "vc_ci": {k: ci(v) for k, v in boots_vc.items()},
                    "seed_main_fraction_of_seed_term": float(seed_main_frac)}
    # A1 on ten unscreened datasets
    screen = {"scene", "mscv2", "protein_pyrococcus"}
    ps, info = [], {}
    for a in A.FIVE:
        b = A.seed_table(cells, a, "bce", "AveragePrecision")
        r = A.seed_table(cells, a, "rankasl", "AveragePrecision")
        diff = np.array([r[d][0] - b[d][0] for d in A.DATASETS if d not in screen])
        p = float(wilcoxon(diff).pvalue)
        ps.append(p)
        info[a] = {"n": len(diff), "wins": int((diff > 0).sum()), "p": p}
    adj = A.holm(ps)
    for a, p in zip(A.FIVE, adj):
        info[a]["p_holm"] = p
    out["A1_unscreened"] = info
    # A5 intervals
    a5 = {}
    for name, (x, ox, y, oy) in {"bce": ("moe_k4", "bce", "miml_lcga", "bce"),
                                 "rank": ("moe_k4", "rankasl", "miml_lcga", "rankasl"),
                                 "wide": ("moe_k4", "rankasl", "lcga_attn128", "rankasl")}.items():
        tx = A.seed_table(cells, x, ox, "AveragePrecision")
        ty = A.seed_table(cells, y, oy, "AveragePrecision")
        d = np.array([tx[k][0] - ty[k][0] for k in A.DATASETS])
        bm = [RNG.choice(d, size=len(d), replace=True).mean() for _ in range(B)]
        a5[name] = {"mean": float(d.mean()), "ci": ci(bm)}
    out["A5_ci"] = a5
    json.dump(out, open(os.path.join(CAMP, "robustness.json"), "w"), indent=2)

    pct = lambda v: f"{100 * v:.0f}"
    m = []
    for k, t in (("seed_total", "Seed"), ("arch", "Arch"), ("obj", "Obj")):
        m.append(f"\\newcommand{{\\dkBootVc{t}}}{{{pct(point_vc[k])}}}")
        lo, hi = out["A4_AP"]["vc_ci"][k]
        m.append(f"\\newcommand{{\\dkBootVc{t}CI}}{{{pct(lo)}--{pct(hi)}}}")
        lo, hi = out["A4_AP"]["ss_ci"][k]
        m.append(f"\\newcommand{{\\dkBootSs{t}CI}}{{{pct(lo)}--{pct(hi)}}}")
    m.append(f"\\newcommand{{\\dkSeedMainFrac}}{{{pct(seed_main_frac)}}}")
    tag = {"miml_lcga": "Lcga", "abmil_ml": "Abmil", "transformer_pool_ml": "Tpool",
           "mlp_pool_ml": "Mlp", "clam_multilabel_ml": "Clam"}
    nrej = sum(1 for a in A.CORE if info[a]["p_holm"] < 0.05)
    m.append(f"\\newcommand{{\\dkUnscrNRejCore}}{{{nrej}}}")
    m.append(f"\\newcommand{{\\dkUnscrPmax}}{{{max(info[a]['p_holm'] for a in A.FIVE):.3f}}}")
    for a in A.FIVE:
        m.append(f"\\newcommand{{\\dkUnscrWin{tag[a]}}}{{{info[a]['wins']}}}")
    for k, t in (("bce", "Bce"), ("rank", "Rank"), ("wide", "Wide")):
        m.append(f"\\newcommand{{\\dkMoeMean{t}}}{{{a5[k]['mean']:.4f}}}")
        m.append(f"\\newcommand{{\\dkMoeCI{t}}}{{[{a5[k]['ci'][0]:.4f},\\ {a5[k]['ci'][1]:.4f}]}}")
    open(os.path.join(ROOT, "paper_prai", "tables", "gen_robust_numbers.tex"), "w").write(
        "%% GENERATED by scripts/dmkd_robustness.py\n" + "\n".join(m) + "\n")
    print(json.dumps({"vc": point_vc, "vc_ci": out["A4_AP"]["vc_ci"], "ss_ci": out["A4_AP"]["ss_ci"],
                      "seed_main_frac": seed_main_frac, "A1_unscreened": info, "A5": a5}, indent=1))


if __name__ == "__main__":
    main()
