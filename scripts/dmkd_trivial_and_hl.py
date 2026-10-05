#!/usr/bin/env python3
"""Trivial reference predictors and fixed-threshold Hamming Loss (post hoc).

Added 2026-10-05 after an internal referee review (not pre-registered).

1. Trivial predictors on the same outer folds as every other result:
   * all-negative: predicts no label for any bag (Hamming Loss = label density
     of the test fold);
   * label prior: scores every bag with the label frequencies of the outer
     training fold (a single fixed ranking), thresholded at 0.5 for Hamming Loss.
2. Hamming Loss of every neural run at a fixed 0.5 threshold, recomputed from
   the saved test probabilities with the same head-selection rule as training
   (coupled vs raw head by validation AP), next to the reported Hamming Loss
   with thresholds fitted on the validation split.

Writes results/dmkd_campaign/trivial_hl.json, paper_prai/tables/gen_trivial.tex
and paper_prai/tables/gen_trivial_numbers.tex.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dmkd_analysis as A  # noqa: E402
import run_baseline_experiments as rbe  # noqa: E402
from miml_clam.data import MIMLDatasetCV, MIMLDatasetDD, FlatKNNBagDataset, get_cv_splits  # noqa: E402
from miml_clam.metrics.miml_metrics import AllFive, AveragePrecision  # noqa: E402

CAMP = os.path.join(ROOT, "results", "dmkd_campaign")
DSN = {"scene": "Scene", "reuters": "Reuters", "mscv2": "MSCV2", "letter_frost": "Letter Frost",
       "letter_carroll": "Letter Carroll", "yeast": "Yeast", "birdsong": "Birdsong",
       "protein_haloarcula": "Prot.\\ Haloarcula", "protein_pyrococcus": "Prot.\\ Pyrococcus",
       "protein_geobacter": "Prot.\\ Geobacter", "protein_azotobacter": "Prot.\\ Azotobacter",
       "emotions": "Emotions", "medical": "Medical"}


def labels_and_splits(ds):
    cfg = rbe.DATASET_CONFIGS[ds]
    if ds in ("emotions", "medical", "yeast"):
        d = FlatKNNBagDataset(cfg["data_path"])
    elif cfg.get("dataset_format") == "dd":
        d = MIMLDatasetDD(mat_path=cfg["data_path"])
    else:
        d = MIMLDatasetCV(mat_path=cfg["data_path"])
    Y = np.stack([np.asarray(l) for l in d.labels]).astype(int)
    if cfg.get("cv_mat_path"):
        splits = get_cv_splits(cfg["cv_mat_path"], len(Y), 10)
    else:
        idx = np.arange(len(Y))
        np.random.seed(42)
        np.random.shuffle(idx)
        k = 5
        f = len(Y) // k
        splits = [(np.concatenate([idx[:i * f], idx[(i + 1) * f:]]).tolist(),
                   idx[i * f:(i + 1) * f].tolist()) for i in range(k)]
    return Y, splits


def trivial(ds):
    Y, splits = labels_and_splits(ds)
    neg, prior = [], []
    for tr, te in splits:
        Yte = Y[te]
        freq = Y[tr].mean(0)
        neg.append(float(Yte.mean()))                      # HL of predicting nothing
        scores = np.tile(freq, (len(te), 1)) + 1e-9 * np.arange(Y.shape[1])[None, :]
        m = AllFive(Yte, scores)
        prior.append({k: float(m[k]) for k in A.METRICS})
    return {"allneg_HL": float(np.mean(neg)),
            "prior": {k: float(np.mean([p[k] for p in prior])) for k in A.METRICS}}


def hl_fixed(cell):
    z = np.load(os.path.join(ROOT, cell["preds"]))
    vals = []
    for f in range(cell["num_folds"]):
        vl, vr, vc, tl, tr, tc = A.fold_arrays(z, f)
        _, tp = A.pick_head(vl, vr, vc, tr, tc)
        vals.append(float(np.mean((tp > 0.5).astype(int) != tl)))
    return float(np.mean(vals))


def main():
    cells = A.load_units(CAMP)
    out = {}
    for ds in A.DATASETS:
        t = trivial(ds)
        best_fit = min(A.seed_table(cells, a, o, "HammingLoss")[ds][0] for a in A.FIVE for o in A.OBJS)
        fixed = {}
        for a in A.FIVE:
            for o in A.OBJS:
                fixed[f"{a}/{o}"] = float(np.mean([hl_fixed(cells[(a, o, s, ds)]) for s in A.SEEDS]))
        best_ap = max(A.seed_table(cells, a, o, "AveragePrecision")[ds][0] for a in A.FIVE for o in A.OBJS)
        out[ds] = {**t, "best_neural_HL_fitted": best_fit, "best_neural_HL_fixed05": min(fixed.values()),
                   "neural_HL_fixed05": fixed, "best_neural_AP": best_ap}
        print(ds, round(t["allneg_HL"], 3), round(best_fit, 3), round(min(fixed.values()), 3),
              "| prior AP", round(t["prior"]["AveragePrecision"], 3), "best neural AP", round(best_ap, 3))
    json.dump(out, open(os.path.join(CAMP, "trivial_hl.json"), "w"), indent=2)

    rows, n_fit_worse, n_fix_worse, names_fit = [], 0, 0, []
    for ds in A.DATASETS:
        r = out[ds]
        fw = r["best_neural_HL_fitted"] > r["allneg_HL"]
        xw = r["best_neural_HL_fixed05"] > r["allneg_HL"]
        n_fit_worse += fw
        n_fix_worse += xw
        if fw:
            names_fit.append(DSN[ds])
        b = lambda v, w: f"\\textit{{{v:.3f}}}" if w else f"{v:.3f}"
        rows.append(f"{DSN[ds]} & {r['allneg_HL']:.3f} & {b(r['best_neural_HL_fitted'], fw)} & "
                    f"{b(r['best_neural_HL_fixed05'], xw)} & {r['prior']['AveragePrecision']:.3f} & "
                    f"{r['best_neural_AP']:.3f} \\\\")
    tab = r"""%% GENERATED by scripts/dmkd_trivial_and_hl.py; do not edit numbers by hand.
\begin{table}[t]
\centering
\caption{Trivial reference predictors on the same folds (added after the campaign). Hamming Loss of predicting no label (all-negative) against the best neural configuration (seed-mean single model), with thresholds fitted on the validation split as in the rest of the paper and with a fixed threshold of 0.5; italics mark values worse than predicting nothing. Average Precision of ranking labels by their training-fold frequency (label prior) against the best neural configuration.}
\label{tab:trivial}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l r rr rr@{}}
\toprule
 & \multicolumn{3}{c}{Hamming Loss} & \multicolumn{2}{c}{Average Precision} \\
\cmidrule(lr){2-4}\cmidrule(l){5-6}
Dataset & all-negative & best neural (fitted) & best neural (0.5) & label prior & best neural \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(ROOT, "paper_prai", "tables", "gen_trivial.tex"), "w").write(tab)
    mac = [f"\\newcommand{{\\dkTrivFitWorseN}}{{{n_fit_worse}}}",
           f"\\newcommand{{\\dkTrivFixWorseN}}{{{n_fix_worse}}}",
           f"\\newcommand{{\\dkTrivFitWorseNames}}{{{', '.join(names_fit) or 'none'}}}"]
    open(os.path.join(ROOT, "paper_prai", "tables", "gen_trivial_numbers.tex"), "w").write(
        "%% GENERATED by scripts/dmkd_trivial_and_hl.py\n" + "\n".join(mac) + "\n")
    print("fitted-threshold worse than all-negative:", n_fit_worse, names_fit, "| fixed 0.5 worse:", n_fix_worse)


if __name__ == "__main__":
    main()
