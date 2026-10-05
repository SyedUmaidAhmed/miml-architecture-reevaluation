#!/usr/bin/env python3
"""Critical-difference diagrams for A2 (Demsar 2006 style) from analysis.json.

Panels: Average Precision and Coverage, under BCE and Rank+ASL. Each panel
shows the average rank of the five architectures over the 13 datasets and a
bar of length CD (Nemenyi, alpha=0.05); groups of architectures whose ranks
differ by less than CD are joined by a thick line. The Friedman p-value is in
each panel title; post-hoc groups are drawn only where Friedman rejects.

Writes paper_dmkd/figures/cd_diagrams.pdf (TrueType fonts).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = json.load(open(os.path.join(ROOT, "results", "dmkd_campaign", "analysis.json")))
NAMES = {"miml_lcga": "LCGA", "abmil_ml": "ABMIL-ML", "transformer_pool_ml": "Transformer-Pool",
         "mlp_pool_ml": "MLP-Pool", "clam_multilabel_ml": "CLAM-ML"}
Q05_K5 = 2.728


def panel(ax, key, title):
    r = AN["A2_architecture"][key]
    ranks = r["avg_rank"]
    n, k = r["n"], len(ranks)
    cd = Q05_K5 * np.sqrt(k * (k + 1) / (6.0 * n))
    order = sorted(ranks, key=ranks.get)
    ax.set_xlim(1, k)
    ax.set_ylim(-0.05, 1.12)
    ax.invert_xaxis()
    ax.axhline(0.8, color="black", lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [0.8, 0.83], color="black", lw=1)
        ax.text(t, 0.86, str(t), ha="center", va="bottom", fontsize=7)
    for i, a in enumerate(order):
        x = ranks[a]
        y = 0.55 - 0.12 * i
        ax.plot([x, x], [0.8, y], color="black", lw=0.7)
        side = "right" if x > (k + 1) / 2 else "left"
        xt = k if side == "right" else 1
        ax.plot([x, xt], [y, y], color="black", lw=0.7)
        ax.text(xt + (0.05 if side == "right" else -0.05), y,
                f"{NAMES[a]} ({x:.2f})", ha=side, va="center", fontsize=7)
    ax.plot([k, k - cd], [1.0, 1.0], color="black", lw=1.2)
    ax.text(k - cd / 2, 1.02, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    p = r["friedman_p"]
    if p < 0.05:
        # join maximal groups not significantly different
        xs = [ranks[a] for a in order]
        lvl = 0
        for i in range(len(xs)):
            j = max(jj for jj in range(i, len(xs)) if xs[jj] - xs[i] < cd)
            if j > i and not any(i >= a0 and j <= b0 for a0, b0 in getattr(panel, "_done", [])):
                ax.plot([xs[i] - 0.03, xs[j] + 0.03], [0.75 - 0.05 * lvl] * 2, color="black", lw=2.2, solid_capstyle="butt")
                lvl += 1
                panel._done = getattr(panel, "_done", []) + [(i, j)]
        panel._done = []
    ax.set_title(f"{title}  (Friedman p = {p:.3f})", fontsize=8, pad=2)
    ax.axis("off")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.0))
    panel(axes[0, 0], "bce/AveragePrecision", "Average Precision, BCE")
    panel(axes[0, 1], "rankasl/AveragePrecision", "Average Precision, Rank+ASL")
    panel(axes[1, 0], "bce/Coverage", "Coverage, BCE")
    panel(axes[1, 1], "rankasl/Coverage", "Coverage, Rank+ASL")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper_dmkd", "figures", "cd_diagrams.pdf")
    fig.savefig(out)
    print("written", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
