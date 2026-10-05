# Pre-registration — DMKD campaign (objective × architecture × ensembling)

Written 2026-10-04, **before any unit of `scripts/run_dmkd_campaign.py` ran**
(only the 2-fold smoke test in `results/dmkd_campaign_smoke/`, which is never
analysed). Nothing below may change after the first real unit finishes.
Any later deviation is recorded at the bottom under *Deviations*, with its date
and reason, and reported in the manuscript.

## Why this campaign exists

1. The 2026-09 MoE controls (`results/moe_method/aggregate.json`) showed that,
   for MIML-LCGA, the apparent gain of mixture-of-experts attention came from the
   training objective (ranking loss + ASL), not from the experts. Whether that
   is a fact about MIML-LCGA or about MIML learning is unknown.
2. On 2026-10-04 the shipped `emotions.mat`, `medical.mat`, `yeast.mat` were
   found to be built with transductive k-NN bagging (scaler + neighbour index fit
   on the whole dataset before CV). In fold 1, 161/534, 211/881 and 694/2176
   training bags contain a test-fold sample (`scripts/check_leakfree_bags.py`).
   All three are rebuilt per fold by `miml_clam/data/flat_knn_dataset.py`
   (neighbours from the fitting split only); the check script passes on all 30
   folds. Every number in the new manuscript for these datasets comes from the
   leak-free construction.
3. Older baseline rows (`all_baselines_merged_v2.json`) used a global
   `torch.manual_seed(42)`, not per-fold seeds, and no baseline was ever
   ensembled. This campaign re-runs every architecture under one protocol.

4. Also found 2026-10-04, before any campaign run: every loader standardised
   features with `std.clamp(min=1e-6)`. On Medical (1,449 sparse bag-of-words
   features), up to 94 features per fold are constant on the training rows but
   non-zero on test rows, so ~9.5% of test instances carried values of order
   1e6 for every model. Loaders now give constant-on-train features scale 1
   (scikit-learn's convention). No other dataset has such features
   (checked on all folds of all 13), so only Medical changes.

## Fixed design

- **Datasets (13):** scene, reuters, mscv2, letter_frost, letter_carroll, yeast,
  birdsong, protein_haloarcula, protein_pyrococcus, protein_geobacter,
  protein_azotobacter, emotions, medical. Default fold count (10 for `.mat`
  datasets with `*_10CV.mat`, 5 for DD datasets). Emotions/Medical/Yeast use
  `dataset_format='flat_knn'`, k = 4 (bags of 5, as before).
- **Architectures:** `miml_lcga` (K=1 shared label-conditional attention),
  `abmil_ml`, `transformer_pool_ml`, `mlp_pool_ml` (the *core four*);
  `clam_multilabel_ml` (one independent gated-attention branch per label);
  MoE arm: `moe_k4` (K=4) and `lcga_attn128` (K=1, attn_dim=128, parameter-matched
  to K=4).
- **Objectives:** `bce` (published config: BCE + label smoothing 0.05);
  `rankasl` (`lambda_rank=0.5`, `use_asl=True` — the configuration promoted by the
  2026-08 validation-only screen; not re-tuned).
- **Seeds:** 42, 123, 456, 789, 1024. Each unit = one explicitly seeded single
  model per fold (`ensemble_seeds=[seed]`).
- **Hyper-parameters:** the per-dataset configs in
  `run_baseline_experiments.DATASET_CONFIGS`, unchanged, identical for every
  architecture. No per-dataset, per-model or per-objective tuning.
- **Grid:** core four × 2 objectives × 5 seeds (40 units); CLAM-ML × 2 × 5 (10);
  MoE arm: moe_k4 × {bce, rankasl} × 5 and lcga_attn128 × rankasl × 5 (15).
  65 units × 13 datasets. Queue order is in the runner. Every cell is run
  fresh for this campaign (no imports from earlier result files).
- **Device:** CPU (Apple M4 Pro), several workers in parallel, each claiming
  whole (architecture, objective, seed) units from the queue. A timing test
  showed CPU ~2.7× faster than the MPS GPU for these models (2 folds of
  Emotions: 29 s vs 78 s, AP 0.7587 vs 0.7589). Device and thread count are
  recorded in every cell.
- **External baselines:** `scripts/run_miml_knn.py --leakfree` re-run for
  emotions, medical, yeast (MIML-kNN, ML-kNN over Hausdorff distances, BR
  logistic regression on bag means). All eight datasets are re-run into
  `results/dmkd_campaign/external/`: the original script also normalised the same
  dataset object in place every fold, so fold k was scaled on top of earlier
  folds' statistics (which include fold k's test rows). Fixed 2026-10-04 by
  restoring the raw bags before each fold.
- **Outputs:** per-unit JSON (`summary`, `per_fold`) and per-(fold) val/test
  probability dumps in `results/dmkd_campaign/preds/`.

## Analysis plan (fixed)

Primary metric: **Average Precision** (threshold-independent). Ranking Loss,
Coverage, One-Error reported as secondary; Hamming Loss reported but never used
for a headline (it alone depends on the fitted thresholds). The **dataset is
the unit of analysis**: per (architecture, objective, dataset) we take the mean
over the 5 seeds of the CV-mean metric. The 65 dataset–metric cells are never
treated as independent trials and no "chance rate" win-count is a headline.

- **A1 — Objective effect, per architecture.** For each of the five
  architectures: paired Wilcoxon signed-rank over the 13 datasets of seed-mean AP,
  `rankasl` vs `bce`; Holm correction over the five. Effect size: median ΔAP and
  the number of datasets where |ΔAP| exceeds the across-seed sd of that cell.
- **A2 — Architecture effect, per objective.** Friedman test over 13 datasets
  on seed-mean AP of the five architectures (core four + CLAM-ML), separately for
  `bce` and `rankasl`. Average ranks and critical-difference diagrams; Nemenyi
  post-hoc only if Friedman rejects at 0.05.
- **A3 — Ensembling effect.** For every architecture × objective, the 5-seed
  ensemble (mean of test probabilities; for each seed the coupled-vs-raw head is
  chosen by that seed's validation AP, and thresholds are fitted on the averaged
  validation probabilities, exactly as `_run_fold_ensemble` does) vs the seed-mean
  single model: Wilcoxon over 13 datasets, Holm over the ten. Like-for-like:
  ensembles are only ever compared with ensembles, singles with singles.
- **A4 — Variance decomposition (headline quantity).** On the per-(dataset,
  architecture, objective, seed) AP values of the core four, a linear model with
  dataset fixed effects; report the share of the remaining sum of squares
  attributable to architecture, objective, architecture × objective, and seed
  (residual). Also report, per dataset, the architecture spread (max − min
  seed-mean AP over architectures, per objective), the objective gap, and the
  ensemble gain, side by side.
- **A5 — MoE arm.** 2 × 2 {K=1, K=4} × {bce, rankasl} plus the parameter-matched
  K=1 (attn_dim=128) under `rankasl`: Wilcoxon over 13 datasets on seed-mean AP
  for K4 vs K1 within each objective and for K4 vs attn128; Holm over the three.
- **A6 — External baselines.** Seed-mean single-model results of every neural
  architecture × objective vs MIML-kNN / ML-kNN / BR-bagmean on the eight datasets
  where they exist, per metric, both directions reported.

All tests two-sided, α = 0.05 after correction. Every cell is reported whatever
it shows; significant losses are reported with the same prominence as wins.

## Decision rules (fixed)

- **"The objective transfers across architectures"** may be claimed only if A1
  rejects (Holm) for at least 3 of the core four in the same direction.
  If it rejects for 0–1, the paper states that the objective effect is specific
  to MIML-LCGA and does not transfer. 2 of 4 is reported as "mixed".
- **"Objective matters more than architecture"** may be claimed only if, in A4,
  the objective share exceeds the architecture share **and** A1 meets the rule
  above. Otherwise the paper reports the shares as they are.
- **"Ensembling matters more than architecture"** may be claimed only if A3
  rejects for at least 3 of the core four (in at least one objective) **and** the
  median per-dataset ensemble gain exceeds the median per-dataset architecture
  spread.
- **"Architecture does not matter"** may never be claimed from a non-rejecting
  Friedman test alone; the paper says "no detectable difference at n = 13
  datasets" and reports the average ranks.
- **MoE:** experts may be credited only if A5's K4-vs-K1 test rejects under the
  same objective. Otherwise: "no measurable contribution of the experts".
- **Leak-free numbers replace the leaky ones everywhere.** The leaky results
  appear once, in a correction paragraph that gives the before/after for the
  three datasets.

## Deviations

- 2026-10-05 06:42 UTC (operational, not a design change): with 57/65 units
  complete and three units unclaimed, three extra CPU workers with one torch
  thread each were started so the tail ran in parallel. Grid, protocol and
  analysis unchanged; the thread count is recorded in every cell. A bug fix in
  the analysis script (a loop variable shadowing the A2 results dictionary in
  the Nemenyi post-hoc code) changed no pre-registered computation.

## Addendum A (2026-10-05, after the campaign; NOT part of the original pre-registration)

An internal review pointed out that the before/after comparison for Medical
changes two things at once (leak-free bags and the scaling fix), so the
manuscript's attribution of the change to the scaling defect was not isolated.
Isolation study, fixed before running: Medical only; 2×2 = {shipped transductive
bags, per-fold leak-free bags} × {legacy std clamp 1e-6, unit scale for
constant-on-train features}; LCGA with BCE at seeds 42/123/456/789/1024 plus the
three reference learners; everything else identical to the campaign. The
(leak-free, unit) cell reuses the campaign runs. Reported in full whatever it
shows; it can only change the attribution sentence, not any A1–A6 result.
Also added post hoc and labelled as such in the manuscript: ANOVA mean squares
and variance components for A4 (the pre-registered SS shares ignore degrees of
freedom), and a 5-seed-ensemble version of A6.

## Addendum B (2026-10-05, after the campaign; NOT part of the original pre-registration)

Added after a second internal review, reported in the manuscript as post hoc:
(a) the three reference learners run on the five benchmarks outside the
pre-registered eight (scene, reuters, mscv2, letter_frost, letter_carroll), so
A6 covers all thirteen, reported separately for native-bag and k-NN-framed
datasets; (b) trivial predictors (all-negative; training-fold label prior) and
Hamming Loss at a fixed 0.5 threshold recomputed from the saved probabilities
(scripts/dmkd_trivial_and_hl.py); (c) the Addendum-A isolation study was also
run, for the leaky-bag condition only, on Emotions and Yeast.
(d) Added after a third internal review (scripts/dmkd_robustness.py): percentile
bootstrap over datasets (2000 resamples) of the A4 shares for Average Precision
and the split of the seed term into seed main effect and seed-by-configuration
variation; A1 repeated on the ten datasets outside the objective screen; 95%
bootstrap intervals for the A5 mean differences.
