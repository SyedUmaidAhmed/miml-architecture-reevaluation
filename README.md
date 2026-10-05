# How much does architecture buy in multi-instance multi-label learning?

Code, results and pre-registration for the article *How Much Does Architecture
Buy in Multi-Instance Multi-Label Learning? A Pre-Registered Re-Evaluation of
Attention, Training Objective and Ensembling* (Ahmed, Tahir, Waqas; under review).

The study crosses five bag aggregators (mean pooling, Transformer + mean
pooling, shared gated attention, label-conditional gated attention, independent
per-label attention branches) with two training objectives (cross-entropy;
ranking loss + asymmetric loss), trains every combination with five seeds on
thirteen MIML benchmarks, and compares them with three non-neural reference
learners on identical folds: 845 cross-validated runs, 7,475 trained networks.
No new method is proposed.

## Contents

| Path | What |
|---|---|
| `miml_clam/` | Model, training and metric code (all five architectures, the mixture-of-experts variant, the two-phase trainer, the five MIML metrics). |
| `miml_clam/data/flat_knn_dataset.py` | **Leak-free per-fold k-NN framing** of flat multi-label datasets (Emotions, Medical, Yeast). |
| `run_baseline_experiments.py` | Per-dataset configurations shared by every run. |
| `scripts/run_dmkd_campaign.py` | The campaign runner (resumable; several CPU workers can run in parallel). |
| `scripts/run_miml_knn.py` | Non-neural reference learners (MIML-kNN, ML-kNN, BR logistic regression). |
| `scripts/check_leakfree_bags.py` | Verifies that no held-out example enters any bag, on every fold. |
| `scripts/dmkd_medical_isolation.py` | The 2×2 study separating the leak from the scaling clamp. |
| `scripts/dmkd_analysis.py` | Pre-registered analyses A0–A6 → `results/dmkd_campaign/analysis.json`. |
| `scripts/dmkd_tables.py`, `dmkd_correction_table.py`, `dmkd_trivial_and_hl.py`, `dmkd_cd_figure.py` | Generate the tables, number macros and figure of the article (written to `paper_prai/`). |
| `results/dmkd_campaign/PREREGISTERED.md` | The pre-registration, its addendum and the deviations log. |
| `results/dmkd_campaign/*.json` | One file per (architecture, objective, seed): per-fold and summary metrics for all 13 datasets. |
| `results/dmkd_campaign/external/`, `isolation/` | Reference learners; isolation study. |
| `results/preliminary/` | Earlier (pre-study) result files whose values the article quotes. |

## Data and saved predictions

Two archives are attached to the GitHub release:

- `data.zip` — the thirteen benchmarks as `.mat` files with their fold partitions;
  unzip into `data/` (or set `MIML_DATA_DIR`). Original sources: Scene and Reuters
  (Zhou et al., AIJ 2012); MSCV2, Letter Frost, Letter Carroll (Briggs, Fern & Raich,
  KDD 2012); Birdsong (MLSP 2013 bird-song challenge) and the four protein sets
  (Wu et al., 2014) as distributed in the [kdis-lab MIML collection](https://github.com/kdis-lab/MIML);
  Yeast (Elisseeff & Weston, NIPS 2001), Emotions (Trohidis et al., ISMIR 2008),
  Medical (Pestian et al., 2007). They are redistributed here only for
  reproducibility; please cite the original sources.
  Note: the stored bags of Emotions, Medical and Yeast were built transductively;
  the code never uses them as bags (it recovers the flat features and rebuilds
  the bags inside each fold).
- `preds.zip` — validation and test probabilities of every run (needed only to
  recompute the 5-seed ensembles with `dmkd_analysis.py`); unzip into
  `results/dmkd_campaign/`.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/check_leakfree_bags.py                  # leak check, all folds
python scripts/run_dmkd_campaign.py --smoke            # 2-fold smoke test
# full campaign (CPU; start several workers, they claim units without overlap)
python scripts/run_dmkd_campaign.py --device cpu --threads 2
python scripts/run_miml_knn.py --leakfree --dataset all --out-dir results/dmkd_campaign/external
python scripts/dmkd_analysis.py                        # needs preds/ for the ensembles
python scripts/dmkd_tables.py && python scripts/dmkd_correction_table.py && python scripts/dmkd_trivial_and_hl.py && python scripts/dmkd_cd_figure.py
```

Running the last line on the released results reproduces every table, figure
and in-text number of the article exactly.

## Licence

Code: MIT (see `LICENSE`). Datasets remain under their original terms.
