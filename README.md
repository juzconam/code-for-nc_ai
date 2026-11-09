# Code and derived records for *Better reconstructions can mean worse error bars in physics-informed field reconstruction*

This repository contains the analysis code and the small per-sample records needed to reproduce the main-text numbers, figures and tables of the manuscript. It is deliberately split into two levels:

1. **Analysis (CPU-only, no raw data, no checkpoints).** The per-sample derived quantities are included under `piv_runs/`, so every main-text number can be regenerated in minutes from `scripts/`.
2. **Full pipeline (GPU, optional).** The training/inference entry points used to produce those records are under `pipeline/`; they require the public PIV data and the released checkpoints (see `docs/FULL_PIPELINE.md` and `docs/CHECKPOINTS.md`).

The final Nature-styled composite figures are assembled from the panel-level outputs and tables in `paper_figures/uncertainty_study/`; the panel-to-script map is in `docs/FIGURE_MAP.md`.

## What is included

| Path | Contents |
| --- | --- |
| `scripts/` | Analysis: conformal intervals, width ratio `R`, coverage, paired tests, permutation checks, held-out domains, three routes, panel figures |
| `src/evaluation/` | Core implementations of split conformal prediction, interval scores and scientific quantities |
| `pipeline/` | Optional training/inference entry points for the main instance (GPU; checkpoints required) |
| `piv_runs/` | Per-sample derived records (small `.npz` files; **no raw images or checkpoints**) |
| `paper_figures/uncertainty_study/tables/` | Derived tables used by the figures, including the precomputed permutation and replication results |
| `splits/` | Fixed Class-I split manifest used for all comparisons |
| `docs/` | Figure map, full-pipeline instructions, checkpoint manifest, reference composites (`docs/figures/fig1-3.png`) |

## Quick start (CPU only)

```bash
python -m pip install -r requirements-analysis.txt

# 1. Main-instance tables and panel figures (Fig. 2, Supplementary Tables S1-S2, S5-S8)
python scripts/analyze_piv_uncertainty_study_en.py \
  --study-dir piv_runs/uncertainty_study \
  --output-dir paper_figures/uncertainty_study

# 2. Referee checks: Wilson coverage; the 2x2 and 1000-permutation blocks run when the
#    optional field cache is present, otherwise the precomputed JSON results are used.
python scripts/recompute_all.py

# 3. Held-out domains (Fig. 3)
python scripts/holdout_all.py
python scripts/plot_holdout_figure.py

# 4. Three routes (Table 1, upper block)
python scripts/analyse_three_routes.py

# 5. Panel-level main figures
python scripts/plot_main_figures.py
```

`recompute_all.py` and `permutation_test.py` print a `SKIP` line when the 42 MB `permutation_fields.npz` cache is absent; the precomputed null distribution is shipped as `paper_figures/uncertainty_study/tables/permutation_1000.json`. The full cache is available from the archive DOI if the permutation must be rerun from fields.

## What the main records contain

`piv_runs/uncertainty_study/study_test.npz` holds one row per test field (1,065 fields) and, for each intervention arm, the six scientific quantities for all candidates, the aggregated point estimate, the candidate spread, and the EPE. `study_validation.npz` (958 fields) and `study_purged.npz` (250 fields) are the calibration and purified sets. Held-out domains are in `piv_runs/holdout_mhd`, `holdout_channel` and `holdout_uniform`; the second independently trained gate checkpoint is in `piv_runs/replicate_agent_v2`.

The arrays are sufficient to recompute every interval, coverage value, width ratio and rank correlation in the paper without touching a network. They are **derived records**, not raw PIV images.

## Key definitions

- Split conformal quantile: the `ceil((n + 1)(1 - alpha))`-th order statistic, `alpha = 0.10` throughout.
- Width ratio: `R = q(|E|) / (E[s] q(|E|/s))`, where `s` is the candidate spread; `R > 1` means the spread-scaled interval is narrower than the constant-width interval.
- Candidate noise is fixed by SHA-256 of seed, sample identifier, direction and candidate index, so candidate sets are bit-for-bit reproducible.
- Gated and ungated arms share the same candidates; paired comparisons therefore isolate the gate.

## Full pipeline

The main instance uses:

- backbone checkpoint: `piv_runs/residual_ddpm_v6_x0_seed3407/diffusion_backbone_best.pth`
- agent checkpoint: `piv_runs/physics_unrolled_v3_warmup_scratch_20ep_seed3417/physics_agent_best.pth`
- split manifest: `splits/class1_primary_split_v2.json`

These files are not redistributed here. See `docs/CHECKPOINTS.md` for sizes and SHA-256 values, and `docs/FULL_PIPELINE.md` for the exact commands.

## Data

Raw PIV images and flow fields are public benchmark data and are not redistributed. `DATA_AVAILABILITY.md` lists the sources and the derived records included here.

## Environment

- Analysis: Python 3.9+, NumPy 2.0, SciPy, Matplotlib 3.9. No GPU required.
- Full pipeline: PyTorch 2.8 / torchvision 0.23 (see `environment.yml`), CUDA recommended.

## Citation and license

If you use this code, please cite the manuscript (see `CITATION.cff`). The code is released under the MIT License (see `LICENSE`).
