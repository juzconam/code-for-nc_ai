# Full pipeline (optional)

The CPU-only analysis in `scripts/` is sufficient to reproduce every main-text number from the included per-sample records. This document describes how those records were produced, for readers who want to rerun the network.

## 1. Requirements

- Python 3.9, PyTorch 2.8, torchvision 0.23, NumPy 2.0, SciPy, Matplotlib (see `environment.yml`).
- A CUDA-capable GPU is recommended but not required for a small smoke test.
- The public PIV data (not redistributed) and the checkpoints in `docs/CHECKPOINTS.md`.

## 2. Data layout

`--data-path` must point to a directory with one subdirectory per domain:

```text
sample_data/
  DNS_turbulence/
  JHTDB_channel/
  SQG/
  backstep/
  cylinder/
  JHTDB_mhd1024_hd/      # held-out
  JHTDB_channel_hd/      # held-out
  uniform/               # held-out
```

Each domain contains paired `*_img1.tif`, `*_img2.tif` and `*_flow.flo` files. The fixed split is `splits/class1_primary_split_v2.json`; do not regenerate it for a comparison.

Place the two main-instance checkpoints at the paths expected by the runner:

```text
piv_runs/residual_ddpm_v6_x0_seed3407/diffusion_backbone_best.pth
piv_runs/physics_unrolled_v3_warmup_scratch_20ep_seed3417/physics_agent_best.pth
```

## 3. Reproduce the per-sample records

```bash
python pipeline/run_piv_uncertainty_study.py \
  --data-path <PATH_TO_SAMPLE_DATA> \
  --split test \
  --arms full \
  --output-dir piv_runs/uncertainty_study

python pipeline/run_piv_uncertainty_study.py \
  --data-path <PATH_TO_SAMPLE_DATA> \
  --split purged \
  --arms calibration \
  --output-dir piv_runs/uncertainty_study
```

The runner writes `study_<split>.npz` and `study_<split>_meta.json`. The metadata records the backbone and agent checkpoint paths, the split manifest hash, the arm configuration and the seed; those are the values quoted in the paper.

Held-out records were generated in the same way after pointing `--data-path` at the held-out domain and `--split-manifest` at the corresponding manifest in `external/holdout/`; the exact configuration is preserved in each `study_test_meta.json`.

## 4. Three introduction routes

```bash
python pipeline/run_three_routes.py
```

This trains the baseline, three loss-penalty strengths and two structural-constraint seeds and writes `paper_figures/uncertainty_study/tables/three_routes.npz`. The analysis (`scripts/analyse_three_routes.py`) runs on the cached file and does not retrain.

## 5. Second implementation (FunDiff)

The cross-implementation check uses the public FunDiff repository. Clone it into `external/fundiff-main`:

```bash
git clone <FUNDIFF_REPOSITORY_URL> external/fundiff-main
python pipeline/export_fields_for_fundiff.py
python pipeline/run_fundiff_formal.py
```

`run_fundiff_formal.py` fits the FunDiff encoder/decoder/DiT on the exported PIV fields and writes `paper_figures/uncertainty_study/tables/fundiff_check_formal.json`.

## 6. Training from scratch

The backbone and agent training commands are the same as in the original project:

```bash
python pipeline/train_holdout_backbone.py \
  --data-path <PATH_TO_SAMPLE_DATA> \
  --output-dir piv_runs \
  --split-manifest splits/class1_primary_split_v2.json \
  --external-pretraining-declaration "Frozen torchvision RAFT weights only; no PIV supervision." \
  --epochs 50 --batch-size 4 --eval-batch-size 4 \
  --run-name ddpm_class1_seed3407

python main.py --mode train_agent \
  --data-path <PATH_TO_SAMPLE_DATA> \
  --output-dir piv_runs \
  --checkpoint piv_runs/ddpm_class1_seed3407/diffusion_backbone_best.pth \
  --split-protocol manifest \
  --split-manifest splits/class1_primary_split_v2.json \
  --dataset-kind simulation --seed 3407 \
  --batch-size 8 --eval-batch-size 8 \
  --agent-epochs 8 --agent-lr 1e-5 --agent-max-delta 6 \
  --agent-refine-steps 2 --agent-matching-radius 1 \
  --run-name upr_ddpm_seed3407_train
```

The released main-instance checkpoints are the ones used for the paper; retraining is only needed to test a new configuration.

## 7. Expected outputs

| Command | Output |
| --- | --- |
| `pipeline/run_piv_uncertainty_study.py` | `piv_runs/uncertainty_study/study_*.npz`, `study_*_meta.json` |
| `scripts/analyze_piv_uncertainty_study_en.py` | `paper_figures/uncertainty_study/{analysis.json,summary.md,tables/*,figures/*}` |
| `scripts/recompute_all.py` | `tables/coverage_wilson.json`, `tables/second_2x2_unified.json`, `tables/permutation_1000.json` (the last two use the precomputed cache if `permutation_fields.npz` is absent) |
| `scripts/holdout_all.py` | `tables/holdout_all.json` |
| `scripts/analyse_three_routes.py` | `tables/three_routes.json` |
| `scripts/plot_main_figures.py` | `paper_figures/uncertainty_study/main_figures/*.png` |

## 8. Cost

On the publication hardware, the five-domain test pass with 10 candidates and three intervention strengths takes a few GPU-hours; the CPU-only analysis takes minutes. The 1000-permutation check is the most expensive CPU step and is why the field cache is archived separately.
