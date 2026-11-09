# Figure and table map

This file maps each display item of the manuscript to the primary data and script that generate it. The final Nature-styled composites were assembled from these panel-level outputs; the composite images are in `docs/figures/` for reference.

## Main figures

| Item | Content | Primary data | Script / command |
| --- | --- | --- | --- |
| Fig. 1a | Four stages of physics-guided generative inference | schematic | `python scripts/plot_concept_figure.py` |
| Fig. 1b-d | DNS, backstep and cylinder velocity and divergence | `paper_figures/uncertainty_study/tables/qualitative_plate.npz` | `pipeline/plot_qualitative_plate.py` (raw data + checkpoints required) |
| Fig. 2a | Four illustrative constructions | `analysis.json` -> construction block | `python scripts/analyze_piv_uncertainty_study_en.py` |
| Fig. 2b | Width ratio vs scale-to-error ratio, 48 units | `tables/diagnostic_points.csv` | `python scripts/analyze_piv_uncertainty_study_en.py` |
| Fig. 2c | Measured minus nominal coverage vs nominal level | `analysis.json` -> `alpha_sweep` | `python scripts/analyze_piv_uncertainty_study_en.py` |
| Fig. 2d | Interval width on derivative-type quantities | `analysis.json` -> `coverage`, `tension_summary` | `python scripts/analyze_piv_uncertainty_study_en.py` |
| Fig. 2e | Normalized spread-scaled interval width | `analysis.json` -> `tension_summary` | `python scripts/analyze_piv_uncertainty_study_en.py` |
| Fig. 2f | Gate lowers R of constrained quantities | `analysis.json` -> `coverage`, `tension_summary` | `python scripts/analyze_piv_uncertainty_study_en.py` |
| Fig. 2g | Coverage by flow domain and quantity | `analysis.json` -> `per_domain` | `python scripts/coverage_by_domain_quantity.py` |
| Fig. 2h | Divergence distributions of truth and means | `piv_runs/aggregation_mechanism/mechanism.npz` | `python scripts/plot_main_figures.py` |
| Fig. 2i | Divergence power spectra of candidate and truth | `piv_runs/divergence_spectrum/spectrum.npz` | `python scripts/plot_main_figures.py` |
| Fig. 2j | Spectral truncation does not restore truth | `piv_runs/band_divergence/band_divergence.npz` | `python scripts/plot_main_figures.py` |
| Fig. 2k | Five constraint operators | `tables/operator_family.csv` | `python scripts/analyze_piv_uncertainty_study_en.py` |
| Fig. 3 | Held-out MHD and channel domains | `tables/holdout_examples.npz`, `tables/holdout_all.json` | `python scripts/holdout_all.py`; `python scripts/plot_holdout_figure.py` |
| Table 1, upper block | Three introduction routes | `tables/three_routes.npz` | `python scripts/analyse_three_routes.py` |
| Table 1, lower block | Five intervention arms on the second implementation | `tables/second_pipeline_allarms_v2.json`, `tables/second_pipeline_2x2.npz` | `python scripts/centre_scale_ablation.py` |

## Supplementary figures

| Item | Content | Primary data |
| --- | --- | --- |
| Fig. S1 | Cross-implementation and cross-system check | `tables/diagnostic_points.csv` |
| Fig. S2 | Candidate number and conditional coverage | `analysis.json` -> `k_sweep`, `per_domain` |
| Fig. S3 | Three physical-introduction routes | `tables/three_routes.npz` |
| Fig. S4 | Generator families and in-loop guidance | `tables/operator_family.csv`, `tables/second_pipeline_allarms_v2.json` |
| Fig. S5 | Diagnostics and calibration methods | `tables/diagnostic_points.csv`, `tables/conditional_coverage.csv` |
| Fig. S6 | Spread dynamic range on experimental data | `tables/real_piv_domains.json` |
| Fig. S7 | Dose-response and nominal level | `analysis.json` -> `dose_response`, `alpha_sweep` |
| Fig. S8 | Reliability and mechanism | `analysis.json` -> `reliability_summary`, `tables/reliability_bins.csv` |
| Fig. S9 | Spectra and the nonlinear ensemble | `piv_runs/divergence_spectrum/spectrum.npz`, `piv_runs/band_divergence/band_divergence.npz` |
| Fig. S10 | Measurement-quality scan on the second instance | `tables/second_pipeline_2x2.npz`, `tables/fundiff_check_formal.json` |

## Supplementary tables

| Item | Primary data / script |
| --- | --- |
| Tables S1-S2 | `python scripts/analyze_piv_uncertainty_study_en.py`; `python scripts/coverage_by_domain_quantity.py`; `python scripts/run_interval_score_comparison.py` |
| Table S3 | `tables/second_pipeline_allarms_v2.json` |
| Table S4 | `tables/second_pipeline_allarms_v2.json`; `tables/three_routes.npz` |
| Table S5 | `tables/real_piv_domains.json` |
| Table S6 | `piv_runs/replicate_agent_v2/study_test.npz`; `python scripts/replication_paired.py` |
| Table S7 | `tables/holdout_all.json`, `tables/holdout_mhd.json` |
| Table S8 | `analysis.json` -> `coverage`; `python scripts/run_interval_score_comparison.py` |

## Reproducibility notes

- `alpha = 0.10` throughout; the split-conformal quantile is the `ceil((n + 1)(1 - alpha))`-th order statistic.
- `R` is the ratio of the constant-interval width to the spread-scaled interval width at matched marginal coverage; `R > 1` means the spread scale is useful.
- All paired comparisons reuse the same candidate set; gated and ungated arms differ only by the gate.
- The `uid` field is the sample identifier; `domain` and `index` preserve the split ordering.

If a number in the manuscript does not match a table, the table is the source of truth and the manuscript should be updated to it.
