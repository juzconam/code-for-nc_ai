# Data availability

Raw PIV images, flow fields and checkpoints are **not** redistributed in this archive. The archive contains the derived per-sample records needed to reproduce the paper's numbers (`piv_runs/`) and the derived tables (`paper_figures/uncertainty_study/tables/`).

## Public benchmark data

| Domain | Role in the paper | Source |
| --- | --- | --- |
| DNS turbulence | training / in-domain test | Public PIV benchmark; see manuscript refs. 18-20 and `<DATASET_URL_1>` |
| JHTDB channel | training / in-domain test | Johns Hopkins Turbulence Database (JHTDB), channel-flow cutouts; `<JHTDB_URL>` |
| SQG | training / in-domain test | Public PIV benchmark; `<DATASET_URL_2>` |
| Backstep | training / in-domain test | Public PIV benchmark; `<DATASET_URL_3>` |
| Cylinder | training / in-domain test | Public PIV benchmark; `<DATASET_URL_4>` |
| MHD turbulence (`JHTDB_mhd1024_hd`) | fully held out | JHTDB, MHD1024; `<JHTDB_URL>` |
| High-resolution channel (`JHTDB_channel_hd`) | fully held out | JHTDB, channel; `<JHTDB_URL>` |
| Uniform flow | fully held out (weaker-effect control) | Public benchmark; `<DATASET_URL_5>` |
| In vitro cylinder (Re = 413) | experimental check | `<EXPERIMENTAL_URL_1>` |
| In vivo aorta / left ventricle | experimental check | Public in vivo PIV dataset, ref. 46; `<EXPERIMENTAL_URL_2>` |

The exact dataset versions, licences and download dates should be copied from the manuscript's Methods/References and `DATA.md` before submission. The filenames and split identifiers used here are preserved in `splits/class1_primary_split_v2.json` and in the `uid` fields of the `.npz` records.

## Included derived records

| Path | Contents |
| --- | --- |
| `piv_runs/uncertainty_study/study_test.npz` | 1,065 test fields; six quantities for 10 candidates at each intervention arm; point estimate, spread and EPE |
| `piv_runs/uncertainty_study/study_validation.npz` | 958 validation fields (calibration) |
| `piv_runs/uncertainty_study/study_purged.npz` | 250 purified fields |
| `piv_runs/holdout_mhd/`, `holdout_channel/`, `holdout_uniform/` | held-out-domain records |
| `piv_runs/replicate_agent_v2/` | second independently trained gate checkpoint |
| `piv_runs/aggregation_mechanism/`, `divergence_spectrum/`, `band_divergence/` | aggregation identity, divergence spectra and band-resolved divergence |
| `paper_figures/uncertainty_study/tables/` | all derived tables used by the figures |

## Checkpoints

The two checkpoints used for the main instance are listed in `docs/CHECKPOINTS.md` with sizes and SHA-256 values. They are archived at `<ZENODO_DOI>` and are not required for the CPU-only analysis.

## Suggested statement for the manuscript

> The raw PIV images and flow fields are public benchmark data (sources listed in Methods and `DATA.md`) and are not redistributed. The derived per-sample records, analysis code and derived tables needed to reproduce all main-text numbers are available at `https://github.com/<ORG>/<REPO>` (tag `v1.0.0`) and archived at `https://doi.org/<ZENODO_DOI>`. The trained checkpoints are archived at the same DOI.
