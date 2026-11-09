# Code availability

The code and derived records supporting this study are available at:

- GitHub: `https://github.com/<ORG>/<REPO>` (release tag `v1.0.0`)
- Archived archive: `https://doi.org/<ZENODO_DOI>`

The release contains:

- `scripts/`: analysis code that reproduces the main-text numbers, Fig. 2, Fig. 3, Table 1 and the Supplementary Tables from the per-sample records;
- `piv_runs/`: per-sample derived quantities for the main instance, the three held-out domains and the second gate checkpoint (no raw images);
- `paper_figures/uncertainty_study/tables/`: derived tables, including the precomputed 1000-permutation null and the second-pipeline arm results;
- `pipeline/` and `src/`: training/inference entry points for the main instance, included for provenance; running them requires the public PIV data and the checkpoints listed in `docs/CHECKPOINTS.md`.

A CPU-only environment reproduces the analysis; no GPU or raw data are required for the main-text claims. The 42 MB `permutation_fields.npz` field cache is not included in the minimal archive; the precomputed null distribution is included as `permutation_1000.json`, and the cache is available from the archive DOI for exact recomputation.

Suggested statement for the manuscript:

> Code is available at `https://github.com/<ORG>/<REPO>` (tag `v1.0.0`) and archived at `https://doi.org/<ZENODO_DOI>`. The archive contains the analysis code, the per-sample derived records needed to reproduce all main-text numbers, and the training/inference entry points for the main instance. Raw PIV images are public benchmark data and are not redistributed.
