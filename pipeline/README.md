# Optional pipeline entry points

The scripts here reproduce the network outputs from which the per-sample records were built. They require the public PIV data and the checkpoints in `docs/CHECKPOINTS.md`; the CPU-only analysis in `scripts/` does not import them.

| Script | Purpose |
| --- | --- |
| `run_piv_uncertainty_study.py` | Draws the 10 DDIM candidates per field, applies the intervention arms and writes `study_<split>.npz` |
| `run_three_routes.py` | Trains the six configurations of the three-route comparison and writes `three_routes.npz` |
| `export_fields_for_fundiff.py` | Exports the PIV fields for the FunDiff cross-implementation check |
| `run_fundiff_formal.py` | Trains the FunDiff encoder/decoder/DiT and writes `fundiff_check_formal.json` |
| `train_backbone.py`, `train_holdout_backbone.py` | Backbone training (all-domain or leave-one-domain-out) |
| `plot_qualitative_plate.py` | Regenerates the qualitative velocity/divergence plate for Fig. 1b-d |

Commands and expected outputs are in `docs/FULL_PIPELINE.md`.

Note: `main.py`, `models.py`, `physics_utils.py` and `src/` live at the repository root so that both `pipeline/` and `scripts/` can import them with a single `sys.path` entry.
