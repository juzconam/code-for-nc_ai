# External code

This directory is for third-party code used by the optional cross-implementation check. It is not redistributed here.

## FunDiff

The second implementation uses the public FunDiff repository (encoder, decoder, DiT and rectified-flow objective). Clone it into `external/fundiff-main`:

```bash
git clone <FUNDIFF_REPOSITORY_URL> external/fundiff-main
```

Then export the PIV fields and run the formal check:

```bash
python pipeline/export_fields_for_fundiff.py
python pipeline/run_fundiff_formal.py
```

The exported fields are written to `external/fundiff_data/`; the result is `paper_figures/uncertainty_study/tables/fundiff_check_formal.json`.

If you do not need the cross-implementation check, this directory can remain empty. The CPU-only analysis does not import FunDiff.
