# archive - old versions and original bundles

this folder keeps earlier code versions and original zip bundles for reference.
nothing here is needed for the active project. its just an audit trail.

## whats in here

### code_v1_earliest/
the original codebase before any kaggle fixes. smaller model, simpler loss function, basic trainer.
- model.py (9KB) - no variable-length padding support
- losses.py (5KB) - fewer loss terms
- trainer.py (15KB) - basic training loop without the stage2 fixes
- visualizations.py
- pipeline.log - from a local test run

### code_v2_intermediate/
this is what actually ran stage 1 on kaggle and worked. bigger model, more loss terms, better readme.
still missing the padding/masking stuff that stage 2 needed though.
- model.py (11KB)
- losses.py (5KB)
- trainer.py (17KB)
- publication_pipeline.py
- README.md (8.5KB)
- pipeline.log

### with_kaggle_session/
files from the with_kaggle working directory that had kaggle-specific outputs and analysis.
- complete_project.py - analysis/completion script
- FINAL_RESULTS.md - the honest results writeup
- pipeline.log
- README.md

### old_zips/
original zip bundles from various kaggle uploads and recovery attempts.
contents are already extracted into the active folders, these are kept for provenance.
- kaggle_full_recovery_bundle_v1.zip
- recovered_stage1_bundle_v1.zip
- medium_mn_neural_ode_kaggle_input.zip
- with_kaggle_upload.zip
- stage2_fixed_artifacts.zip
- improved_notebook_cells.zip
- failed_notebook_r5.ipynb and .log - the stage 1 notebook that also tried stage 2 and crashed
