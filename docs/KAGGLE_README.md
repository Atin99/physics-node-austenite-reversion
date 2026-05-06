# Kaggle Run Guide

This workspace ships with a one-cell Kaggle workflow for Commit/Run plus the older split-cell workflow for manual debugging.

## Upload target

Upload:

- `kaggle/upload/project_4_3_kaggle_input.zip`

## Recommended: one cell, Commit/Run

Create a Kaggle notebook with GPU enabled, attach the zip above as an input dataset, paste the full contents of:

- `kaggle/notebook_cells/00_RUN_ALL_SINGLE_CELL.py`

Then use Kaggle Commit/Run. The single cell runs:

1. real-data formatting + simple ML + basic NN + fusion baselines
2. latent PhysicsNODE Stage 1 full-data training (`120` epochs)
3. latent PhysicsNODE Stage 2 real-only training (`150` epochs)
4. evaluation/export pass

It writes stage markers, logs, mirrored checkpoints, and snapshots under:

- `/kaggle/working/project_4_3_final_artifacts`
- `/kaggle/working/project_4_3_final_artifacts_snapshot.zip`

If a stage fails, completed earlier stages remain in the artifact folder. The runner continues to later independent stages by default and records failures in `_state/*.failed.json`.

## Manual fallback: split cells

Use these only when debugging manually:

1. `01_setup_and_unzip.py`
2. `02_run_real_pipeline.py`
3. `03_run_baselines_with_synth.py`
4. `04_prepare_latent_node_workspace.py`
5. `05_stage1_pretrain.py`
6. `06_stage2_retrain.py`
7. `07_evaluate_and_export.py`

## Pause points

- After Cell 2: cleaned data and splits are ready
- After Cell 3: baseline/fusion results are ready
- After Cell 5: Stage 1 checkpoint should exist
- After Cell 6: Stage 2 checkpoint should exist
- After Cell 7: export results and final artifacts

## Path note

The bundle zip is created with normalized archive paths using `/`, so unzip behavior is stable on Kaggle Linux.

## Legacy notebook files

`notebooks/KAGGLE_FINAL_RUN.md` and `notebooks/train_colab.py` are preserved from the original `project_4` history as references. Do not use `train_colab.py` for the current Kaggle run; the current handoff is the one-cell runner above.
