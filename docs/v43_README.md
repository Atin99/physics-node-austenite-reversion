# Project 4.3

`project_4.3` is the merged master workspace for this retained-austenite project.

It combines:

- the original latent Neural ODE project from `project_4`
- the data-salvage, real-data tiering, and baseline/fusion pipeline from `project_4.2`

## Working layout

- `src/latent_node`
  Original latent NODE code imported from `project_4`
- `src/fusion_baselines`
  Real-data consolidation, synthetic-surrogate generation, simple ML, basic NN, and fusion baselines
- `scripts/run_project43_pipeline.py`
  Main merged pipeline entrypoint
- `data/raw`
  Source-traced CSV, PDF, TXT, and extracted table assets
- `data/processed`
  Cleaned train-ready real-data splits
- `data/synthetic`
  Explicitly labeled synthetic-surrogate rows
- `docs/prompts`
  The prompt set and project brief
- `docs/reports`
  Status, inventory, and pipeline reports
- `kaggle/notebook_cells`
  One-cell Kaggle runner for Commit/Run plus split cells for manual debugging
- `kaggle/upload/project_4_3_kaggle_input.zip`
  Upload-ready Kaggle input bundle
- `references/`
  Preserved copies of `project_4` and `project_4.2`

## Local run

```powershell
cd "C:\project 4\project_4.3"
$env:PYTHONPATH="C:\project 4\project_4.3\references\project_4_2_original\_pydeps"
python scripts\run_project43_pipeline.py --n-synth 5000
```

## Kaggle bundle

```powershell
python scripts\build_kaggle_bundle.py
```

The bundle builder writes a forward-slash-safe zip:

- `kaggle\input_build\project_4_3_kaggle_input.zip`
- `kaggle\upload\project_4_3_kaggle_input.zip`

For the current Kaggle job, paste `kaggle\notebook_cells\00_RUN_ALL_SINGLE_CELL.py` into one GPU notebook cell. The older `notebooks\train_colab.py` file is retained only as original-project reference material.
