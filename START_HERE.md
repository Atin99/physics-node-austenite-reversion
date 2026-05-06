# Start Here

Use `C:\project 4\project_4.3` as the master folder for this project.

## What this folder is

- merged `project_4` + `project_4.2`
- cleaned structure for data, code, docs, outputs, and Kaggle handoff
- preserves the old folders under `references/` so nothing is lost

## The most important files

- `README.md`
  Main overview and local pipeline command
- `docs/reports/PROJECT_4_3_STATUS.md`
  Current verified status and metrics
- `scripts/run_project43_pipeline.py`
  Main merged pipeline runner
- `scripts/build_kaggle_bundle.py`
  Rebuilds the Kaggle upload zip
- `kaggle/upload/project_4_3_kaggle_input.zip`
  Upload-ready Kaggle dataset zip
- `kaggle/notebook_cells/`
  Kaggle one-cell runner plus split debug cells

## Fast local commands

```powershell
cd "C:\project 4\project_4.3"
$env:PYTHONPATH="C:\project 4\project_4.3\references\project_4_2_original\_pydeps"
python scripts\run_project43_pipeline.py --n-synth 5000
python scripts\build_kaggle_bundle.py
```

## Data honesty

- `data/processed`
  real/source-traced and transfer-tagged rows
- `data/synthetic`
  synthetic surrogate only

They are kept separate on purpose.

## Kaggle order

Recommended for submit/commit:

1. Upload `kaggle/upload/project_4_3_kaggle_input.zip`
2. Paste `kaggle/notebook_cells/00_RUN_ALL_SINGLE_CELL.py` into one Kaggle notebook cell
3. Commit/Run with GPU enabled

Manual debug fallback:

1. `01_setup_and_unzip.py`
2. `02_run_real_pipeline.py`
3. `03_run_baselines_with_synth.py`
4. `04_prepare_latent_node_workspace.py`
5. `05_stage1_pretrain.py`
6. `06_stage2_retrain.py`
7. `07_evaluate_and_export.py`
