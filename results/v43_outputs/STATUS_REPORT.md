# Project 4.2 Status Report (2026-05-03)

## 1) Real Data Status

- Total standardized rows (`master_rows_all.csv`): **409**
- Core exact ML-ready (`experimental_core_exact.csv`): **88**
- Digitized ML-ready (`experimental_digitized_with_uncertainty.csv`): **96**
- Transfer weak-label ML-ready (`transfer_real_weak_labels.csv`): **189**
- Candidate/not-ready (`experimental_candidates_needs_review.csv`): **29**
- Legacy unverified (`legacy_unverified_rows.csv`): **7**

Primary ML pools:
- `ml_ready_real_pool.csv`: **184 rows**, **33 studies**
- `ml_ready_with_transfer.csv`: **373 rows**, **47 studies**

## 2) Synthetic Status

- Generated labeled synthetic surrogate rows: **5000**
- File: `data/synthetic/synthetic_calphad_surrogate.csv`
- Provenance is explicit (`synthetic_calphad_surrogate`), not mixed as real.

## 3) Baseline Model Status (LOSO)

Scenario: `real_only` (184 rows)
- OLS RMSE 0.1526, R2 -0.3018
- Ridge RMSE 0.1517, R2 -0.2871
- kNN RMSE 0.1298, R2 0.0574
- basic_nn RMSE 0.1672, R2 -0.5629
- Fusion RMSE 0.1353, R2 -0.0239

Scenario: `real_plus_transfer` (373 rows)
- OLS RMSE 0.1136, R2 0.3451
- Ridge RMSE 0.1135, R2 0.3473
- kNN RMSE 0.1122, R2 0.3621
- basic_nn RMSE 0.1383, R2 0.0302
- Fusion RMSE 0.1092, R2 0.3957

## 4) Files Added for Continuation

- `src42/build_real_dataset.py`
- `src42/generate_physics_surrogate.py`
- `src42/train_baselines.py`
- `run_project42_pipeline.py`
- `README_4_2_PIPELINE.md`
- `NEXT_TODO_PROMPT.md`

## 5) Honesty Notes

- No row here is labeled as experimental unless sourced from local source-traced files.
- Legacy questionable rows are isolated in `legacy_unverified_rows.csv`.
- Candidate/approx rows are isolated in `experimental_candidates_needs_review.csv`.

