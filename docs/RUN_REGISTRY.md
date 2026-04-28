# Kaggle Run Registry — Project 4 Neural ODE

All GPU runs are documented here. Every run is preserved, including failures.

## Run Timeline

### April 25, 2026 — Early Development Runs

| ID | Time | Result File | Log | Status | Notes |
|---|---|---|---|---|---|
| R1 | ~14:29 | `results (2).zip` (13.1 MB) | `project-4-neural-ode.log` (28 KB) | ❓ Unknown | Short early run, also produced `physics_node_last.pt` |
| R2 | ~16:03 | `results (3).zip` (14.4 MB) | `project-4-neural-ode (1).log` (190 KB) | ❓ Unknown | Longer run |
| R3 | ~20:57 | `results (4).zip` (14.2 MB) | same log session | ❓ Unknown | Another attempt |
| R4 | ~21:06 | `results (5).zip` (26.1 MB) | same log session | ❓ Unknown | Largest Apr 25 run |

**Archive location:** `all_runs_archive/run_apr25_early/` and `all_runs_archive/run_apr25_long/`
**Also stored in:** `FINAL_PROJECT_4/with_kaggle/kaggle_output_zips/` (results 1-5)

---

### April 27-28, 2026 — Main Training Run

| ID | Time | Result File | Notebook | Log | Status | Notes |
|---|---|---|---|---|---|---|
| R5 | Apr 27 09:54 – Apr 28 | `results (6).zip` (8.4 MB) | `notebook3a269bab41.ipynb` | `notebook3a269bab41.log` (292 KB) | ⚠️ Stage 1 ✅, Stage 2 ❌ | Full 120-epoch stage 1 completed. Stage 2 crashed. |

**Key metrics (stage 1 best @ epoch 109):**
- val_real_rmse = 0.21278
- test_real_rmse = 0.31368

**Archive location:** `kaggle_safe_archive/` (already archived earlier)

---

### April 28, 2026 — Recovery Run

| ID | Time | Result File | Status | Notes |
|---|---|---|---|---|
| R6 | 02:33-02:34 UTC | `results (7).zip` (10.5 MB) | ✅ Success | Loaded stage1_best.pt, evaluated on all splits. No new training. |

**Key metrics (re-evaluated stage 1 checkpoint):**
- val_real_rmse = 0.21278
- test_real_rmse = 0.31368

**Archive location:** `recovered_results7/` and `kaggle_safe_archive/results_7_recovery.zip`

---

### April 28, 2026 — Stage 2 Retries (same notebook session)

Both ran in ONE Kaggle notebook: `notebooke4e9673ee1`

| ID | Time | Log | Status | Epochs | val_real_rmse | test_real_rmse | Notes |
|---|---|---|---|---|---|---|---|
| R7 | 02:49 UTC | `notebooke4e9673ee1 (1).log` | ⚠️ Early stopped | 19/60 | 0.21266 | 0.32150 | Unfixed. Many skipped batches ("t must be strictly increasing"). Test WORSE than stage 1. No checkpoint saved (overwritten by R8). |
| R8 | 02:56 UTC | `notebooke4e9673ee1.log` | ✅ Best result | 16/60 | **0.21244** | **0.31214** | Fixed: time-sanitized, batch_size=1. Zero skipped batches. Both val AND test improved vs stage 1. Early stopped due to patience. |

**R8 is the final best model.**

**Result zip:** `results (8).zip` → contains `final_project_4_stage2_fixed_artifacts/final_best_real.pt`

**Archive location:** `all_runs_archive/run_apr28_stage2_fixed/` and `all_runs_archive/run_apr28_stage2_unfixed/`

---

## Final Best Metrics

| Metric | Stage 1 (ep 109) | Stage 2 Fixed (ep 6/16) | Delta |
|---|---|---|---|
| val_real_rmse | 0.21278 | **0.21244** | −0.00035 |
| test_real_rmse | 0.31368 | **0.31214** | −0.00154 |
| val_real_mae | 0.18623 | 0.18679 | +0.00056 |
| test_real_mae | 0.27103 | **0.26699** | −0.00404 |
| test_real_endpoint_mae | 0.29241 | **0.28673** | −0.00568 |

## GPU Time Spent

- April 25 runs: ~3-5 hours (multiple attempts)
- April 27-28 main run: ~22 hours (full stage 1 + failed stage 2)
- April 28 recovery + retries: ~0.5 hours
- **Total estimated: ~25-28 hours of 30-hour weekly quota**

## Early Stopping Issue

Both stage 2 retries were cut short (16 and 19 epochs out of 60) because:
- Default patience was ~10 epochs
- The real validation set has only 16 curves / 32 points
- Small random fluctuations trigger the patience counter
- The model was already near a local minimum from stage 1

**If doing another run: set patience ≥ 50 or disable it entirely.**
