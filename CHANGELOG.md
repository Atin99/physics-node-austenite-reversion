# Changelog

## v4.4 — Final Consolidation (May 2026)

### Content merge
- Merged ALL code, data, models, docs, scripts from v4, v4.2, v4.3
- Deep scan of every folder in workspace — nothing left behind
- Extracted checkpoints from results(12).zip and results(13).zip (trial2 + final runs)
- Archived all 4 Kaggle run folders with notebooks, logs, and result zips
- Copied scripts/, .streamlit/, HANDOFF_PROMPT.md, kaggle input zips
- Root zip files (hifi_eval_repair, kaggle_input) preserved in kaggle/

### Source code
- Deduplicated: 15 identical files kept once, 3 upgraded to v4.3 versions
- Fixed eval scripts: checkpoint path discovery, missing dir creation, sys.path
- config.py: dual-layout resolver (works both locally and on Kaggle)
- Streamlit app: complete rewrite with 7-pipeline selector

### Models (9 checkpoints)
- v4_stage2_fixed_best.pt — BEST (test R²=+0.378, 16ep)
- v4_stage2_extended_best.pt — 200ep extended
- v4_stage2_run7_best.pt — early Stage 2
- v4_stage1_best_ep109.pt — pre-training only
- v43_hifi_stage1_best.pt, v43_hifi_stage2_best.pt — HiFi config
- v43_trial2_stage2_best.pt — run1_trial2 (fast trained)
- v43_final_stage2_fixed_best.pt, v43_final_stage2_extended_best.pt — run2_final_trial

### Documentation
- Paper draft: fixed abstract (R²=+0.378), added cross-version comparison
- MODEL_CARD.md: standard ML model card for best checkpoint
- HONEST_RESULTS.md: full cross-version metric comparison
- 35+ docs from all versions preserved

### Data
- 375 data files across processed, raw, synthetic, calphad, literature
- 125-pt literature validation CSV verified (126 lines incl header)
- 373-pt ML-ready dataset with transfer labels
- 5000-row CALPHAD synthetic surrogate

## v4.3 — Pipeline + HiFi Training (May 2026)
- Built multi-stage Kaggle pipeline runner
- Profile system (fast/balanced/kaggle)
- log10 time transform, env variable overrides
- 4 Kaggle runs: trial2, final_trial, hifi_trial1, hifi_fix
- HiFi run: val_rmse=0.207, but test R²=-3.563 (overfit)
- Eval repair: fixed checkpoint aliasing

## v4.2 — Data Expansion (May 2026)
- Expanded: 184 ML-ready real points, 373 with transfer
- CALPHAD surrogate: 5000 synthetic rows
- LOSO baselines (Ridge, kNN, Fusion, OLS, basic_nn)
- Best baseline: Fusion R²=0.396 on 373 points

## v4 — Original (April 2026)
- Physics-Constrained Latent Neural ODE (78K params)
- 125-point literature dataset from 25 studies
- Two-stage pre-training protocol
- Key breakthrough: thermodynamic recalibration (Ac1 + f_eq)
- Best: Stage2 Fixed test R²=+0.378, 12/21 positive R²
- Physics: 0 monotonicity violations, boundary conditions met
- Streamlit app, SHAP explainability, publication figures
