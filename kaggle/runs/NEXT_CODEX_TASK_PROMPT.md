# Prompt for Next Codex Session

You are working in `C:\project 4`.

The user has three Kaggle Project 4.3 runs archived locally. First, preserve the archive as evidence. Do not delete or overwrite the original downloaded logs, notebooks, or result zips.

Archived files are under:

`C:\project 4\project_4.3\run_archives\2026-05-06_kaggle_runs`

Run folders:

1. `run1_trial2`
   - `p4-trial2.ipynb`
   - `p4-trial2.log`
   - `download (9).txt`
   - `results (12).zip`

2. `run2_final_trial`
   - `p4-final-trial.ipynb`
   - `p4-final-trial.log`
   - `download (10).txt`
   - `results (13).zip`

3. `run3_hifi_trial1`
   - `p4-hifi-trial-1.ipynb`
   - `p4-hifi-trial-1.log`
   - `download (11).txt`
   - `results (14).zip`
   - `project_4_3_hifi_eval_repair_input.zip`

The HiFi run is the important one. It completed training but had post-training legacy evaluation filename errors.

HiFi facts from `p4-hifi-trial-1.log`:

- Wrapper: `PROJECT 4.3 HIFI20 WRAPPER`
- Base run profile: `balanced`
- Profile params: `baseline_n_synth=3500`, `stage1_epochs=90`, `stage2_epochs=180`
- Model params: `batch=32`, `amp=1`, `adjoint=0`, `sn=0`, `rtol=2e-4`, `atol=1e-6`, `max_steps=3000`, `time=log10`
- HiFi Stage 1 trained successfully: `Best val_real_rmse: 0.163158`
- HiFi Stage 2 real-only trained successfully: `Best val_real_rmse: 0.207348`
- Final summary says `PROJECT 4.3 SINGLE-CELL RUN FINISHED`, `elapsed_hours=4.5967`, `failed_stages=[]`
- Only failed pieces were `evaluate_comprehensive.py` and `ablation_study.py`, because they expected:
  - `/kaggle/working/project43_latent_stage2_real_only/models/stage2_fixed_best.pt`
  - `/kaggle/working/project43_latent_stage2_real_only/models/stage2_extended_best.pt`
- Actual checkpoint exists in `results (14).zip`:
  - `project43_latent_stage2_real_only/src/models/checkpoints/physics_node_best.pt`

Already-created HiFi repair artifacts:

- Kaggle input zip:
  - `C:\project 4\project_4_3_hifi_eval_repair_input.zip`
  - `C:\project 4\project_4.3\kaggle_hifi_eval_repair\project_4_3_hifi_eval_repair_input.zip`
  - `C:\project 4\project_4.3\run_archives\2026-05-06_kaggle_runs\run3_hifi_trial1\project_4_3_hifi_eval_repair_input.zip`
- Paste-ready Kaggle eval-only cell:
  - `C:\project 4\project_4.3\kaggle_hifi_eval_repair\RUN_HIFI_EVAL_ONLY.py`

Repair zip contents have been verified:

- `RUN_HIFI_EVAL_ONLY.py`
- `project43_latent_stage2_real_only/src/models/checkpoints/physics_node_best.pt`
- `project43_latent_stage2_real_only/models/stage2_fixed_best.pt`
- `project43_latent_stage2_real_only/models/stage2_extended_best.pt`
- `hifi_eval_repair_manifest.json`

Next task:

1. Help the user upload `project_4_3_hifi_eval_repair_input.zip` to Kaggle as a dataset.
2. Tell the user to paste and run `RUN_HIFI_EVAL_ONLY.py` in a fresh Kaggle notebook cell.
3. The repair cell should not train. It only restores checkpoint aliases and reruns evaluation/export.
4. After the repair run, collect `project_4_3_hifi_eval_repair_outputs.zip` from Kaggle.
5. If the user provides that output zip, inspect it and update the final project report to use the HiFi Stage 2 checkpoint as the main model.
6. Be honest in the final narrative: the original HiFi training was successful; only legacy post-training eval needed checkpoint alias repair.
