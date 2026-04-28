# HANDOFF PROMPT — paste this into your next chat

---

I have a project at `c:\project 4\project_5\`. It was just reorganized from a messy multi-folder setup into one clean folder. I need you to continue working on it. READ EVERYTHING BELOW BEFORE DOING ANYTHING.

## WHAT THIS PROJECT IS

Physics-Constrained Latent Neural ODE for austenite reversion kinetics in medium-Mn steels. Its a metallurgy ML project. The model predicts transformation kinetics (austenite fraction vs time) using a Neural ODE with physics constraints (monotonicity, thermodynamic bounds).

The project was trained on Kaggle GPUs across multiple runs. All training is done. The code, models, figures, logs, and run outputs are all in this one folder.

## FOLDER STRUCTURE — READ THIS CAREFULLY

```
c:\project 4\project_5\
  README.md              - project overview
  requirements.txt       - python deps

  src/                   - THE PRODUCTION SOURCE CODE (v3, correct version)
    config.py            - hyperparameters, paths
    model.py             - Neural ODE architecture (has variable-length padding support)
    trainer.py           - training loop with stage1 + stage2 support (27KB, the big one)
    losses.py            - physics-informed loss function
    data_generator.py    - synthetic + real data pipeline
    real_data.py         - literature data extraction (66KB, huge file with all papers)
    features.py          - feature engineering
    explainability.py    - SHAP-based model interpretation
    main.py              - entry point
    publication_pipeline.py - figure generation pipeline
    visualizations.py    - plotting functions
    complete_project.py  - analysis/completion script
    (plus: optimizer_annealing, streamlit_app, symbolic_regression, thermodynamics)

  data/                  - input data
    calphad_tables/      - thermodynamic lookup tables (npy files)
    literature_validation/ - literature_validation.csv (the real experimental data)
    synthetic/           - empty (generated at runtime)
    user_experimental/   - empty (placeholder)

  models/                - ALL model checkpoints
    stage1_best_ep109.pt - best stage 1 model (120 epochs, picked at epoch 109)
    stage1_last.pt       - last stage 1 model
    stage2_fixed_best.pt - BEST OVERALL MODEL (from run 06, stage 2 with time-sanitization)
    stage2_fixed_last.pt
    stage2_run7_best.pt  - from run 07 (60 full epochs, better val but worse test)
    physics_node_*.pt    - old smoke test models (keep but not important)
    *.csv, *.json        - training histories and run summaries

  figures/               - publication figures (22 main + 2 from run7)
    fig0 through fig9    - png and pdf versions
    fig7_shap.*          - SHAP explainability plot
    fig2_parity_run7_latest.png - parity plot from the full 60-epoch run
    fig8_training_run7_latest.png - training curve from the full 60-epoch run

  kaggle/
    cells/               - the python scripts pasted into kaggle notebooks
      kaggle_cell_v1.py          - original combined stage1+2 cell
      kaggle_stage2_cell.py      - latest stage 2 only cell (the one that produced run 07)
      kaggle_recover_from_stage1.py - recovery cell
      stage2_only_retry_fixed.py - another retry variant
    upload_bundles/      - zip files uploaded as kaggle datasets
    runs/                - ALL 7 kaggle GPU runs, organized
      run_01 through run_07 with logs, notebooks, result zips
      README_RUNS.md     - index of all runs

  docs/                  - all documentation
    FINAL_RESULTS.md     - honest results interpretation (important for viva)
    PROJECT_DEFENSE_QBANK.md - viva defense question bank
    RUN_REGISTRY.md      - timeline of all kaggle runs with metrics
    newplan_improvement.txt - project reframing notes (read this, its the real direction)
    data_sources.md      - literature sources
    kaggle_run_notes/    - per-run analysis notes
    report_notes/        - report drafts and notes

  tests/                 - unit tests
  archive/               - old code versions v1 and v2, original zip bundles
```

## WHAT THE MODELS ACTUALLY ACHIEVED

Three things to know:

1. Best model is stage2_fixed_best.pt (from run 06):
   - val_real_rmse = 0.21244 (16 curves, 32 points)
   - test_real_rmse = 0.31214 (16 curves, 35 points)

2. Run 07 did a full 60-epoch stage 2 (no early stopping):
   - val_real_rmse = 0.21145 (better val)
   - test_real_rmse = 0.32797 (worse test — overfit on tiny val set)

3. The val-test gap (0.21 vs 0.31) is THE finding. Cross-study generalization is hard with heterogeneous literature data. This is not a failure, this IS the result.

## SOURCE CODE VERSIONS — DO NOT MIX THESE UP

There were 3 versions of the code during development:

- v3 (in src/) — CORRECT production version. Has point_mask/obs_mask for variable-length padding, 3-tuple _validate(), val_real_rmse checkpoint selection. USE THIS ONE.
- v2 (in archive/code_v2_intermediate/) — what ran stage 1 on kaggle. Missing the padding fixes.
- v1 (in archive/code_v1_earliest/) — original. Smaller model.

The src/ folder IS v3. Do not modify it to look like v2 or v1. If you see code that looks different from whats in src/, the src/ version is correct.

## WHAT NEEDS TO BE DONE NEXT

The training is finished. The models exist. What remains:

1. CHECK THE LATEST RUN (run_07): Look at kaggle/runs/run_07_apr28_stage2_latest/artifacts/ — it has the full 60-epoch stage2 output including parity_real.png and stage2_history.png. Compare these with run_06 results. Run 06 has better test metrics but run 07 ran longer.

2. UPDATE FIGURES IF NEEDED: The figures/ folder has publication figures. Some were generated from an older model state. If you can run the publication_pipeline.py locally or regenerate on kaggle using the best checkpoint (stage2_fixed_best.pt), do it. If not, the existing figures are usable.

3. FINALIZE DOCS: The docs/FINAL_RESULTS.md and docs/PROJECT_DEFENSE_QBANK.md are good. Read docs/newplan_improvement.txt — it has the correct framing for the project (not "I made a great model" but "I diagnosed what limits predictive quality when literature data is messy").

4. IF ANOTHER KAGGLE RUN IS NEEDED: Use kaggle/cells/kaggle_stage2_cell.py. Upload kaggle/upload_bundles/kaggle_stage2_upload.zip as a kaggle dataset. The cell is self-contained — just paste it as one cell in a GPU notebook.

5. COMPLETE THE PROJECT: This means having a coherent set of: source code, trained models, publication figures, honest results writeup, and defense prep. Most of this exists already. Just needs final polish.

## IMPORTANT RULES

- Do NOT create new folders outside project_5. Everything stays in c:\project 4\project_5\
- Do NOT duplicate files. If something exists in one place, dont copy it to another.
- The user has ~500MB RAM free. Dont do heavy operations.
- Keep all markdown casual and human-like. No emojis, no AI-style showoff language.
- Keep typos in comments. Dont over-polish.
- The project framing is in docs/newplan_improvement.txt. Read it. The project is about diagnosing bottlenecks in literature-derived metallurgy data, not about showing off an architecture.

---
