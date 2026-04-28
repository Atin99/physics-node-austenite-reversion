# Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics

Medium-Mn steel transformation kinetics modeling using a physics-informed Neural ODE trained on literature-derived data.

## what this project actually is

The goal was to build a kinetics predictor for austenite reversion in medium-Mn steels. There are plenty of papers on this topic, but the data across them is fragmented, inconsistent, and not ML-ready. Different studies use different alloys, temps, measurement methods, and reporting styles.

So this project ended up being less about "look at my model" and more about testing what happens when you try to learn kinetics from messy heterogeneous literature data. The answer: the model works, but the bottleneck is data quality, not architecture.

## the dataset

125 experimental measurements from 25 peer-reviewed studies (2010-2024). Every data point has a DOI, source reference, measurement method, and quality flag. See `data/DATASET_CARD.md` for full documentation.

Composition coverage: Mn 3.93-12%, C 0-0.40%, Al 0-4.3%, T 25-1000C, time 0-604800s.

## results

| metric | stage 1 (ep 109) | stage 2 best (run 06) | stage 2 full (run 07) |
|---|---|---|---|
| val_real_rmse | 0.21278 | 0.21244 | 0.21145 |
| test_real_rmse | 0.31368 | 0.31214 | 0.32797 |
| test_real_mae | 0.27103 | 0.26699 | 0.27695 |

run 06 is the best model on test. run 07 ran more epochs but overfit on the tiny val set.

the val-test gap (0.21 vs 0.31) is the main finding. cross-study generalization is hard when the data comes from different labs with different protocols. thats not a bug, thats the result.

## folder structure

```
project_5/
  src/                 - production source code (v3, with padding/masking fixes)
  data/                - calphad tables, literature validation csv, dataset card
  models/              - all checkpoints (stage1, stage2, smoke tests)
  figures/             - publication figures (22 files including shap)
  analysis.py          - post-hoc analysis script (cpu only, no gpu needed)
  analysis_results/    - output from the analysis script
  notebooks/           - colab training script, kaggle run instructions
  kaggle/
    cells/             - all kaggle execution cells
    upload_bundles/    - zip files for kaggle dataset upload
    runs/              - all 7 kaggle runs with logs notebooks and result zips
  docs/                - paper draft, defense qbank, run registry, notes
  tests/               - unit tests
  archive/             - old code versions and original zip bundles
```

## source code versions

there are 3 versions of the code. only v3 is active.

- v3 (src/) - production. has point_mask/obs_mask, proper val_real_rmse checkpoint selection.
- v2 (archive/code_v2_intermediate/) - ran stage 1 on kaggle. missing the stage 2 fixes.
- v1 (archive/code_v1_earliest/) - original version.

## the hard part

the hard part was not finding papers. it was turning heterogeneous, partially incompatible literature evidence into something a kinetics model could learn from without giving fake confidence.

## running the analysis

```
python analysis.py
```

this runs cpu-only and generates per-study stats, training dynamics, JMAK baseline comparisons.

## requirements

```
pip install -r requirements.txt
```
