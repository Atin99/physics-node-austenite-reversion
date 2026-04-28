# kaggle run archive

all gpu runs are here, including the failed ones. see docs/RUN_REGISTRY.md for the full timeline.

## runs

| folder | date | what happened | result |
|---|---|---|---|
| run_01_apr25_stage1_early/ | apr 25 | first stage 1 attempt, short run | results_2.zip + physics_node_last.pt |
| run_02_apr25_stage1_extended/ | apr 25 | longer stage 1, multiple attempts | results_3/4/5.zip |
| run_03_apr27_stage1_full/ | apr 27-28 | full 120 epoch stage 1, stage 2 crashed | results_6.zip |
| run_04_apr28_recovery/ | apr 28 | re-evaluation of stage1 checkpoint only | results_7.zip |
| run_05_apr28_stage2_unfixed/ | apr 28 | stage 2 without time sanitization, early stopped, worse than stage 1 | no result zip |
| run_06_apr28_stage2_fixed/ | apr 28 | stage 2 with time sanitization + batch_size=1, best result | results_8.zip |
| run_07_apr28_stage2_latest/ | apr 28 | latest run using the new kaggle_stage2_cell.py | results_9.zip, needs review |

## best model

run 06 gave the best model. its saved as models/stage2_fixed_best.pt
- val_real_rmse = 0.21244
- test_real_rmse = 0.31214
