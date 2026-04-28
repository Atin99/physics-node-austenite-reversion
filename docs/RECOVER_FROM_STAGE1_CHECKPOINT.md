# Recover From Saved Stage-1 Checkpoint

Use this when a long Kaggle run finished stage 1 but crashed later.

## What survived in the failed run

From the downloaded `results (6).zip`, the following stage-1 outputs survived:

- `final_project_4_artifacts/stage1_best.pt`
- `final_project_4_artifacts/stage1_last.pt`
- `final_project_4_artifacts/stage1_history.csv`
- `final_project_4_artifacts/stage1_history.png`

The log also shows:

- stage 1 completed all `120` epochs
- best `val_real_rmse` reached about `0.212782`
- crash happened in stage 2 during backprop, not during stage 1 saving

## Best immediate recovery strategy

Do **not** rerun the full notebook.

Instead:

1. Use the saved `stage1_best.pt` as the main recovered model
2. Skip stage 1 completely
3. Skip broken stage 2 for now
4. Run only:
   - dataset build
   - checkpoint load
   - validation / test evaluation
   - artifact packaging

This turns a 9+ hour rerun into a short recovery run.

## Why this is the safest move

- stage 1 already contains the expensive learning
- stage 2 is currently buggy
- rerunning stage 1 unchanged is unnecessary and risky
- packaging + evaluation from a saved checkpoint is cheap

## What to use as the recovered model

Prefer:

- `stage1_best.pt`

Keep also:

- `stage1_last.pt`

## Stage-2 crash summary

The failed run died with:

`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

This means the stage-2 loss being backpropagated was not connected to gradients.

Treat stage 2 as broken until explicitly patched and smoke-tested.

## Practical recommendation

For the next Kaggle attempt:

- make a short recovery notebook
- mount the recovered `stage1_best.pt` as an input artifact
- rebuild dataset
- load checkpoint
- evaluate and export outputs
- do not touch stage 2 until fixed locally first

