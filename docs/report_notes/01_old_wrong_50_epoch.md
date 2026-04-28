# Part 1: Old Wrong 50-Epoch Run

Status: nonessential, removable, keep only for comparison.

## Purpose

This file preserves the earlier 50-epoch quick Kaggle training run. It should not be treated as the final reported model. It is retained only for comparative analysis, limitations discussion, and "studying the wrong" if you want to document how the training process evolved.

## Sources

- Notebook quick-test cell: `medium_mn_neural_ode_final/notebooks/train_colab.py`
- Quick-test section label in notebook: `CELL 4: Quick Test Run (50 epochs)`
- Artifact inspected: `C:\Users\dmana\Downloads\results (2).zip`
- Checkpoint inside artifact: `medium_mn_neural_ode/models/checkpoints/physics_node_50ep_best.pt`

## Keepable Facts

- The run was explicitly configured as a quick 50-epoch diagnostic pass.
- Recorded epochs in checkpoint history: `50`
- Saved checkpoint: `physics_node_50ep_best.pt`
- Best total validation objective recorded in checkpoint: `-1.380907416343689`
- Train total objective moved from `-0.014647839956783823` to `-1.4821996944291251`
- Validation total objective moved from `-0.03309726055998068` to `-1.380907416343689`
- Train data loss moved from `0.0015042750902856433` to `0.00035549526884486634`
- Validation data loss recorded in checkpoint history: `0.00036046633389420237`
- Validation physics loss recorded in checkpoint history: `8.304349711816774e-07`
- Physics violations were `0.0` at the start and end of the stored history

## Caution

The reported total loss is the weighted homoscedastic training objective, not a plain prediction-error metric. This makes the run useful as a preliminary or "wrong/intermediate" training record, but not as a clean final-performance claim.

## Suggested Report Use

Possible wording:

> A preliminary 50-epoch diagnostic run was retained for comparative analysis only and was not used as the final reported model.

## Why Keep It At All

- It shows that the training pipeline executed end to end.
- It gives a concrete earlier checkpoint for comparison against later runs.
- It can support a short methodological note about why preliminary optimization logs were not treated as final evidence of model quality.
