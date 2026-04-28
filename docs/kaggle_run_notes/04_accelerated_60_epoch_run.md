# Accelerated 60-Epoch Run

Archive: `results (5).zip`

## Purpose

This is the main completed artifact from the final accelerated training sequence.

## Keepable Facts

- Runtime: `17.47` minutes
- Epochs: `60`
- Best logged validation objective: `-0.5500311085156032`
- Dataset: `2,607` curves and `124,970` total points
- Training split: `1,824` train curves and `783` validation curves
- Synthetic calibration samples: `500`
- Synthetic exploration samples: `2,000`
- Time points per trajectory: `48`
- Solver: `rk4`
- Adjoint: `False`
- Batch size: `128`

## Why It Matters

- It completed cleanly under tight remaining compute limits.
- It retained the full curve count while reducing per-curve temporal resolution.
- It is the most usable finished run from the repaired fast-training path.

## Caution

This is still a reduced-budget configuration, so it should be described as an accelerated final run rather than a full unrestricted retraining.
