# Fast Smoke 30-Epoch Run

Archive: `results (4).zip`

## Purpose

This run was the minimal smoke test used to confirm that the repaired fast training path completed end to end.

## Keepable Facts

- Runtime: `0.32` minutes
- Epochs: `30`
- Best logged validation objective: `-0.03577897697687149`
- Dataset: `267` curves and `3,182` total points
- Training split: `186` train curves and `81` validation curves
- Solver: `rk4`
- Adjoint: `False`
- Batch size: `128`

## How To Use It

- Use it only as proof of pipeline execution after the fast-path repair.
- Do not present it as a serious final training result.
