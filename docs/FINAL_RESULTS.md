# Final Results

## The numbers

### Best model: stage2_fixed_best.pt (retrained with corrected thermodynamics)

**Kaggle training metrics:**
```
val_real_rmse  = 0.161   (16 curves, 32 points)
test_real_rmse = 0.131   (16 curves, 35 points)
```

**Backend validation vs published literature (10 independent cases):**
```
MAE  = 0.138 (13.8%)
RMSE = 0.173 (17.3%)
Max single-point error = 0.363
Pass rate (< 0.15 error) = 6 / 10
```

### Previous model (broken thermodynamics) for comparison

```
val_real_rmse  = 0.212
test_real_rmse = 0.312
MAE vs literature = 0.271
RMSE vs literature = 0.302
```

### What changed

The improvement comes from fixing two thermodynamic input functions:

1. **Ac1 formula** — the Andrews-type correlation overestimated Ac1 by 50-100C for medium-Mn steels. Recalibrated with a nonlinear Mn correction.
2. **Equilibrium RA fraction** — the empirical formula returned 0 or 1.0 instead of realistic 0.30-0.65 for most compositions. Replaced with a calibrated sigmoid model.

The model architecture, training procedure, and loss function were unchanged. Only the thermodynamic inputs changed, and the model was retrained with the corrected inputs.

---

## What these numbers mean

- On **validation data**, the average prediction error is about 16% of the target scale. Reasonable for a prototype.
- On **test data**, the error is about 13%. The test set happens to contain compositions where the thermodynamic corrections are most effective.
- Against **independent literature cases**, 6 out of 10 predictions are within 15% of published values. The 4 failures have identifiable thermodynamic causes (Al-containing steels, short-time overshooting).

### Is this good enough for production?

No. An MAE of 0.138 means individual predictions can be off by up to 0.363. For alloy design, you would still need experimental verification.

### Is this good for a research prototype?

Yes. The model demonstrates cross-composition generalization, respects physics constraints, and quantifies where the remaining errors come from. That is the contribution.

---

## Training history

### Stage 1 (120 epochs, synthetic + real data)

Most learning happened in the first 30 epochs (RMSE dropped from 0.357 to 0.215). The remaining 90 epochs gave less than 0.002 improvement. The model was data-limited, not training-limited.

### Stage 2 retrain (with corrected thermodynamics)

Restarted from the stage 1 checkpoint with fixed Ac1/f_eq inputs. Converged within ~16 epochs to the new best performance.

---

## What would help more

Ranked by expected impact:

1. Better thermodynamics (CALPHAD with pycalphad, not empirical correlations)
2. More real data, especially for underrepresented compositions
3. Measurement method as input feature (XRD vs EBSD correction)
4. Checkpoint ensemble for uncertainty estimates

## What would not help

- More synthetic data (already 95% synthetic, diminishing returns)
- Bigger model (architecture is not the bottleneck)
- Fancier optimizer (already using cosine annealing + gradient clipping)
- More training epochs (converged well within the given budget)
