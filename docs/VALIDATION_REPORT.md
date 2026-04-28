# Model Validation Report

Validated with `validate_model.py` using checkpoint `stage2_fixed_best.pt` (retrained with corrected thermodynamics).

## Summary

| Metric | Value | Notes |
|---|---|---|
| Prediction MAE vs literature | 0.138 (13.8%) | 10 independent test cases |
| Prediction RMSE vs literature | 0.173 (17.3%) | Improved from 0.302 after thermo fix |
| Max single-point error | 0.363 | PMC6266817 cold-rolled case |
| Monotonicity violations | 3 / 294 | Negligible magnitude (~1e-6) |
| Boundary violations | 0 | All predictions in [0, f_eq] |

## Predictions vs published data

| Study | Actual | Predicted | Error | f_eq | Pass? |
|---|---|---|---|---|---|
| Gibbs 2011 650C 1wk | 0.435 | 0.425 | -0.010 | 0.425 | yes |
| Gibbs 2011 625C 1wk | 0.428 | 0.442 | +0.014 | 0.442 | yes |
| Gibbs 2011 600C 1wk | 0.343 | 0.454 | +0.111 | 0.454 | yes |
| Luo 2011 650C 24h | 0.350 | 0.469 | +0.119 | 0.469 | yes |
| Luo 2011 650C 1h | 0.100 | 0.368 | +0.268 | 0.469 | no |
| PMC6266817 CR peak | 0.390 | 0.027 | -0.363 | 0.170 | no |
| Han 2014 peak | 0.350 | 0.190 | -0.160 | 0.379 | no |
| Zhao 2014 peak | 0.380 | 0.218 | -0.162 | 0.419 | no |
| Suh 2017 peak | 0.450 | 0.412 | -0.038 | 0.483 | yes |
| Sun 2018 peak | 0.250 | 0.120 | -0.130 | 0.392 | yes |

Pass threshold: absolute error < 0.15.

6 out of 10 cases pass. The 4 failures have identifiable causes.

## Error analysis

### Cases that work well

The model handles Gibbs 2011 and Suh 2017 data accurately. These are compositions where the recalibrated Ac1/f_eq correlations return correct thermodynamic inputs. When f_eq is right, the Neural ODE produces good predictions.

### Cases that fail

**PMC6266817 (Fe-5Mn-0.12C-1Al):** The Al correction in the Ac1 formula pushes estimated Ac1 too high, which forces f_eq down to 0.170 when the actual equilibrium fraction is much higher. The model cannot predict more than f_eq by design, so the output collapses to 0.027.

**Luo 2011 at 1h:** The 24h prediction is reasonable (0.469 vs 0.350), but the 1h prediction overshoots badly (0.368 vs 0.100). The model predicts fast early kinetics for this composition. This suggests the synthetic pre-training biases the model toward faster transformation than Luo 2011 observed at short times.

**Han 2014 and Zhao 2014:** Both underpredicted by ~0.16. The f_eq values are reasonable (0.38, 0.42), but the predicted RA at 1h does not reach the published peak values. These compositions (9Mn and 7.9Mn) may have faster reversion kinetics than the model learned from the training distribution.

### Remaining thermodynamic issues

The recalibrated Ac1 formula works well for Fe-Mn-C ternaries but still overestimates Ac1 for Al-containing steels. A CALPHAD lookup with pycalphad would fix this, but requires a proper thermodynamic database (TCFE or similar).

## Physical sanity

### Monotonicity

291 out of 294 integration steps are strictly non-decreasing. The 3 violations occur at extreme temperatures (575C, 675C, 700C) with magnitudes on the order of 1e-6. These are numerical artifacts from the ODE solver, not physical violations.

### Boundary conditions

All predictions stay in [0, f_eq]. No negative values, no super-equilibrium fractions.

### Temperature dependence

Peak RA at 630C for Fe-7Mn-0.1C (1h hold). Physically reasonable — low enough for Mn partitioning to be effective, high enough for significant driving force. RA drops off at both extremes as expected.

### Composition sensitivity

Mn = 4: near-zero RA (below Ac1 at 650C).
Mn = 6-8: highest RA (optimal intercritical range).
Mn = 10-12: decreasing RA (kinetics slow as equilibrium fraction saturates).

This matches the well-known Mn dependence in medium-Mn steels.

### Time dependence

Fe-7Mn-0.1C at 650C:

| Time | Predicted RA |
|---|---|
| 1 min | 0.014 |
| 5 min | 0.138 |
| 30 min | 0.211 |
| 1 h | 0.274 |
| 4 h | 0.417 |
| 1 d | 0.425 |
| 7 d | 0.425 |

Proper sigmoidal shape. Saturates near f_eq for long holds.

### Edge cases

- Below Ac1: RA = 0.000 at T = Ac1 - 50C. Correct.
- Very short time (1s): RA = 0.001. Correct.
- Very long time (7d): RA/f_eq = 1.00. Approaches equilibrium. Correct.
- High-Al alloy (Fe-9.4Mn-0.2C-4.3Al at 800C): predicted 0.460 vs published 0.599. Underpredicted due to Al-induced Ac1 error, but order of magnitude is right.

## Comparison with previous model

| Metric | Old model | New model |
|---|---|---|
| MAE vs literature | 0.271 | 0.138 |
| RMSE vs literature | 0.302 | 0.173 |
| Pass rate (< 0.15) | 3/10 | 6/10 |
| Boundary violations | 0 | 0 |

The improvement comes entirely from fixing the thermodynamic input functions (Ac1 and f_eq), not from changes to the model architecture or training procedure. The model weights are the only thing that changed, and they changed because they received correct inputs during retraining.
