# Model Card: v4_stage2_fixed_best.pt

## Model Details

- **Name:** PhysicsNODE v4 Stage 2 Fixed
- **Architecture:** Physics-Constrained Latent Neural ODE
- **Parameters:** 78,474 (all trainable)
- **Solver:** Dormand-Prince 4/5 (dopri5), rtol=1e-5, atol=1e-7
- **Framework:** PyTorch + torchdiffeq
- **Training hardware:** Kaggle T4 GPU (16GB), 31 min total
- **File size:** 981 KB

## Training Data

- **Source:** 125 experimental observations from 25 peer-reviewed studies (2010-2024)
- **Composition range:** Fe-(3.9-12.0)Mn-(0.0-0.40)C-(0.0-4.3)Al wt%
- **Temperature range:** 25-1000 °C
- **Time range:** 0-604,800 s (0-7 days)
- **Measurement methods:** XRD (88 pts), EBSD (5), neutron diffraction (5)
- **Pre-training:** Stage 1 on ~1055 synthetic JMAK curves (57,227 points), 120 epochs
- **Fine-tuning:** Stage 2 on real data only, batch_size=1, 16 epochs, early-stopped

## Performance

| Metric | Value |
|--------|-------|
| Test R² (held-out studies) | **+0.378** |
| Test RMSE | **0.135** |
| Overall RMSE (all 124 pts) | 0.135 |
| Overall MAE | 0.104 |
| Per-study median R² | +0.205 |
| Studies with R² > 0 | 12/21 |
| Monotonicity violations | 0% |
| Boundary violations | 0% |

### Best-predicted studies
- Hu & Luo 2017: R² = 0.63
- Yan 2022: R² = 0.68
- Lee & De Cooman 2014: R² = 0.60
- PMC11053108: R² = 0.56

### Worst-predicted studies
- PMC6266817 (Fe-5Mn-0.12C-1Al, cold-rolled): R² = -11.7 (Al causes Ac1 overestimation)
- Hausman 2017: R² = -3.1 (short-time overshoot)
- Nakada 2014: R² = -2.0 (boundary of training distribution)

## Intended Use

- **Primary:** Screening tool for narrowing composition-temperature combinations before experiments
- **Secondary:** Teaching tool for austenite reversion kinetics
- **NOT for:** Production alloy design without experimental verification

## Limitations

1. **Small dataset:** 125 points from heterogeneous literature sources
2. **Al-containing steels:** Ac1 overestimation for Al > 1 wt% degrades predictions
3. **Measurement bias:** XRD vs EBSD gives 4-17% different RA on same sample — the model cannot distinguish
4. **Missing covariates:** Grain size, prior Mn distribution, and processing history are not inputs
5. **Validation set:** Only 2 points — checkpoint selection is effectively random

## Physics Constraints

| Constraint | Implementation | Compliance |
|------------|---------------|------------|
| Monotonicity (df/dt ≥ 0) | Physics gate: rate × saturation × nucleation | 100% |
| Boundary (f ≤ f_eq) | Output clamping to f_eq × 1.02 | 100% |
| Initial condition (f(0) ≈ 0) | y0 initialized to 0.001 | 100% |
| Thermodynamic consistency | ΔG gate in ODE right-hand side | 100% |

## Ethical Considerations

- All training data is from published, peer-reviewed literature with full citation
- No proprietary or restricted data used
- Model predictions should never replace metallurgical judgment
- Uncertainty estimates (MC dropout, 95% CI) are provided to quantify confidence
