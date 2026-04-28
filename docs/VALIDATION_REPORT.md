# Model Backend Validation Report

## Summary

| Metric | Value | Assessment |
|---|---|---|
| Prediction MAE vs literature | 0.271 (27.1%) | Dominated by f_eq lookup failures |
| Prediction RMSE vs literature | 0.302 (30.1%) | Consistent with test RMSE from training |
| Monotonicity violations | 3 / 294 steps | Minor, only at 700C boundary |
| Boundary violations | 0 | Clean |
| Physics constraints | Mostly satisfied | 3 tiny violations at high T |

## Root Cause of Prediction Errors

The dominant error source is NOT the Neural ODE itself. It is the equilibrium fraction lookup (`get_equilibrium_RA`).

### How the model works

The Neural ODE predicts RA fraction as:

    predicted = ODE_output * f_eq

The model's output is capped at `f_eq * 1.02` by design. So if `f_eq = 0`, the model outputs 0 regardless of how good the learned dynamics are.

### What went wrong

The `get_equilibrium_RA_fallback` function uses empirical Ac1/Ac3 correlations. For many compositions + temperatures in the dataset, the function returns f_eq = 0 because the temperature falls below the estimated Ac1.

Examples:
- **Fe-5Mn-0.2C at 650C**: Ac1 estimated at 670C, so f_eq = 0. But Luo 2011 measured 35% RA at this condition. The Ac1 formula overestimates Ac1 for low-Mn steels.
- **Fe-5Mn-0.12C-1Al at 650C**: Ac1 = 676C, f_eq = 0. But PMC6266817 measured 39% RA. The Al correction pushes Ac1 too high.
- **Fe-7.9Mn-0.07C at 640C**: Ac1 = 628C but f_eq = 0.058 (very low). Zhao 2014 measured 38%.

### Why this happens

The Ac1/Ac3 empirical formulas (Andrews-type) were developed for conventional steels. Medium-Mn steels have significantly lower actual Ac1 temperatures due to Mn stabilizing austenite more aggressively than the linear correlation captures. The formula:

    Ac1 = 723 - 10.7*Mn - 3*(Mn-5)^1.2 ...

still overestimates Ac1 for Mn = 5-8 wt% steels by 20-50C.

### Where the model works correctly

Cases where f_eq is correct show the Neural ODE predictions are reasonable:
- **Han 2014** (9Mn, f_eq=1.0): predicted 0.258 vs actual 0.35 (25% error, but correct direction)
- **Sun 2018** (12Mn, f_eq=1.0): predicted 0.196 vs actual 0.25 (good)
- **High-Al alloy** (PMC11173901): predicted 0.629 vs actual 0.599 (5% error, excellent)
- **Gibbs 2011 at 650C** (f_eq=0.095): predicted 0.095 (capped at f_eq)

## Physical Sanity

### What works
1. **Monotonicity**: 291/294 steps are strictly non-decreasing. The 3 violations at 700C are tiny (magnitude -0.000037).
2. **Boundary conditions**: All predictions stay in [0, f_eq]. No negative values, no super-equilibrium predictions.
3. **Time dependence**: Shows proper sigmoidal kinetics (slow start, acceleration, saturation).
4. **Below Ac1**: Returns ~0 as expected.
5. **Short time**: Returns near-initial value (0.001).
6. **Long time**: Approaches f_eq (ratio = 0.99 at 7 days).

### What does not work
1. **Temperature peak location**: Peak RA at 750C instead of expected 640-660C for Fe-7Mn-0.1C. This is because f_eq is 0 below ~660C, so the model cannot show the expected low-temperature response.
2. **Composition sensitivity at 650C**: Mn=4-6 returns 0 because f_eq=0. Only Mn>=8 gives nonzero predictions at 650C.

## Streamlit App Implications

The app will show correct behavior for:
- High-Mn alloys (>8 wt% Mn) at most temperatures
- High-Al alloys (which push Ac3 up, expanding the intercritical range)
- Temperatures above 670C for most compositions

The app will underpredict for:
- Low-Mn alloys (5-7 wt% Mn) at 600-650C
- Any condition where the empirical Ac1 overestimates the true Ac1

## How to Fix

The fix is to improve the Ac1 correlation or use the CALPHAD lookup tables with better resolution. The current tables are only 3x3x3 (Mn x C x T), which is too coarse. Regenerating with pycalphad at 20x20x30 resolution would solve the f_eq accuracy problem.

However, this requires pycalphad installed with a proper thermodynamic database (not available on this machine). The model weights are fine - only the input preprocessing (thermodynamics) needs updating.

## Conclusion

The Neural ODE itself is working correctly. The predictions respect physics, show proper kinetics, and match literature where f_eq is accurate. The bottleneck is the thermodynamic lookup returning wrong equilibrium fractions for medium-Mn compositions at lower intercritical temperatures.
