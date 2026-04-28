# Project Conclusion

**Project:** Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics in Medium-Mn Steels
**Author:** Atin, Dept. of Metallurgy and Materials Engineering, Jadavpur University
**Date:** April 2026

---

## What was built

A physics-constrained Neural ODE that predicts retained austenite fraction in medium-Mn steels (3-12 wt% Mn) as a function of composition, annealing temperature, and time. The model was trained on 125 experimental data points from 25 published studies spanning 2010-2024.

The training uses a two-stage protocol:
- Stage 1: pre-training on ~1055 synthetic kinetic curves to teach basic sigmoidal behavior
- Stage 2: fine-tuning on real experimental data only

Physics constraints (monotonicity, boundary conditions, thermodynamic driving force) are built into the loss function. The equilibrium austenite fraction and transformation temperatures come from empirical correlations recalibrated against the 25-study dataset.

A Streamlit web app allows interactive predictions.

## Final metrics

### Kaggle training metrics (retrained with corrected thermodynamics)

| Metric | Old model | New model | Change |
|---|---|---|---|
| val_real_rmse | 0.212 | 0.161 | -24% |
| test_real_rmse | 0.312 | 0.131 | -58% |

### Backend validation against literature (10 independent test cases)

| Metric | Old model | New model |
|---|---|---|
| MAE vs literature | 0.271 (27.1%) | 0.138 (13.8%) |
| RMSE vs literature | 0.302 (30.1%) | 0.173 (17.3%) |
| Max single-point error | — | 0.363 |
| Monotonicity violations | 3 / 294 | 3 / 294 |
| Boundary violations | 0 | 0 |

### What changed

The original model had broken thermodynamic inputs:
- The Ac1 formula (Andrews-type) overestimated Ac1 by 50-100C for medium-Mn steels, cutting off predictions below the true intercritical range
- The equilibrium RA fraction formula returned 0 or 1.0 instead of realistic values (0.30-0.65)

Both were recalibrated against published phase diagram data from the 25-study dataset. After retraining with corrected inputs, the val-test gap collapsed from 0.10 to ~0.03, confirming the original gap was thermodynamic input error, not data heterogeneity alone.

## Honest assessment

### What works well
- Cross-composition prediction with a single model (18 alloy compositions, one set of weights)
- Physics constraints hold: no negative RA, no super-equilibrium values, monotonic kinetic curves
- Temperature dependence is physically reasonable (peak at intermediate T, low at extremes)
- Time dependence shows proper sigmoidal shape
- Below Ac1 returns near-zero, long holds approach equilibrium

### What does not work well
- 4 out of 10 literature test cases still fail the 15% error threshold
- The PMC6266817 case (Fe-5Mn-0.12C-1Al, cold-rolled) is badly underpredicted (0.027 vs 0.390 actual). The Al correction in the Ac1 formula pushes the estimated Ac1 too high, collapsing f_eq.
- Short-time predictions for some compositions overshoot (Luo 2011 at 1h: 0.368 predicted vs 0.100 actual)
- 125 data points from 25 studies is still a small dataset. Some composition-temperature regions are poorly covered.

### The remaining error sources
1. **Measurement method bias** — XRD and EBSD give 4-17% different RA fractions on the same sample. The model has no way to learn this.
2. **Missing covariates** — grain size, Mn microsegregation, prior processing history are not inputs to the model but affect kinetics.
3. **Empirical Ac1/Ac3** — the recalibrated correlations are better but still approximate. CALPHAD with a proper thermodynamic database would improve this further.

## What was learned

1. The bottleneck was not model architecture or training procedure. It was the thermodynamic preprocessing. Fixing Ac1 and f_eq cut the test RMSE by more than half.

2. The original val-test gap (0.212 vs 0.312) was partly real data heterogeneity and partly a thermodynamic input bug. After the fix, the gap is much smaller, suggesting the data heterogeneity story was overstated in the earlier analysis.

3. For literature-derived datasets, the quality of physics-based feature engineering matters more than training hyperparameters. Getting the equilibrium fraction right was worth more than 100 extra training epochs.

4. 125 data points from 25 studies is enough to train a useful prototype but not enough for reliable out-of-distribution prediction. The model works best within the composition ranges well-represented in the training data (Fe-5-9Mn, 0.05-0.20C).

## What would improve things further

Ranked by expected impact:

1. **Better thermodynamics** — CALPHAD lookup with pycalphad and a proper database (TCFE or similar) instead of empirical Ac1 correlations. This would fix the remaining f_eq errors for Al-containing and low-Mn steels.

2. **More real data** — especially for compositions currently underrepresented (high-Al, low-Mn, high-C). Even 30-50 additional well-characterized points would help.

3. **Measurement method as input** — adding XRD/EBSD/neutron as a categorical feature so the model can learn the systematic bias between techniques.

4. **Ensemble prediction** — averaging the top-k checkpoints for more stable predictions and uncertainty estimates.

5. **Standardized reporting** — the community adopting consistent protocols for RA measurement and reporting would make all future data-driven work more reliable.

## Publication route

### Target venues (in order of preference)

1. **SSRN preprint** — free, immediate, gets a DOI, can be cited
2. **arXiv** (cond-mat.mtrl-sci cross-listed with cs.LG) — free, high visibility in both communities
3. **Computational Materials Science** (Elsevier) — peer-reviewed, accepts computational + ML work on steels
4. **Materials Today Communications** — shorter format, open access option available
5. **Zenodo** — DOI for the dataset and code separately

### Practical notes

- Paper draft is at `docs/PAPER_DRAFT.md`
- No LaTeX setup available — manuscript is in markdown, can be converted via pandoc
- Solo author (undergraduate project)
- All data has DOIs and source references, no proprietary data
- Code and trained model are open source on GitHub

### What the paper should emphasize

- The dataset curation effort (125 points, 25 studies, full provenance)
- The thermodynamic recalibration story (this is a practical finding others can use)
- Cross-composition generalization with a single model vs JMAK per-alloy fitting
- The honest data quality assessment

---

## Repository state

- **Checkpoint:** `models/stage2_fixed_best.pt` (retrained with corrected thermodynamics)
- **App:** `src/streamlit_app.py` (run via `launch.bat` or `streamlit run src/streamlit_app.py`)
- **Validation:** `python validate_model.py` (runs all backend tests)
- **Paper draft:** `docs/PAPER_DRAFT.md`
- **Full technical report:** `docs/TECHNICAL_REPORT.md`
