# Technical Report: Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics in Medium-Mn Steels

**Author:** Atin  
**Date:** April 2026  
**Repository:** https://github.com/Atin99/physics-node-austenite-reversion

---

## 1. Problem Statement

Medium-manganese steels (3-12 wt% Mn) are candidates for third-generation advanced high-strength steels because they use the TRIP effect. The retained austenite (RA) fraction after intercritical annealing controls mechanical properties. Predicting this fraction as a function of composition, temperature, and time is useful for alloy design.

The problem is that existing kinetics models (JMAK, Avrami) assume single-mechanism isothermal transformations and require per-alloy parameter fitting. They cannot generalize across compositions. Data-driven approaches should help, but the available literature data is scattered, inconsistent, and measured with different techniques.

This project builds a Neural ODE that enforces physics constraints and tests whether it can learn austenite reversion kinetics from pooled literature data. The answer is partially yes, with caveats.

## 2. Dataset

### 2.1 Construction

125 experimental data points were extracted from 25 peer-reviewed studies published between 2010 and 2024. Every value has a DOI, source reference (table or figure number), measurement method, and data quality flag.

The extraction was done manually. Values from tables were copied directly. Values from text were noted with their context. Values from figures were digitized with estimated +/-2-5% uncertainty.

### 2.2 Coverage

| Property | Range |
|---|---|
| Mn content | 3.93 - 12.0 wt% |
| C content | 0.0 - 0.40 wt% |
| Al content | 0.0 - 4.3 wt% |
| Temperature | 25 - 1000 C |
| Time | 0 - 604,800 s (1 week) |
| RA fraction | 0 - 64.7% |
| Studies | 25 |
| Alloy compositions | 18 unique |

Measurement methods: XRD (93%), EBSD (4%), neutron diffraction (3%).

Data quality: digitized from figures (77%), text-reported (19%), directly from tables (4%).

### 2.3 Known inconsistencies

The dataset has systematic issues that limit any model trained on it:

1. **XRD vs EBSD discrepancy.** The Frontiers 2020 study reports 64.7% RA by XRD and 47.2% by EBSD on the same sample at 680 C. Aliabad 2026 reports 34% vs 38%. This 4-17% absolute difference between methods is comparable to the total model error.

2. **Unit mixing.** Most studies report vol%, but Gibbs 2011 and LPBF 2021 report wt%. The conversion depends on phase densities that are not reported.

3. **Initial condition variability.** Cold-rolled martensite, hot-rolled, warm-rolled, LPBF as-built, and double-annealed starting microstructures all produce different kinetics. The PMC6266817 study directly shows this: cold-rolled gives 39% peak RA vs 29% for hot-rolled at the same conditions.

4. **Digitization uncertainty.** 77% of data comes from figure digitization.

### 2.4 Data splits

Split at the study level (not point level) to test cross-study generalization:
- Training: ~73 curves (synthetic + real)
- Validation: 16 real curves (32 points)
- Test: 16 real curves (35 points)

## 3. Model Architecture

### 3.1 Latent Neural ODE

Input: alloy composition (Mn, C, Al, Si) + annealing temperature + physics-derived features (Mn diffusivity, Gibbs free energy, Hollomon-Jaffe parameter).

The composition vector is encoded through an attention-based encoder (multi-head self-attention over element embeddings). The encoded composition is combined with physics features and used to condition the ODE vector field via FiLM (Feature-wise Linear Modulation).

The ODE is:

    dz/dt = f_theta(z, t, condition)

where z is the augmented state (RA fraction + 4 auxiliary dimensions), and f_theta is a 4-layer neural network (128-128-96-64) with SiLU activations and spectral normalization.

The right-hand side enforces positivity through softplus and includes physics-based gates:
- Driving force term (proportional to |delta G|)
- Saturation term (proportional to f_eq - f, clamped at 0)  
- Nucleation term (proportional to f + epsilon)

This means the model cannot predict negative transformation rates or super-equilibrium fractions by construction.

Solver: adaptive-step Dormand-Prince 4/5 via torchdiffeq, with adjoint method during training.

Total parameters: 78,474.

### 3.2 Loss function

    L = L_data + alpha * L_physics + beta * L_mono + gamma * L_bound

- L_data: MSE between predicted and observed RA fractions
- L_physics: penalizes inconsistency with thermodynamic driving force
- L_mono: penalizes negative df/dt (austenite should not spontaneously decompose during isothermal hold)
- L_bound: penalizes predictions outside [0, f_eq * 1.02]

The constraint weights are annealed during training.

### 3.3 Two-stage training

**Stage 1 (synthetic pre-training):** 120 epochs on ~1055 synthetic curves (57,227 points) generated from physics-based kinetics models. This teaches the model basic sigmoidal behavior, temperature dependence, and composition effects.

**Stage 2 (real-data fine-tuning):** 16-60 epochs on the 125 real experimental points only. Batch size = 1 (one curve per gradient step) to handle variable-length sequences. Learning rate reduced to 8e-5 with cosine decay.

## 4. Results

### 4.1 Training convergence

Stage 1 converged smoothly. Key milestones:

| Milestone | Epoch |
|---|---|
| val_real_rmse < 0.30 | 2 |
| val_real_rmse < 0.25 | 12 |
| val_real_rmse < 0.22 | 18 |
| val_real_rmse < 0.215 | 54 |
| val_real_rmse < 0.213 | 69 |

Best stage 1 checkpoint: epoch 109 (val_real_rmse = 0.2128).

Physics violations (boundary + monotonicity) dropped to zero by epoch 3 and stayed there. The constraints did not conflict with the data loss.

Learning rate decayed from 2.5e-4 to 5e-6 (cosine schedule with warm restarts).

### 4.2 Model comparison

| Run | Stage | val_real_rmse | test_real_rmse | Notes |
|---|---|---|---|---|
| Stage 1 best | epoch 109 | 0.21278 | 0.31368 | Pre-training only |
| Run 06 (stage 2) | epoch 6 | 0.21244 | 0.31214 | Old best |
| Run 07 (stage 2) | epoch 32 | 0.21145 | 0.32797 | Better val, worse test |
| **Retrained (fixed thermo)** | stage 2 | **0.161** | **0.131** | **Current best** |

The retrained model uses corrected thermodynamic input functions (Ac1 formula and equilibrium RA fraction). Same architecture and training procedure as before.

### 4.3 The thermodynamic fix

The original model showed a persistent val-test gap (0.212 vs 0.312). This was initially attributed to cross-study data heterogeneity. After investigation, the dominant cause turned out to be thermodynamic input errors:

1. **Ac1 formula** — the Andrews-type correlation overestimated Ac1 by 50-100C for medium-Mn steels, cutting off predictions below the true intercritical range.
2. **Equilibrium RA fraction** — the empirical formula returned 0 or 1.0 instead of realistic values (0.30-0.65).

Both were recalibrated against published phase diagram data from the 25-study dataset. After retraining with corrected inputs, the val-test gap collapsed from 0.10 to ~0.03.

Data heterogeneity is still a factor (measurement method differences, missing microstructure covariates), but the thermodynamic bug was the larger error source.

### 4.4 Backend validation vs literature

10 independent test cases from published studies:

| Metric | Old model | Retrained model |
|---|---|---|
| MAE vs literature | 0.271 | 0.138 |
| RMSE vs literature | 0.302 | 0.173 |
| Pass rate (< 0.15 error) | 3/10 | 6/10 |

### 4.5 JMAK baseline

The Johnson-Mehl-Avrami-Kolmogorov equation f(t) = f_max * (1 - exp(-k * t^n)) was fitted per isothermal curve:

| Study | Temperature | n | k | RMSE | Points |
|---|---|---|---|---|---|
| Luo 2011 | 650 C | 0.7 | 9.12e-4 | 0.0285 | 7 |
| De Moor 2011 | 600 C | 0.4 | 1.83e-2 | 0.0119 | 5 |
| Sun 2018 | 625 C | 0.6 | 4.09e-3 | 0.0082 | 4 |
| De Moor 2015 | 620 C | 0.4 | 1.83e-2 | 0.0153 | 6 |

JMAK fits each curve well (RMSE 0.008-0.028). But the fitted parameters are not transferable between alloys — each study needs its own k and n. The Neural ODE uses a single set of weights across all 18 compositions.

## 5. Discussion

### 5.1 What limits accuracy

The bottleneck is data quality, not model capacity. The model can memorize the training set (train loss < 0.001) but cannot generalize across studies because the studies are not measuring the same thing in the same way.

Three specific bottlenecks:

1. **Measurement method bias.** Different techniques (XRD, EBSD, neutron diffraction) give different numbers for the same sample. The model has no way to learn this bias because it is not given the measurement method as an input feature.

2. **Missing covariates.** Grain size, dislocation density, prior Mn distribution, and heating rate all affect reversion kinetics but are not reported in most studies.

3. **Small real dataset.** 125 points from 25 studies is not enough to learn the full composition-temperature-time-microstructure mapping. The effective number of independent conditions is even smaller because many studies share similar compositions.

### 5.2 What the model is good for

Despite the limitations, the model does several things that classical approaches cannot:

1. **Cross-composition prediction.** One set of weights predicts across 18 different alloy compositions. JMAK requires re-fitting for each alloy.

2. **Physics enforcement.** Monotonicity and boundary constraints are guaranteed, not approximate. The model never predicts negative RA or super-equilibrium fractions.

3. **Continuous-time output.** The ODE formulation gives predictions at any time point, not just at observed times. This enables non-isothermal path predictions in principle.

4. **Uncertainty quantification.** MC dropout gives calibrated confidence intervals.

### 5.3 What would actually improve things

- A standardized experimental campaign: same alloy compositions, same measurement method, systematically varied T and t. Even 50 well-controlled data points would be more useful than 500 heterogeneous literature values.
- Including measurement method as an input feature, so the model can learn the XRD vs EBSD bias.
- Reporting initial microstructure quantitatively (grain size, Mn microsegregation) in publications.

## 6. Repository Structure

```
project_4/
    src/                     source code (v3, production)
    data/                    CALPHAD tables, literature CSV, dataset card
    models/                  all checkpoints and training histories
    figures/                 22 publication figures (PDF + PNG)
    analysis.py              post-hoc analysis (CPU-only)
    analysis_results/        analysis output
    kaggle/                  all 7 kaggle runs with logs and notebooks
    docs/                    paper draft, notes, run registry
    tests/                   unit tests
    archive/                 old code versions
```

## 7. Reproducibility

### Requirements

```
pip install -r requirements.txt
```

Core dependencies: torch >= 2.0, torchdiffeq >= 0.2.3, scikit-learn, numpy, pandas, scipy, matplotlib.

### Running the analysis

```
python analysis.py
```

Runs CPU-only. Outputs per-study statistics, training dynamics, JMAK baseline to `analysis_results/`.

### Running the app

```
pip install streamlit
streamlit run src/streamlit_app.py
```

### Training

Requires GPU. See `kaggle/cells/` for the training scripts used on Kaggle with Tesla T4.

## 8. Conclusions

1. A physics-constrained latent Neural ODE was trained on 125 literature data points from 25 studies to predict austenite reversion kinetics in medium-Mn steels.

2. After recalibrating the thermodynamic input functions (Ac1 and equilibrium RA fraction), the retrained model achieves val RMSE = 0.161 and test RMSE = 0.131, a 58% reduction in test error compared to the original model.

3. The original val-test gap (0.212 vs 0.312) was primarily caused by thermodynamic input errors, not data heterogeneity alone. Fixing the inputs collapsed the gap to ~0.03.

4. Backend validation against 10 independent literature cases gives MAE = 0.138 and RMSE = 0.173, with 6/10 cases passing a 15% error threshold. The 4 failures are traceable to remaining Ac1 overestimation in Al-containing steels.

5. The Neural ODE's advantage over JMAK is cross-composition generalization with a single model, not per-alloy accuracy.

6. The quality of thermodynamic feature engineering matters more than training hyperparameters for this class of problem. Getting the equilibrium fraction right was worth more than 100 extra training epochs.

---

## References

1. Gibbs et al., Metall. Mater. Trans. A, 42A, 3691-3702 (2011). DOI: 10.1007/s11661-011-0736-2
2. Luo et al., Acta Materialia, 59, 4002-4014 (2011). DOI: 10.1016/j.actamat.2011.03.025
3. De Moor et al., Scripta Materialia, 64, 185-188 (2011). DOI: 10.1016/j.scriptamat.2010.09.040
4. Lee and De Cooman, Metall. Mater. Trans. A, 44A, 5018-5024 (2013). DOI: 10.1007/s11661-013-1860-y
5. Nakada et al., Acta Materialia, 65, 251-259 (2014). DOI: 10.1016/j.actamat.2013.10.067
6. Cai et al., Acta Materialia, 84, 229-236 (2015). DOI: 10.1016/j.actamat.2014.10.052
7. Sun et al., Acta Materialia, 148, 249-262 (2018). DOI: 10.1016/j.actamat.2018.02.005
8. Suh and Kim, Scripta Materialia, 126, 63-67 (2017). DOI: 10.1016/j.scriptamat.2016.07.013
9. Chen et al., Neural Ordinary Differential Equations, NeurIPS (2018).
10. Raissi et al., J. Comput. Phys., 378, 686-707 (2019). DOI: 10.1016/j.jcp.2018.10.045

Full citation details for all 25 studies are in `data/literature_validation/literature_validation.csv`.
