# Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics

A physics-informed Neural ODE for predicting retained austenite fraction in medium-Mn steels (3–12 wt% Mn) during intercritical annealing. Trained on 125 experimental measurements from 25 peer-reviewed studies (2010–2024), augmented with physics-calibrated synthetic kinetics data.

Built for submission to *Computational Materials Science* / *Acta Materialia*.

---

## The Problem

Medium-Mn steels use the TRIP (Transformation-Induced Plasticity) effect to achieve exceptional strength-ductility combinations. The retained austenite (RA) fraction after intercritical annealing is the key design parameter — but predicting it remains hard:

- **No ML-ready kinetics database exists.** Data is scattered across journals, measured with different techniques (XRD, EBSD, neutron diffraction), reported in different units (vol% vs wt%), and collected on alloys with varying processing histories.
- **Classical JMAK models don't generalize.** They fit individual isothermal curves well but require per-alloy parameter refitting. A single JMAK model cannot predict across 18 different compositions.
- **Measurement methods disagree.** The same specimen can give 64.7% RA by XRD and 47.2% by EBSD (Frontiers 2020) — a 17.5% absolute discrepancy, comparable to total model error.

This project tackles all three problems.

---

## What This Does

Takes alloy composition (Mn, C, Al, Si), annealing temperature, and time as inputs. Returns predicted retained austenite fraction with physics constraints enforced (monotonicity, boundary conditions, thermodynamic consistency).

The model is a **latent Neural ODE** — it learns a continuous-time vector field for the transformation kinetics rather than assuming a fixed functional form. Physics constraints are embedded directly in the loss function and the ODE architecture (driving force gates, saturation terms, positivity enforcement via softplus).

---

## Results

### Point-Level Evaluation (124 experimental points, 25 studies)

| Metric | Value |
|---|---|
| **RMSE** | **0.135** |
| **MAE** | **0.104** |
| **R²** | **+0.013** |
| Test R² (17 held-out points) | **+0.378** |
| Studies with R² > 0 | 12 / 21 |
| Monotonicity violations | 0 / 29 curves |
| Boundary violations | 0 |

### Training Progression

| Stage | Description | val_real_rmse | test_real_rmse |
|---|---|---|---|
| Stage 1 only | Synthetic + real pre-training (120 ep) | 0.213 | 0.314 |
| Stage 2 (60 ep) | Real-data fine-tuning, fixed thermodynamics | 0.161 | 0.131 |
| Stage 2 Extended (200 ep) | Cosine warm restarts, no early stopping | **0.157** | **0.136** |

### Key Breakthroughs

1. **Thermodynamic recalibration** — Correcting the Ac1 correlation and equilibrium RA fraction formulae reduced RMSE by **57%** (0.312 → 0.136). This single fix mattered more than all architecture/hyperparameter changes combined.
2. **CALPHAD validation** — Constructed a Fe-Mn-C TDB database (Huang 1989, Djurovic 2011). Found that standard CALPHAD *without* magnetic ordering gives Ac1 = 395°C (constant) vs. empirical 417–606°C. Magnetic ordering corrections are essential for medium-Mn steels.
3. **Two-stage transfer learning** — Synthetic pre-training teaches general kinetics behavior; real-data fine-tuning corrects toward experimental measurements.

---

## The Dataset

### Tier 1: Real Experimental Data (125 points)

Curated from 25 peer-reviewed studies. Every data point has:
- DOI and source reference (table/figure number)
- Measurement method (XRD, EBSD, neutron diffraction)
- Data quality flag (`table`, `text_reported`, `digitized_figure`)
- Provenance tag for loss weighting

| Property | Coverage |
|---|---|
| Mn | 3.93 – 12.0 wt% |
| C | 0.0 – 0.40 wt% |
| Al | 0.0 – 4.3 wt% |
| Temperature | 25 – 1000°C |
| Time | 0 – 604,800 s (up to 1 week) |
| RA fraction | 0 – 64.7% |
| Methods | XRD (93%), EBSD (4%), neutron (3%) |

Full documentation: [`data/DATASET_CARD.md`](data/DATASET_CARD.md)

### Tier 2: Calibrated Synthetic Data (500 curves)

Physics-based JMAK kinetics curves calibrated to bracket real data endpoints. Generated via Latin Hypercube Sampling in composition-temperature gaps not covered by real data. Tagged with `provenance='synthetic_calibrated'`.

### Tier 3: Exploratory Synthetic Data (2000 curves)

Broad LHS-sampled compositions across the full design space with JMAK kinetics + realistic noise. Teaches the model general sigmoidal behavior, temperature dependence, and composition sensitivity. Tagged with `provenance='synthetic_exploratory'`.

### Why Synthetic Data Matters

The real dataset has 125 points from 18 unique compositions — not enough to train a complex Neural ODE from scratch. The three-tier pipeline solves this:

| Tier | Curves | Points | Role |
|---|---|---|---|
| Real (Tier 1) | 105 | 227 | Ground truth, 5× loss upweighting |
| Calibrated Synthetic (Tier 2) | 500 | 30,000 | Fill composition/temperature gaps |
| Exploratory Synthetic (Tier 3) | 2,000 | 120,000 | Teach general kinetic shapes |
| **Total** | **2,605** | **~150,000** | |

Real data is upweighted 5× in the provenance-aware loss function, so the model prioritizes experimental accuracy while using synthetic data as a physics-based regularizer.

---

## Two-Stage Training Protocol

### Stage 1: Synthetic Pre-Training (120 epochs)

Trains on the **full three-tier dataset** (150K+ points). This stage teaches the model:
- Sigmoidal JMAK-like curve shapes
- Temperature dependence of transformation kinetics
- Composition effects (Mn accelerates, Al modifies intercritical range)
- Physics constraints (monotonicity, boundary conditions)

The model learns the *physics* of austenite reversion without being limited by the small real dataset.

**Config:** batch_size=24, lr=2.5e-4, cosine schedule, adjoint ODE solver, provenance-aware loss with 5× real upweighting.

### Stage 2: Real-Data Fine-Tuning (60–200 epochs)

Loads the best Stage 1 checkpoint and fine-tunes on **real experimental data only** (227 points). This corrects the model from "physics-plausible" to "experimentally accurate."

**Config:** batch_size=1 (per-curve), lr=3e-5, cosine warm restarts (T₀=40, T_mult=2), checkpoint on `val_real_rmse`.

The separation matters: Stage 1 gives a well-initialized model that already respects physics constraints. Stage 2 adapts it to the actual (noisy, heterogeneous) experimental landscape. Without Stage 1, the model would overfit on 227 points or fail to converge.

---

## Model Architecture

**Latent Neural ODE** with physics-constrained vector field.

| Component | Details |
|---|---|
| Encoder | Attention-based composition encoder (4-head self-attention over element embeddings) |
| Conditioning | FiLM (Feature-wise Linear Modulation) |
| ODE vector field | 4-layer MLP (128-128-96-64), SiLU activation, spectral normalization |
| Augmented state | RA fraction + 4 auxiliary dims |
| Physics gates | Driving force gate (∝ \|ΔG\|), saturation gate (∝ f_eq − f), nucleation gate (∝ f + ε) |
| Solver | Dormand-Prince 4/5 (adaptive step), torchdiffeq |
| Parameters | 78,474 |

### Loss Function

```
L = L_data + α·L_physics + β·L_mono + γ·L_bound
```

- **L_data**: MSE on observed austenite fractions (provenance-weighted)
- **L_physics**: Thermodynamic driving force consistency
- **L_mono**: Penalizes negative df/dt (austenite cannot spontaneously decompose)
- **L_bound**: Penalizes predictions outside [0, f_eq]

Constraint weights are annealed during training.

---

## Running the App

```bash
pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

Or use `launch.bat` (Windows) / `launch.ps1` (PowerShell).

Interactive web app for exploring predictions: enter a composition + temperature, get a kinetic curve with uncertainty bands.

## Running Validation

```bash
python evaluate_comprehensive.py    # per-study metrics, parity plots, R² breakdown
python validate_calphad.py          # CALPHAD vs empirical thermodynamic comparison
python ablation_study.py            # physics constraint effectiveness analysis
python analysis.py                  # full post-hoc analysis (training dynamics, JMAK baseline)
```

All scripts run CPU-only. Tests predictions against all 125 experimental points, checks monotonicity, boundary conditions, and generates publication-ready figures.

## Regenerating Data

```bash
cd src
python -c "from data_generator import build_full_dataset; from config import get_config; build_full_dataset(get_config())"
```

Or use `python main.py --generate-data` to regenerate synthetic data and save to `data/synthetic/`.

## Training (Requires GPU)

See `kaggle/cells/` for the training scripts used on Kaggle with Tesla T4. The complete run registry is in [`docs/RUN_REGISTRY.md`](docs/RUN_REGISTRY.md).

---

## Folder Structure

```
project_4/
├── src/                        Core source code
│   ├── model.py                Latent Neural ODE architecture (78K params)
│   ├── trainer.py              Training loop with physics-constrained loss
│   ├── losses.py               Multi-component loss (data + physics + mono + bound)
│   ├── data_generator.py       Three-tier data pipeline (real + calibrated + exploratory)
│   ├── real_data.py            125 literature data points with inline citations
│   ├── thermodynamics.py       Ac1/Ac3, equilibrium RA, driving force calculations
│   ├── config.py               All hyperparameters and paths
│   ├── streamlit_app.py        Interactive web application
│   ├── visualizations.py       Publication figure generation
│   └── features.py             JMAK, Arrhenius, diffusivity calculations
│
├── data/
│   ├── DATASET_CARD.md         Full dataset documentation
│   ├── synthetic/              Generated training data (3-tier, ~150K points)
│   ├── literature_validation/  Curated experimental CSV (125 points, 25 studies)
│   ├── calphad_tables/         Pre-computed thermodynamic lookup tables (f_eq, ΔG)
│   └── user_experimental/      Drop user CSV files here for custom predictions
│
├── models/                     All checkpoints (stage1, stage2, retrained, extended)
├── figures/                    Publication-ready figures (PDF + PNG)
├── analysis.py                 Post-hoc analysis script (CPU-only)
├── analysis_results/           Output from analysis.py
├── evaluate_comprehensive.py   Per-study evaluation with parity plots
├── validate_calphad.py         CALPHAD vs empirical comparison
├── ablation_study.py           Physics constraint ablation
│
├── kaggle/
│   ├── cells/                  Training scripts for Kaggle GPU sessions
│   └── runs/                   All 8+ Kaggle runs with logs, notebooks, result zips
│
├── docs/
│   ├── PAPER_DRAFT.md          Full manuscript draft
│   ├── MATHEMATICAL_SUPPLEMENT.md  All equations, constants, derivations
│   ├── TECHNICAL_REPORT.md     Detailed technical report
│   ├── FINAL_RESULTS.md        Final metrics and publication readiness assessment
│   ├── RUN_REGISTRY.md         Complete training run registry
│   ├── PROJECT_DEFENSE_QBANK.md  Viva question bank with answers
│   └── PROJECT_CONCLUSION.md   Honest assessment and limitations
│
├── notebooks/                  Colab training script, Kaggle instructions
├── tests/                      Unit tests (physics constraints, data integrity)
└── archive/                    Old code versions (v1, v2)
```

---

## Known Limitations

1. **Measurement method bias** — XRD vs EBSD discrepancies (4–17% absolute) set a noise floor no model can eliminate.
2. **Al-containing steels** — The Ac1 correlation was designed for C-Mn steels and overestimates for Fe-Mn-Al-C. Studies like PMC6266817 (R² = −11.7) are poorly predicted.
3. **Missing covariates** — Grain size, cold-rolling reduction, initial Mn distribution affect kinetics but are rarely reported. The model cannot distinguish between different processing histories for the same nominal composition.
4. **Digitization uncertainty** — 77% of data points come from figure digitization with ±2–5% estimated uncertainty.

---

## Requirements

```bash
pip install -r requirements.txt
```

Core: `torch >= 2.0`, `torchdiffeq >= 0.2.3`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `streamlit`.

---

## Citation

```bibtex
@article{datta2026physicsnodeaustenite,
  title={Physics-Constrained Latent Neural ODE for Austenite Reversion
         Kinetics in Medium-Mn Steels},
  author={Datta, Atin},
  journal={Draft — Jadavpur University, Dept. of Metallurgy and Materials Engineering},
  year={2026}
}
```
