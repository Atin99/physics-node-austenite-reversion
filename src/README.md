# Source Code: Physics-Constrained Neural ODE for Austenite Reversion

This directory contains the core source code for the medium-Mn steel austenite reversion project.

## Architecture Overview

```
Composition (Mn, C, Al, Si) + Temperature
         │
    ┌────▼────┐
    │ Encoder  │  Attention-based composition encoder (4-head)
    └────┬────┘
         │  FiLM conditioning
    ┌────▼─────────────┐
    │ Latent Neural ODE │  dz/dt = f_θ(z, t, condition)
    │  4-layer MLP      │  128-128-96-64, SiLU, spectral norm
    │  Physics gates    │  Driving force × saturation × nucleation
    └────┬─────────────┘
         │  Dormand-Prince 4/5 solver
    ┌────▼────┐
    │ Decoder  │  z(t) → f_RA(t)
    └─────────┘
```

## Module Guide

| File | Purpose |
|---|---|
| `model.py` | Latent Neural ODE architecture (78,474 params) |
| `trainer.py` | Training loop with physics-constrained loss, early stopping, checkpointing |
| `losses.py` | Multi-component loss: L_data + α·L_physics + β·L_mono + γ·L_bound |
| `data_generator.py` | Three-tier data pipeline: real + calibrated synthetic + exploratory synthetic |
| `real_data.py` | 125 literature data points with inline DOI citations for every value |
| `thermodynamics.py` | Ac1/Ac3 correlations, equilibrium RA fraction, Gibbs driving force |
| `features.py` | JMAK kinetics, Arrhenius rate, Mn diffusivity, Hollomon-Jaffe parameter |
| `config.py` | All hyperparameters, paths, composition bounds, physics constants |
| `streamlit_app.py` | Interactive web app for prediction exploration |
| `visualizations.py` | Publication-quality figure generation (10+ figure types) |
| `publication_pipeline.py` | Automated pipeline for generating all manuscript figures |
| `main.py` | CLI entry point: `--generate-data`, `--train`, `--figures`, `--app` |

## Data Pipeline (Three Tiers)

The project uses a provenance-aware data pipeline implemented in `data_generator.py`:

### Tier 1: Real Experimental Data
- Source: `real_data.py` (hardcoded with inline citations) + user CSVs from `data/user_experimental/`
- 125 points from 25 studies, tagged `provenance='experimental'`
- Upweighted 5× in the training loss

### Tier 2: Calibrated Synthetic Data
- JMAK kinetics curves calibrated to bracket real data endpoints
- Latin Hypercube Sampling in composition/temperature gaps
- Default: 500 curves, tagged `provenance='synthetic_calibrated'`

### Tier 3: Exploratory Synthetic Data
- Broad LHS sampling across the full medium-Mn design space
- Teaches general kinetic behavior (sigmoidal shapes, T dependence)
- Default: 2000 curves, tagged `provenance='synthetic_exploratory'`

Every row carries a `provenance` tag used for loss weighting and evaluation filtering.

## Two-Stage Training

1. **Stage 1 (pre-training):** Full three-tier dataset (~150K points). Learns physics-compliant kinetics shapes.
2. **Stage 2 (fine-tuning):** Real data only (~227 points). Corrects toward experimental accuracy.

See `kaggle/cells/` for the actual training scripts run on Tesla T4 GPUs.

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data
python main.py --generate-data

# Train (requires GPU for practical speeds)
python main.py --train --epochs 200

# Generate publication figures
python main.py --figures

# Run interactive app
python main.py --app
```

## Data Directory Layout

```
data/
├── synthetic/                 Generated training data (Tier 2 + 3)
│   ├── synthetic_kinetics.csv   Full combined dataset (~150K rows)
│   ├── full_dataset.csv         Same as above (legacy name)
│   ├── train.csv                Training split
│   ├── val.csv                  Validation split
│   └── test.csv                 Test split
├── literature_validation/     Tier 1 real data
│   └── literature_validation.csv  125 experimental points
├── calphad_tables/            Pre-computed thermodynamic lookups
│   ├── f_eq_table.npy           Equilibrium RA fraction grid
│   ├── delta_G_table.npy        Gibbs driving force grid
│   ├── Mn_grid.npy / C_grid.npy / T_grid.npy
└── user_experimental/         Drop custom CSVs here
```

## Recommended Reading Order

1. `config.py` — understand the hyperparameters and bounds
2. `data_generator.py` — the three-tier data pipeline
3. `model.py` — the Neural ODE architecture
4. `losses.py` — the physics-constrained loss function
5. `trainer.py` — the training loop
6. `real_data.py` — where every experimental value comes from

## User-Provided Data

Drop CSV files into `data/user_experimental/` with columns:

```csv
Mn,C,Al,Si,T_celsius,t_seconds,f_RA
7.0,0.1,1.5,0.5,650,3600,0.35
```

Optional columns: `method`, `data_quality`, `source_ref`, `initial_condition`.
