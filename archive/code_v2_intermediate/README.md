# Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics in Medium-Mn Steels

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU Optional](https://img.shields.io/badge/GPU-optional-orange.svg)]()

## Architecture

| Component | Method |
|---|---|
| Composition encoding | Multi-head self-attention (4 heads) over element tokens |
| ODE conditioning | FiLM (Feature-wise Linear Modulation) |
| ODE function | Augmented Neural ODE (4 auxiliary dimensions) |
| Training stability | Spectral normalization on all linear layers |
| Solver | Adjoint method (O(1) memory backprop) |
| UQ — epistemic | SWAG (Stochastic Weight Averaging Gaussian, rank-20) |
| UQ — aleatoric | Concrete Dropout (learned per-layer via variational inference) |
| Loss balancing | Homoscedastic uncertainty weighting (Kendall et al. 2018) |
| Data loss | 0.7×Huber + 0.3×MSE (robust to outliers) |
| Schedule optimization | Optuna TPE (single + multi-objective Pareto) |
| Interpretability | PySR symbolic regression (equation discovery) |
| Mixed precision | torch.amp when GPU available |

## Quick Start

```bash
pip install -r requirements.txt

# Complete pipeline (generate data → train → optimize → explain → figures)
python main.py --all

# With GPU
python main.py --all --device cuda

# Individual steps
python main.py --generate-data          # 3-tier: real + calibrated + exploratory
python main.py --train --epochs 500     # Train PhysicsNODE
python main.py --optimize               # Optuna schedule search
python main.py --explain                # SHAP + sensitivity
python main.py --symbolic               # PySR equation discovery
python main.py --figures                # Publication figures

# Real-data-only mode (no synthetic augmentation)
python main.py --generate-data --real-only
python main.py --train --real-only --epochs 200

# Interactive web app
python main.py --app
```

## Data Sources

The experimental database contains **150+ data points** from **25 studies** in peer-reviewed journals.
Every data point is tagged with provenance, DOI, measurement method, and data quality flag.

### Three-Tier Data Pipeline

| Tier | Provenance Tag | Source | Weight in Loss |
|------|---------------|--------|----------------|
| 1 (Real) | `experimental` | Published papers, user CSV | 3× (configurable) |
| 2 (Calibrated) | `synthetic_calibrated` | JMAK fitted to real endpoints | 1× |
| 3 (Exploratory) | `synthetic_exploratory` | LHS sampled, broad coverage | 1× |

### Key Papers in Database

| Study | Alloy | DOI | Points | Quality |
|-------|-------|-----|--------|---------|
| Gibbs et al. 2011 | Fe-7.1Mn-0.1C | `10.1007/s11661-011-0736-2` | 5 | table |
| Luo et al. 2011 | Fe-5Mn-0.2C | `10.1016/j.actamat.2011.03.025` | 7 | text/figure |
| De Moor et al. 2011 | Fe-7.1Mn-0.1C | `10.1016/j.scriptamat.2010.09.040` | 5 | figure |
| De Moor et al. 2015 | Fe-7Mn-0.1C | `10.2355/isijinternational.55.234` | 6 | figure |
| Lee & De Cooman 2013 | Fe-5.8Mn-0.12C-2Al | `10.1007/s11661-013-1860-y` | 6 | figure |
| Nakada et al. 2014 | Fe-6Mn | `10.1016/j.actamat.2013.10.067` | 7 | figure |
| Shi et al. 2010 | Fe-5Mn-0.2C | `10.1016/j.scriptamat.2010.06.023` | 5 | figure |
| Cai et al. 2015 | Fe-10Mn-0.3C-2Al | `10.1016/j.actamat.2014.10.052` | 4 | figure |
| Han et al. 2014 | Fe-9Mn-0.05C | `10.1016/j.msea.2015.02.055` | 6 | figure |
| Hu & Luo 2017 | Fe-7Mn-0.1C-0.5Al | `10.1016/j.jallcom.2017.07.174` | 4 | figure |
| Sun et al. 2018 | Fe-12Mn-0.05C | `10.1016/j.actamat.2018.02.005` | 8 | figure |
| Hausman et al. 2017 | Fe-6Mn-0.3C | `10.1016/j.msea.2016.12.055` | 4 | figure |
| + 13 additional studies | Various | Various | ~75 | mixed |

## Adding Your Own Data

Drop CSV files into `data/user_experimental/`. Expected columns:

```csv
Mn,C,Al,Si,T_celsius,t_seconds,f_RA
7.0,0.1,1.5,0.5,650,3600,0.35
```

Optional columns: `method`, `data_quality`, `source_ref`, `initial_condition`

All user data is automatically tagged with `provenance='user_provided'` and included in training.

## Training on GPU (Google Colab)

Neural ODE training is **10-50× slower on CPU** than GPU. Use the provided Colab notebook:

1. Upload `notebooks/train_colab.py` to Google Colab
2. Select GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Upload your project files or clone from GitHub
4. Run all cells

The notebook handles:
- Data generation
- Full training (500-1000 epochs)
- Optuna hyperparameter search (50-100 trials)
- SWAG posterior estimation
- SHAP explainability
- All publication figures
- Checkpoint saving to Google Drive

## Project Structure

```
medium_mn_neural_ode/
├── config.py                  # Centralized config with auto GPU detection
├── real_data.py               # 150+ experimental data points, 25 studies
├── data_generator.py          # 3-tier pipeline: real → calibrated → exploratory
├── thermodynamics.py          # CALPHAD + polynomial fallbacks (Ac1/Ac3, f_eq, ΔG)
├── features.py                # D_Mn, Md30, Hollomon-Jaffe, JMAK
├── model.py                   # Attention-conditioned augmented latent ODE
├── losses.py                  # Homoscedastic uncertainty + GradNorm
├── trainer.py                 # AMP, SWAG, adjoint, warm restarts
├── optimizer_annealing.py     # Bayesian schedule optimization
├── explainability.py          # SHAP, sensitivity, physics validation
├── symbolic_regression.py     # PySR equation discovery
├── visualizations.py          # 11 publication figures
├── streamlit_app.py           # Interactive web app
├── main.py                    # CLI pipeline orchestrator
├── data_sources.md            # Full provenance documentation
├── notebooks/train_colab.py   # GPU training notebook for Colab
├── tests/test_real_data.py    # 20 data integrity tests
├── tests/test_physics.py      # 36 physics + architecture tests
└── requirements.txt
```

## Key Innovations

1. **Attention-conditioned composition encoder** — each alloying element is embedded as a token with learned element-specific embeddings, then processed by multi-head self-attention. This captures interactions (Mn-C synergy, Al-Mn competition) that linear features miss.

2. **Augmented ODE** — the state vector is extended with 4 auxiliary dimensions that learn latent dynamics, avoiding the topology limitations of standard Neural ODEs (Dupont et al. 2019).

3. **FiLM conditioning** — the ODE function is modulated at every hidden layer by composition-dependent affine transforms, not just concatenated at input.

4. **Three-tier provenance-tracked data** — real experimental data (3× loss weight) anchors the model; calibrated synthetic fills gaps; exploratory synthetic provides coverage. Every data point carries provenance tags.

5. **SWAG posterior** — after SWA begins (epoch 400), model weights are collected every 5 epochs to build a low-rank Gaussian posterior over parameters. This gives calibrated uncertainty without ensemble overhead.

6. **Homoscedastic uncertainty weighting** — instead of hand-tuning λ₁, λ₂, λ₃, λ₄, each loss learns its own log-variance (Kendall et al. 2018), automatically balancing data vs physics.

7. **Symbolic regression** — after training, PySR recovers an interpretable closed-form equation from the Neural ODE's input-output mapping, bridging black-box and white-box models.

## Tests

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run specific test modules
python -m pytest tests/test_real_data.py -v
python -m pytest tests/test_physics.py -v
```

## Notes on Results Provenance

- **Experimental data** (provenance=`experimental`): Direct from published tables/figures
- **Calibrated synthetic** (provenance=`synthetic_calibrated`): JMAK curves fitted to real endpoints
- **Exploratory synthetic** (provenance=`synthetic_exploratory`): LHS-sampled compositions, physics-based curves
- Figures overlay real data points (solid circles) on model predictions when available

## Citation

```bibtex
@article{anonymous2025physicsnodeaustenite,
    title={Attention-Conditioned Latent Neural ODE for Austenite Reversion
           Kinetics with SWAG Uncertainty and Symbolic Equation Discovery},
    author={Anonymous},
    journal={Acta Materialia},
    year={2025}
}
```
