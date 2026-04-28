# Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics

Medium-Mn steel transformation kinetics modeling using a physics-informed Neural ODE trained on literature-derived data.

## what this is

A kinetics predictor for austenite reversion in medium-Mn steels (3-12 wt% Mn). Takes alloy composition, annealing temperature, and time as inputs. Returns predicted retained austenite fraction with physics constraints enforced.

The model was trained on 125 experimental measurements from 25 peer-reviewed studies (2010-2024). Every data point has a DOI, source reference, measurement method, and quality flag. See `data/DATASET_CARD.md` for full documentation.

The hard part was not finding papers. It was turning heterogeneous, partially incompatible literature evidence into something a kinetics model could learn from.

## results

| metric | Stage 1 only | Stage 2 (60ep) | Stage 2 Extended (200ep) |
|---|---|---|---|
| val_real_rmse | 0.213 | 0.161 | **0.157** |
| test_real_rmse | 0.314 | 0.131 | **0.136** |
| overall R2 | -4.27 | -0.096 | **+0.013** |
| median per-study R2 | N/A | -0.149 | **+0.205** |

### point-level evaluation (124 experimental points, 25 studies)

| metric | value |
|---|---|
| RMSE | 0.135 |
| MAE | 0.104 |
| R2 | +0.013 |
| studies with R2 > 0 | 12 / 21 |
| monotonicity violations | 0 / 29 |
| boundary violations | 0 |

The key breakthrough was thermodynamic recalibration (Ac1/f_eq corrections), which reduced RMSE by 57%. Extended 200-epoch training with cosine warm restarts further improved R2 from negative to positive.

CALPHAD integration with pycalphad + Fe-Mn-C TDB (Huang 1989, Djurovic 2011) revealed that magnetic ordering corrections are essential for medium-Mn steels - simple CALPHAD gives Ac1 = 395 C (constant) vs. empirical 417-606 C.

## the dataset

125 experimental measurements from 25 peer-reviewed studies (2010-2024).

Composition coverage: Mn 3.93-12%, C 0-0.40%, Al 0-4.3%, T 25-1000C, time 0-604800s.

Measurement methods: XRD (93%), EBSD (4%), neutron diffraction (3%).

## running the app

```
pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

or use `launch.bat` on Windows.

## running validation

```
python evaluate_comprehensive.py    # per-study metrics, parity plots
python validate_calphad.py          # CALPHAD vs empirical comparison
python ablation_study.py            # physics constraint analysis
```

All scripts run CPU-only. Tests predictions against all 125 experimental points, checks monotonicity, boundary conditions, and generates publication figures.

## running the analysis

```
python analysis.py
```

Generates per-study stats, training dynamics, JMAK baseline comparisons.

## folder structure

```
project_4/
  src/                 - source code (model, training, thermodynamics, app)
  data/                - CALPHAD tables, literature validation CSV, dataset card
  models/              - all checkpoints (stage1, stage2, retrained)
  figures/             - publication figures
  analysis.py          - post-hoc analysis script (CPU only)
  analysis_results/    - output from the analysis script
  notebooks/           - colab training script, kaggle run instructions
  kaggle/              - all kaggle runs with logs notebooks and result zips
  docs/                - paper draft, technical report, conclusion, run registry
  tests/               - unit tests
  archive/             - old code versions
```

## requirements

```
pip install -r requirements.txt
```
