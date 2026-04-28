# Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics

Medium-Mn steel transformation kinetics modeling using a physics-informed Neural ODE trained on literature-derived data.

## what this is

A kinetics predictor for austenite reversion in medium-Mn steels (3-12 wt% Mn). Takes alloy composition, annealing temperature, and time as inputs. Returns predicted retained austenite fraction with physics constraints enforced.

The model was trained on 125 experimental measurements from 25 peer-reviewed studies (2010-2024). Every data point has a DOI, source reference, measurement method, and quality flag. See `data/DATASET_CARD.md` for full documentation.

The hard part was not finding papers. It was turning heterogeneous, partially incompatible literature evidence into something a kinetics model could learn from.

## results

| metric | old model (broken thermo) | retrained model (fixed thermo) |
|---|---|---|
| val_real_rmse | 0.212 | 0.161 |
| test_real_rmse | 0.312 | 0.131 |

### backend validation against literature (10 independent cases)

| metric | value |
|---|---|
| MAE vs published data | 0.138 (13.8%) |
| RMSE vs published data | 0.173 (17.3%) |
| monotonicity violations | 3 / 294 (negligible) |
| boundary violations | 0 |

The old model had a persistent val-test gap (0.21 vs 0.31) that looked like a data heterogeneity problem. Turned out the thermodynamic input functions (Ac1 formula, equilibrium RA fraction) were miscalibrated for medium-Mn compositions. Fixing those and retraining cut the test RMSE by 58%.

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
python validate_model.py
```

Runs CPU-only. Tests predictions against 10 known literature values, checks monotonicity, boundary conditions, temperature/composition/time dependence.

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
