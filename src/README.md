# Physics-Constrained Neural ODE for Austenite Reversion

This folder is the clean handoff package for the medium-Mn steel austenite reversion project.
It keeps the core source code, curated figures, tests, and static reference data in one place,
without transient experiment artifacts.

## What Is Included

- Core model, loss, training, and visualization code
- Literature-backed real-data ingestion and preprocessing
- Thermodynamic helper tables and literature validation data
- Publication-oriented figures already generated in `figures/`
- Tests for data integrity and physics constraints

## What Is Intentionally Excluded

- Transient training outputs
- Generated synthetic training CSV files
- Temporary caches and logs
- Ad hoc experiment notes

## Quick Start

```bash
pip install -r requirements.txt

python main.py --generate-data
python main.py --train --epochs 200
python main.py --figures
```

Run the web app with:

```bash
python main.py --app
```

## Data Notes

The project combines published experimental measurements with synthetic augmentation.
User-supplied CSV files can be added to `data/user_experimental/`.

Expected columns:

```csv
Mn,C,Al,Si,T_celsius,t_seconds,f_RA
7.0,0.1,1.5,0.5,650,3600,0.35
```

Optional columns include `method`, `data_quality`, `source_ref`, and `initial_condition`.

## Project Layout

```text
medium_mn_neural_ode_clean_submission/
|-- config.py
|-- main.py
|-- model.py
|-- trainer.py
|-- losses.py
|-- data_generator.py
|-- thermodynamics.py
|-- real_data.py
|-- visualizations.py
|-- publication_pipeline.py
|-- data/
|   |-- calphad_tables/
|   |-- literature_validation/
|   `-- user_experimental/
|-- figures/
|-- models/
|   `-- checkpoints/
|-- tests/
`-- docs/
```

## Recommended Reading Order

1. `docs/PROJECT_STATUS.md`
2. `data_sources.md`
3. `config.py`
4. `main.py`

## Citation Placeholder

```bibtex
@article{anonymous2025physicsnodeaustenite,
  title={Physics-Constrained Neural ODE for Austenite Reversion in Medium-Mn Steels},
  author={Anonymous},
  journal={Draft},
  year={2025}
}
```
