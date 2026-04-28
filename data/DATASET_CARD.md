# Dataset Card: Medium-Mn Steel Austenite Reversion Literature Database

## overview

a curated collection of 125 experimental measurements of retained austenite fraction
in medium-Mn steels during intercritical annealing. extracted from 25 peer-reviewed studies
spanning 2010-2024.

this dataset was built because no ML-ready kinetics database exists for medium-Mn steels.
every data point has a DOI, source reference (table or figure number), measurement method,
and data quality flag.

## what it contains

the csv file `literature_validation.csv` has one row per measurement with columns:

| column | description | example |
|---|---|---|
| Mn | manganese content (wt%) | 7.1 |
| C | carbon content (wt%) | 0.10 |
| Al | aluminum content (wt%) | 0.0 |
| Si | silicon content (wt%) | 0.0 |
| Mo | molybdenum content (wt%) | 0.0 |
| Nb | niobium content (wt%) | 0.0 |
| T_celsius | annealing temperature (C) | 650 |
| t_seconds | holding time (seconds) | 604800 |
| f_RA | retained austenite fraction (0-1 scale) | 0.435 |
| f_RA_pct | retained austenite percentage | 43.5 |
| study_id | unique study identifier | gibbs_2011 |
| doi | digital object identifier | 10.1007/s11661-011-0736-2 |
| data_quality | how the value was extracted | table |
| source_ref | exact table/figure reference | Table III |
| method | measurement technique | neutron_diffraction |
| unit | original reporting unit | wt_pct |
| initial_condition | starting microstructure | cold-rolled, 50% reduction |
| provenance | data tier label | experimental |
| year | publication year | 2011 |
| journal | publication venue | Metall. Mater. Trans. A |

## coverage

- 25 studies, 18 unique alloy compositions
- Mn: 3.93 - 12.0 wt%
- C: 0.0 - 0.40 wt%
- Al: 0.0 - 4.3 wt%
- temperature: 25 - 1000 C
- time: 0 - 604800 s (up to 1 week)
- RA fraction: 0 - 64.7%
- measurement methods: XRD (93%), EBSD (4%), neutron diffraction (3%)

## data quality flags

- `table` (5 points, 4%): copied directly from a published table. highest confidence.
- `text_reported` (24 points, 19%): stated explicitly in the paper text.
- `digitized_figure` (96 points, 77%): extracted from figures. estimated +/-2-5% uncertainty.

## known issues and limitations

1. XRD vs EBSD discrepancy: same sample measured by XRD and EBSD can differ by 4-17.5% absolute.
   see frontiers_2020 (64.7% XRD vs 47.2% EBSD at 680C) and aliabad_2026 (34% vs 38%).

2. unit mixing: most studies report vol%, some report wt%. we normalize to vol fraction
   but the wt%->vol% correction depends on phase densities that are rarely reported.

3. initial condition variability: cold-rolled, hot-rolled, warm-rolled, LPBF, double-annealed
   all behave differently but are pooled together.

4. digitization uncertainty: 77% of points come from figure digitization with unknown systematic error.

5. some studies have approximate DOIs (PMC IDs or informal identifiers) because
   the original DOI could not be confirmed from the available information.

## how to use

```python
import pandas as pd
df = pd.read_csv('literature_validation.csv')

# filter by measurement method
xrd_only = df[df['method'] == 'XRD']

# filter by data quality
high_quality = df[df['data_quality'].isin(['table', 'text_reported'])]

# get all isothermal kinetic curves (multiple time points at same T)
for (sid, T), group in df.groupby(['study_id', 'T_celsius']):
    if len(group) >= 3:
        print(f"{sid} @ {T}C: {len(group)} time points")
```

## citation

if you use this dataset, cite the original papers (DOIs in the csv) and this project:
```
Physics-Constrained Latent Neural ODE for Austenite Reversion Kinetics
in Medium-Mn Steels (2026)
```

## source code

the full extraction code is in `src/real_data.py` which contains the raw data entries
with inline citations and extraction notes for every single value.
