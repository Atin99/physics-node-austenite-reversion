# Data Sources — Full Provenance Documentation

Every experimental data point in this project can be traced to a specific published source.
This document lists all sources, their DOIs, and what data was extracted from each.

## Data Quality Flags

| Flag | Meaning | Uncertainty |
|---|---|---|
| `table` | Value copied directly from a published table | Exact as published |
| `text_reported` | Value stated explicitly in paper text | Exact as stated |
| `digitized_figure` | Value extracted from a published figure | ±2-5% |
| `user_provided` | User-uploaded CSV data | Depends on user |

---

## Study 1: Gibbs et al. 2011

- **Citation:** Gibbs, P.J., De Cooman, B.C., Brown, D.W., Akunets, B., Matlock, D.K., De Moor, E. "Austenite Stability Effects on Tensile Behavior of Manganese-Enriched-Austenite Transformation-Induced Plasticity Steel." *Metallurgical and Materials Transactions A*, 42A, 3691–3702, 2011.
- **DOI:** [10.1007/s11661-011-0736-2](https://doi.org/10.1007/s11661-011-0736-2)
- **Alloy:** Fe-7.1Mn-0.1C (wt%)
- **Initial condition:** Cold-rolled, 50% reduction
- **Data source:** Table III
- **Method:** Neutron diffraction (575–650°C), XRD (675°C)
- **Data points:** 5 (temperature sweep at 168 hours)

| T (°C) | Time (h) | RA (wt%) | Method | Quality |
|---|---|---|---|---|
| 575 | 168 | 23.0 | Neutron diffraction | `table` |
| 600 | 168 | 34.3 | Neutron diffraction | `table` |
| 625 | 168 | 42.8 | Neutron diffraction | `table` |
| 650 | 168 | 43.5 | Neutron diffraction | `table` |
| 675 | 168 | 1.4 | XRD | `table` |

---

## Study 2: Luo et al. 2011

- **Citation:** Luo, H., Shi, J., Wang, C., Cao, W., Sun, X., Dong, H. "Experimental and numerical analysis on formation of stable austenite during the intercritical annealing of 5Mn steel." *Acta Materialia*, 59(10), 4002–4014, 2011.
- **DOI:** [10.1016/j.actamat.2011.03.025](https://doi.org/10.1016/j.actamat.2011.03.025)
- **Alloy:** Fe-5Mn-0.2C (wt%)
- **Initial condition:** Cold-rolled martensitic
- **Data source:** Figure 2 + Results section text
- **Method:** XRD
- **Data points:** 7 (kinetic curve at 650°C)

| T (°C) | Time (h) | RA (vol%) | Quality |
|---|---|---|---|
| 650 | 0.5 | ~3 | `text_reported` |
| 650 | 1 | ~10 | `digitized_figure` |
| 650 | 4 | ~22 | `digitized_figure` |
| 650 | 12 | ~30 | `digitized_figure` |
| 650 | 24 | ~35 | `text_reported` |
| 650 | 48 | ~38 | `text_reported` |
| 650 | 144 | ~40 | `text_reported` |

---

## Study 3: PMC11053108 (2024)

- **Source:** Open-access PMC article PMC11053108
- **Alloy:** Fe-4.7Mn-0.16C-1.6Al-0.2Si-0.2Mo (wt%)
- **Initial condition:** Cold-rolled
- **Data source:** Results section text
- **Method:** XRD
- **Data points:** 7 (temperature sweep at 1 hour)

---

## Study 4: PMC11173901 (2024)

- **Source:** Open-access PMC article PMC11173901
- **Alloy:** Fe-9.4Mn-0.2C-4.3Al (wt%)
- **Data source:** Figure 3
- **Method:** XRD
- **Data points:** 3

---

## Study 5: PMC6266817 (2018)

- **Source:** Open-access PMC article PMC6266817
- **Alloy:** Fe-5Mn-0.12C-1Al-0.2Mo-0.05Nb (wt%)
- **Initial condition:** Cold-rolled (CR) and Hot-rolled (HR)
- **Data source:** Figure 4b + text
- **Method:** XRD
- **Data points:** 10 (6 CR + 4 HR)

---

## Adding Your Own Data

### Option 1: Drop CSV files

Place CSV files in `data/user_experimental/`. Required columns:

```csv
Mn,C,Al,Si,T_celsius,t_seconds,f_RA
7.0,0.10,1.5,0.5,650,3600,0.28
7.0,0.10,1.5,0.5,650,7200,0.35
```

Optional columns: `method`, `source`, `data_quality`, `notes`

### Option 2: Use WebPlotDigitizer

1. Go to https://automeris.io/WebPlotDigitizer/
2. Upload a figure from a paper
3. Set axes calibration
4. Click data points
5. Export as CSV
6. Drop into `data/user_experimental/`

### Option 3: Request from authors

Most papers include "Data available upon reasonable request." Contact:
- Colorado School of Mines (Gibbs, De Moor): ASPPRC center
- USTB Beijing (Luo, Cao): Beijing lab
- MPIE (Raabe, Ponge): Max Planck Institute

---

## Physical Constants & Equations Sources

| Equation/Value | Source | DOI/Reference |
|---|---|---|
| Ac1 correlation | Andrews (1965), JISI 203:721 | Classic, widely cited |
| Ac3 correlation | Andrews (1965), modified for medium-Mn | - |
| D₀(Mn,γ) = 1.785×10⁻⁵ m²/s | Oikawa (1983), Trans. ISIJ 23:823 | Handbook value |
| Q(Mn,γ) = 264 kJ/mol | Oikawa (1983) | Handbook value |
| D₀(C,γ) = 2.343×10⁻⁵ m²/s | Ågren (1986), Scripta Met. 20:1507 | Standard value |
| JMAK equation | Johnson-Mehl-Avrami-Kolmogorov | Textbook |
| Md30 (Angel) | Angel (1954), JISI 177:165 | Classic |
