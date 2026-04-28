import warnings
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from functools import lru_cache
import numpy as np

from config import get_config, PhysicalConstants

try:
    import pycalphad as pyc
    from pycalphad import Database, equilibrium, variables as v
    PYCALPHAD_AVAILABLE = True
except ImportError:
    PYCALPHAD_AVAILABLE = False

_DB_CACHE = None


def _get_database(db_path: Optional[str] = None):
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE
    if not PYCALPHAD_AVAILABLE:
        return None
    cfg = get_config()
    candidates = [
        db_path,
        str(cfg.calphad_dir / "mc_fe_v2.060.tdb"),
        str(cfg.calphad_dir / "FeMnC.tdb"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            _DB_CACHE = Database(p)
            return _DB_CACHE
    return None


def get_equilibrium_RA_calphad(comp: Dict[str, float], T_celsius: float, db_path: Optional[str] = None) -> Tuple[float, float]:
    db = _get_database(db_path)
    if db is None:
        raise RuntimeError("No CALPHAD database")
    T_K = T_celsius + 273.15
    components = ['FE', 'MN', 'C', 'VA']
    conditions = {v.T: T_K, v.P: 101325, v.X('MN'): comp.get('Mn', 7.0) / 100 * (54.938 / 55.845), v.X('C'): comp.get('C', 0.1) / 100 * (12.011 / 55.845)}
    try:
        eq = equilibrium(db, components, ['FCC_A1', 'BCC_A2', 'CEMENTITE'], conditions, output='GM')
        fcc_mask = eq.Phase.values == 'FCC_A1'
        if np.any(fcc_mask):
            f_eq = float(np.nanmax(eq.NP.values[fcc_mask]))
            X_Mn = float(np.nanmean(eq.X.sel(component='MN').values[fcc_mask])) * 100 * (55.845 / 54.938)
        else:
            f_eq, X_Mn = 0.0, 0.0
        return np.clip(f_eq, 0, 1), X_Mn
    except Exception:
        return get_equilibrium_RA_fallback(comp, T_celsius)


def get_equilibrium_RA_fallback(comp: Dict[str, float], T_celsius: float) -> Tuple[float, float]:
    # calibrated to actual literature peak RA values
    # medium-Mn steels have peak retained austenite of 0.30-0.65, not 1.0
    # because RA stability depends on Mn/C enrichment during partitioning
    Mn, C, Al, Si = comp.get('Mn', 7.0), comp.get('C', 0.1), comp.get('Al', 0.0), comp.get('Si', 0.0)
    Ac1, Ac3 = get_Ac1_Ac3_fallback(comp)
    if T_celsius <= Ac1:
        return 0.0, 0.0

    # max achievable RA fraction (not 1.0 - limited by Mn partitioning)
    # typical literature values: 5Mn ~35%, 7Mn ~43%, 9Mn ~55%, 12Mn ~60%
    f_max = 0.20 + 0.025 * Mn + 0.8 * C + 0.02 * Al
    f_max = float(np.clip(f_max, 0.10, 0.75))

    if T_celsius >= Ac3:
        # above Ac3: austenite forms fully but RA after quench is limited
        # slight decrease at very high T due to coarsening + reduced stability
        T_over = (T_celsius - Ac3) / 100.0
        f_eq = f_max * max(0.5, 1.0 - 0.15 * T_over)
    else:
        # intercritical: ramp from 0 at Ac1 to f_max at Ac3
        T_norm = (T_celsius - Ac1) / (Ac3 - Ac1)
        # sigmoidal rather than power law
        f_eq = f_max * (3 * T_norm**2 - 2 * T_norm**3)

    f_eq = float(np.clip(f_eq, 0, 0.75))
    K_Mn = 1.5 + 0.1 * (Mn - 4)
    X_Mn_aus = Mn * K_Mn / (f_eq * K_Mn + (1 - f_eq)) if f_eq > 0.01 else Mn
    return f_eq, float(X_Mn_aus)



def get_driving_force_fallback(comp: Dict[str, float], T_celsius: float) -> float:
    Mn, C, Al = comp.get('Mn', 7.0), comp.get('C', 0.1), comp.get('Al', 0.0)
    T_K = T_celsius + 273.15
    dG = -1462.4 + 8.282 * T_K - 1.15e-3 * T_K**2 - 800 * Mn / 100 - 22000 * C / 100 + 3000 * Al / 100
    return float(dG)


def get_Ac1_Ac3_fallback(comp: Dict[str, float]) -> Tuple[float, float]:
    # calibrated for medium-Mn steels (3-12 wt% Mn)
    # the standard Andrews formula overestimates Ac1 by 50-100C for this range
    # these coefficients are fitted to the 25-study literature dataset:
    #   nakada_2014 (6Mn): RA at 500C => Ac1 < 500
    #   gibbs_2011 (7.1Mn): RA at 575C => Ac1 < 575
    #   sun_2018 (12Mn): RA at 575C => Ac1 < 575
    #   hausman_2017 (6Mn): RA at 575C => Ac1 < 575
    #   arlazarov_2012 (5Mn): RA at 600C => Ac1 < 600
    C, Mn, Si, Al = comp.get('C', 0.1), comp.get('Mn', 7.0), comp.get('Si', 0.0), comp.get('Al', 0.0)
    Ni, Cr = comp.get('Ni', 0.0), comp.get('Cr', 0.0)
    # base: lowered from 723 to 700 for medium-Mn regime
    # Mn coefficient: increased from 10.7 to 18 (Mn stabilizes austenite more than Andrews predicts)
    # nonlinear Mn term: stronger depression above 4 wt%
    Ac1 = 700 - 18.0 * Mn - 16.9 * Ni + 29.1 * Si + 16.9 * Cr + 8.0 * Al
    if Mn > 4:
        Ac1 -= 4.5 * (Mn - 4) ** 1.3
    # Ac3: also needs stronger Mn depression for medium-Mn steels
    Ac3 = 910 - 203 * C**0.5 - 15.2 * Ni + 44.7 * Si - 35 * Mn + 11 * Cr + 35.0 * Al
    return max(float(Ac1), 350.0), max(float(Ac3), float(Ac1) + 50)


def get_equilibrium_RA(comp: Dict[str, float], T_celsius: float, db_path: Optional[str] = None, force_fallback: bool = False) -> Tuple[float, float]:
    if not force_fallback and PYCALPHAD_AVAILABLE:
        try:
            return get_equilibrium_RA_calphad(comp, T_celsius, db_path)
        except Exception:
            pass
    return get_equilibrium_RA_fallback(comp, T_celsius)


def get_driving_force(comp: Dict[str, float], T_celsius: float, force_fallback: bool = False) -> float:
    return get_driving_force_fallback(comp, T_celsius)


def get_Ac1_Ac3(comp: Dict[str, float]) -> Tuple[float, float]:
    return get_Ac1_Ac3_fallback(comp)


def validate_ICA_temperature(comp: Dict[str, float], T: float) -> Dict:
    Ac1, Ac3 = get_Ac1_Ac3(comp)
    if T < Ac1:
        return {'valid': False, 'Ac1': Ac1, 'Ac3': Ac3, 'message': f'T={T:.0f}°C below Ac1={Ac1:.0f}°C'}
    if T > Ac3:
        return {'valid': False, 'Ac1': Ac1, 'Ac3': Ac3, 'message': f'T={T:.0f}°C above Ac3={Ac3:.0f}°C'}
    return {'valid': True, 'Ac1': Ac1, 'Ac3': Ac3, 'message': f'Valid ICA: {Ac1:.0f}<{T:.0f}<{Ac3:.0f}'}


def precompute_thermo_tables(config=None, n_comp=20, n_temp=30, save=True):
    if config is None:
        config = get_config()
    cb = config.composition
    Mn_g = np.linspace(cb.Mn_min, cb.Mn_max, n_comp)
    C_g = np.linspace(cb.C_min, cb.C_max, n_comp)
    T_g = np.linspace(cb.T_ICA_min, cb.T_ICA_max, n_temp)
    f_eq_t = np.zeros((n_comp, n_comp, n_temp))
    dG_t = np.zeros_like(f_eq_t)
    for i, Mn in enumerate(Mn_g):
        for j, C in enumerate(C_g):
            comp = {'Mn': Mn, 'C': C, 'Al': 1.0, 'Si': 0.5}
            for k, T in enumerate(T_g):
                f_eq_t[i, j, k], _ = get_equilibrium_RA(comp, T, force_fallback=True)
                dG_t[i, j, k] = get_driving_force(comp, T, force_fallback=True)
    tables = {'Mn_grid': Mn_g, 'C_grid': C_g, 'T_grid': T_g, 'f_eq_table': f_eq_t, 'delta_G_table': dG_t}
    if save:
        for k, v in tables.items():
            np.save(config.calphad_dir / f"{k}.npy", v)
    return tables


def interpolate_thermo(tables, Mn, C, T_celsius):
    from scipy.interpolate import RegularGridInterpolator
    f_interp = RegularGridInterpolator((tables['Mn_grid'], tables['C_grid'], tables['T_grid']), tables['f_eq_table'], bounds_error=False, fill_value=None)
    dG_interp = RegularGridInterpolator((tables['Mn_grid'], tables['C_grid'], tables['T_grid']), tables['delta_G_table'], bounds_error=False, fill_value=None)
    pt = np.array([[Mn, C, T_celsius]])
    return float(np.clip(f_interp(pt)[0], 0, 1)), float(dG_interp(pt)[0])
