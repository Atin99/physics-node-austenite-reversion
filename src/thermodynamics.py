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
    Mn, C, Al, Si = comp.get('Mn', 7.0), comp.get('C', 0.1), comp.get('Al', 0.0), comp.get('Si', 0.0)
    Ac1, Ac3 = get_Ac1_Ac3_fallback(comp)
    if T_celsius <= Ac1:
        return 0.0, 0.0
    if T_celsius >= Ac3:
        return 1.0, Mn
    T_norm = (T_celsius - Ac1) / (Ac3 - Ac1)
    f_eq = T_norm ** 1.5 + 0.015 * (Mn - 7) + 0.5 * (C - 0.1) - 0.03 * Al - 0.01 * Si
    f_eq = float(np.clip(f_eq, 0, 1))
    K_Mn = 1.5 + 0.1 * (Mn - 4)
    X_Mn_aus = Mn * K_Mn / (f_eq * K_Mn + (1 - f_eq)) if f_eq > 0.01 else Mn
    return f_eq, float(X_Mn_aus)


def get_driving_force_fallback(comp: Dict[str, float], T_celsius: float) -> float:
    Mn, C, Al = comp.get('Mn', 7.0), comp.get('C', 0.1), comp.get('Al', 0.0)
    T_K = T_celsius + 273.15
    dG = -1462.4 + 8.282 * T_K - 1.15e-3 * T_K**2 - 800 * Mn / 100 - 22000 * C / 100 + 3000 * Al / 100
    return float(dG)


def get_Ac1_Ac3_fallback(comp: Dict[str, float]) -> Tuple[float, float]:
    C, Mn, Si, Al = comp.get('C', 0.1), comp.get('Mn', 7.0), comp.get('Si', 0.0), comp.get('Al', 0.0)
    Ni, Cr = comp.get('Ni', 0.0), comp.get('Cr', 0.0)
    Ac1 = 723 - 10.7 * Mn - 16.9 * Ni + 29.1 * Si + 16.9 * Cr + 6.38 * Al
    if Mn > 5:
        Ac1 -= 3.0 * (Mn - 5) ** 1.2
    Ac3 = 910 - 203 * C**0.5 - 15.2 * Ni + 44.7 * Si - 30 * Mn + 11 * Cr + 31.5 * Al
    return max(float(Ac1), 400.0), max(float(Ac3), float(Ac1) + 50)


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
