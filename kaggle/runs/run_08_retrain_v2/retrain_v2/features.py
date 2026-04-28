from typing import Dict, Union
from pathlib import Path
import pickle
import numpy as np
from config import get_config, PhysicalConstants


def compute_diffusivity(T_kelvin: Union[float, np.ndarray], species: str = "Mn_austenite") -> Union[float, np.ndarray]:
    pc = PhysicalConstants()
    params = {"Mn_austenite": (pc.D0_Mn, pc.Q_Mn), "Mn_ferrite": (pc.D0_Mn_ferrite, pc.Q_Mn_ferrite), "C_austenite": (pc.D0_C, pc.Q_C)}
    D0, Q = params[species]
    return D0 * np.exp(-Q / (pc.R * T_kelvin))


def compute_Md30(comp: Dict[str, float]) -> float:
    pc = PhysicalConstants()
    return float(pc.angel_intercept + pc.angel_C_N * (comp.get('C', 0) + comp.get('N', 0)) + pc.angel_Si * comp.get('Si', 0) + pc.angel_Mn * comp.get('Mn', 0) + pc.angel_Cr * comp.get('Cr', 0) + pc.angel_Ni * comp.get('Ni', 0) + pc.angel_Mo * comp.get('Mo', 0))


def compute_hollomon_jaffe(T_kelvin: Union[float, np.ndarray], t_seconds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    t_hours = np.maximum(t_seconds / 3600.0, 1e-10)
    return T_kelvin * (PhysicalConstants().C_HJ + np.log10(t_hours))


def compute_JMAK(t: np.ndarray, k: float, n: float, f_eq: float = 1.0) -> np.ndarray:
    return np.clip(f_eq * (1.0 - np.exp(-k * np.power(t, n))), 0, f_eq)


def compute_JMAK_rate(t: np.ndarray, k: float, n: float, f_eq: float = 1.0) -> np.ndarray:
    t_s = np.maximum(t, 1e-6)
    return f_eq * k * n * np.power(t_s, n - 1) * np.exp(-k * np.power(t_s, n))


def compute_k_arrhenius(T_K: float, X_Mn: float, X_C: float) -> float:
    pc = PhysicalConstants()
    return float(pc.jmak_k0 * (X_Mn / 7.0) ** 0.8 * (X_C / 0.1) ** 0.3 * np.exp(-pc.jmak_Q_eff / (pc.R * T_K)))


def featurize_sample(comp: Dict[str, float], T_celsius: float, t_seconds: float, f_current: float, f_eq: float, delta_G: float, normalize: bool = True) -> np.ndarray:
    cfg = get_config()
    T_K = T_celsius + 273.15
    D_Mn = compute_diffusivity(T_K)
    P_HJ = compute_hollomon_jaffe(T_K, max(t_seconds, 1.0))
    if normalize:
        T_n = (T_K - cfg.data.T_ref) / cfg.data.T_scale
        lt = np.log(t_seconds + 1.0)
        Dl = np.log10(D_Mn + 1e-30)
        dGn = delta_G / 1000.0
        Pn = P_HJ / 20000.0
    else:
        T_n, lt, Dl, dGn, Pn = T_K, t_seconds, D_Mn, delta_G, P_HJ
    return np.array([f_current, T_n, comp.get('Mn', 0), comp.get('C', 0), comp.get('Al', 0), comp.get('Si', 0), lt, Dl, dGn, Pn], dtype=np.float32)


class FeatureScaler:
    NAMES = ['f', 'T_n', 'Mn', 'C', 'Al', 'Si', 'log_t', 'D_log', 'dG_n', 'PHJ_n']

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X):
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        self.fitted = True
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, 'rb') as f:
            obj.scaler = pickle.load(f)
        obj.fitted = True
        return obj
