from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import torch
from config import get_config, Config
from model import PhysicsNODE
from features import compute_Md30

logger = logging.getLogger(__name__)


def sensitivity_analysis(model, base_comp, base_T=650.0, base_t=3600.0, config=None, n_points=50):
    if config is None:
        config = get_config()
    cb = config.composition
    sweeps = {
        'Mn': {'values': np.linspace(cb.Mn_min, cb.Mn_max, n_points), 'key': 'Mn', 'label': 'Mn (wt%)', 'expected': 'positive'},
        'C': {'values': np.linspace(cb.C_min, cb.C_max, n_points), 'key': 'C', 'label': 'C (wt%)', 'expected': 'positive'},
        'Al': {'values': np.linspace(cb.Al_min, max(cb.Al_max, 0.1), n_points), 'key': 'Al', 'label': 'Al (wt%)', 'expected': 'negative'},
        'Si': {'values': np.linspace(cb.Si_min, max(cb.Si_max, 0.1), n_points), 'key': 'Si', 'label': 'Si (wt%)', 'expected': 'weakly_negative'},
        'T': {'values': np.linspace(cb.T_ICA_min, cb.T_ICA_max, n_points), 'key': 'T_celsius', 'label': 'T (°C)', 'expected': 'positive'},
        't': {'values': np.linspace(60, cb.t_max, n_points), 'key': 't_seconds', 'label': 't (s)', 'expected': 'positive'},
    }
    results = {}
    model.eval()
    from optimizer_annealing import predict_RA_for_schedule
    for name, sw in sweeps.items():
        f_ra, f_lo, f_hi = [], [], []
        for val in sw['values']:
            comp = base_comp.copy()
            T_c, t_s = base_T, base_t
            if sw['key'] in comp:
                comp[sw['key']] = val
            elif sw['key'] == 'T_celsius':
                T_c = val
            elif sw['key'] == 't_seconds':
                t_s = val
            try:
                r = predict_RA_for_schedule(model, comp, T_c, t_s, config, True)
                f_ra.append(r['f_RA_mean'])
                f_lo.append(r['f_RA_lower'])
                f_hi.append(r['f_RA_upper'])
            except Exception:
                f_ra.append(np.nan)
                f_lo.append(np.nan)
                f_hi.append(np.nan)
        results[name] = {'values': sw['values'], 'f_RA': np.array(f_ra), 'f_RA_lower': np.array(f_lo),
                         'f_RA_upper': np.array(f_hi), 'label': sw['label'], 'expected_trend': sw['expected']}
    return results


def validate_physics_consistency(sensitivity_results):
    expected = {'Mn': 'positive', 'C': 'positive', 'Al': 'negative', 'T': 'positive', 't': 'positive'}
    checks = {}
    for feat, direction in expected.items():
        if feat not in sensitivity_results:
            continue
        d = sensitivity_results[feat]
        valid = ~np.isnan(d['f_RA'])
        if valid.sum() < 5:
            checks[feat] = {'passed': False, 'message': f'{feat}: insufficient data'}
            continue
        corr = np.corrcoef(d['values'][valid], d['f_RA'][valid])[0, 1]
        passed = (corr > 0) if direction == 'positive' else (corr < 0)
        checks[feat] = {'passed': passed, 'correlation': float(corr), 'expected': direction,
                        'message': f"{'PASS' if passed else 'FAIL'}: {feat}→RA corr={corr:.3f} (expected {direction})"}
    return checks


def compute_shap_values(model, sample_data, feature_names=None, config=None, n_background=50):
    try:
        import shap
    except ImportError:
        return None
    if config is None:
        config = get_config()
    if feature_names is None:
        feature_names = ['T_n', 'Mn', 'C', 'Al', 'Si', 'D_log', 'dG_n', 'PHJ_n']
    device = config.device

    def predict_fn(X):
        xt = torch.tensor(X, dtype=torch.float32).to(device)
        B = xt.shape[0]
        f_eq = torch.full((B, 1), 0.35, device=device)
        dG = xt[:, 6:7]
        t_sp = torch.linspace(0, 3600, 20).to(device)
        model.eval()
        with torch.no_grad():
            try:
                return model(xt, f_eq, dG, t_sp)[:, -1].cpu().numpy()
            except Exception:
                return np.zeros(B)

    try:
        bg = sample_data[:n_background]
        explainer = shap.KernelExplainer(predict_fn, bg)
        sv = explainer.shap_values(sample_data[:100])
        imp = np.abs(sv).mean(0)
        ranked = np.argsort(imp)[::-1]
        fi = [{'feature': feature_names[i] if i < len(feature_names) else f'f{i}', 'importance': float(imp[i])} for i in ranked]
        return {'shap_values': sv, 'feature_importance': fi, 'feature_names': feature_names, 'base_value': explainer.expected_value}
    except Exception:
        return None


def compute_partial_dependence(model, base_comp, feature_name, feature_range, base_T=650.0, base_t=3600.0, n_points=30, config=None):
    if config is None:
        config = get_config()
    from optimizer_annealing import predict_RA_for_schedule
    vals = np.linspace(*feature_range, n_points)
    f_ra = []
    for v in vals:
        comp = base_comp.copy()
        T_c, t_s = base_T, base_t
        if feature_name in ['Mn', 'C', 'Al', 'Si']:
            comp[feature_name] = v
        elif feature_name == 'T':
            T_c = v
        elif feature_name == 't':
            t_s = v
        try:
            r = predict_RA_for_schedule(model, comp, T_c, t_s, config, False)
            f_ra.append(r['f_RA_mean'])
        except Exception:
            f_ra.append(np.nan)
    return {'values': vals, 'f_RA': np.array(f_ra), 'feature_name': feature_name}


def run_explainability_suite(model, sample_data=None, config=None):
    if config is None:
        config = get_config()
    base_comp = {'Mn': 7.0, 'C': 0.1, 'Al': 1.5, 'Si': 0.5}
    results = {}
    logger.info("Running sensitivity analysis...")
    results['sensitivity'] = sensitivity_analysis(model, base_comp, config=config, n_points=20)
    results['consistency'] = validate_physics_consistency(results['sensitivity'])
    for c in results['consistency'].values():
        logger.info(f"  {c['message']}")
    if sample_data is not None:
        logger.info("Running SHAP analysis...")
        results['shap'] = compute_shap_values(model, sample_data, config=config)
    cb = config.composition
    results['partial_dependence'] = {
        f: compute_partial_dependence(model, base_comp, f, r, config=config, n_points=20)
        for f, r in [('Mn', (cb.Mn_min, cb.Mn_max)), ('T', (cb.T_ICA_min, cb.T_ICA_max)), ('t', (60, cb.t_max))]
    }
    return results
