from typing import Dict, Optional
import logging
import numpy as np
import torch
from config import get_config, Config
from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA, get_driving_force
from features import compute_diffusivity, compute_Md30, compute_hollomon_jaffe
from model import PhysicsNODE

logger = logging.getLogger(__name__)


def predict_RA_for_schedule(model, comp, T_celsius, t_seconds, config=None, with_uncertainty=True):
    if config is None:
        config = get_config()
    device = config.device
    T_K = T_celsius + 273.15
    Mn, C, Al, Si = comp.get('Mn', 7.0), comp.get('C', 0.1), comp.get('Al', 0.0), comp.get('Si', 0.0)
    f_eq, X_Mn_aus = get_equilibrium_RA(comp, T_celsius, force_fallback=True)
    delta_G = get_driving_force(comp, T_celsius, force_fallback=True)
    Md30 = compute_Md30(comp)
    D_Mn = compute_diffusivity(T_K)
    P_HJ = compute_hollomon_jaffe(T_K, max(t_seconds, 1.0))

    static = torch.tensor([[(T_K - config.data.T_ref) / config.data.T_scale, Mn, C, Al, Si,
                             np.log10(D_Mn + 1e-30), delta_G / 1000.0, P_HJ / 20000.0]], dtype=torch.float32).to(device)
    f_eq_t = torch.tensor([[f_eq]], dtype=torch.float32).to(device)
    dG_t = torch.tensor([[delta_G / 1000.0]], dtype=torch.float32).to(device)
    t_span = torch.linspace(0, float(t_seconds), min(config.model.n_eval_points, 50)).to(device)

    if with_uncertainty:
        mean, lo, hi = model.predict_with_uncertainty(static, f_eq_t, dG_t, t_span, n_samples=config.model.n_mc_samples)
        return {'f_RA_mean': mean[0, -1].item(), 'f_RA_lower': lo[0, -1].item(), 'f_RA_upper': hi[0, -1].item(),
                'Md30': Md30, 'f_eq': f_eq, 'X_Mn_aus': X_Mn_aus, 'T_celsius': T_celsius, 't_seconds': t_seconds}
    else:
        model.eval()
        with torch.no_grad():
            traj = model(static, f_eq_t, dG_t, t_span)
        v = traj[0, -1].item()
        return {'f_RA_mean': v, 'f_RA_lower': v, 'f_RA_upper': v, 'Md30': Md30, 'f_eq': f_eq,
                'X_Mn_aus': X_Mn_aus, 'T_celsius': T_celsius, 't_seconds': t_seconds}


def optimize_single_objective(model, comp, target_RA=0.30, config=None, n_trials=100):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if config is None:
        config = get_config()
    oc = config.optimization
    Ac1, Ac3 = get_Ac1_Ac3(comp)
    T_min, T_max = Ac1 + oc.T_margin_from_Ac1, Ac3 - oc.T_margin_from_Ac3
    all_trials = []

    def objective(trial):
        T = trial.suggest_float('T', T_min, T_max)
        t = trial.suggest_float('t', oc.t_min_anneal, oc.t_max_anneal, log=True)
        r = predict_RA_for_schedule(model, comp, T, t, config, False)
        loss = abs(r['f_RA_mean'] - target_RA)
        all_trials.append({'T_celsius': T, 't_seconds': t, 'f_RA': r['f_RA_mean'], 'loss': loss})
        return loss

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    sorted_t = sorted(all_trials, key=lambda x: x['loss'])
    top5 = [predict_RA_for_schedule(model, comp, t['T_celsius'], t['t_seconds'], config, True) for t in sorted_t[:5]]
    return {'best_T': study.best_params['T'], 'best_t': study.best_params['t'],
            'best_RA': sorted_t[0]['f_RA'], 'target_RA': target_RA, 'composition': comp,
            'all_trials': all_trials, 'top_5': top5, 'Ac1': Ac1, 'Ac3': Ac3}


def optimize_multi_objective(model, comp, config=None, n_trials=100):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if config is None:
        config = get_config()
    oc = config.optimization
    Ac1, Ac3 = get_Ac1_Ac3(comp)
    all_trials = []

    def multi_obj(trial):
        T = trial.suggest_float('T', Ac1 + oc.T_margin_from_Ac1, Ac3 - oc.T_margin_from_Ac3)
        t = trial.suggest_float('t', oc.t_min_anneal, oc.t_max_anneal, log=True)
        r = predict_RA_for_schedule(model, comp, T, t, config, False)
        comp_aus = comp.copy()
        comp_aus['Mn'] = r.get('X_Mn_aus', comp['Mn'])
        Md30 = compute_Md30(comp_aus)
        all_trials.append({'T_celsius': T, 't_seconds': t, 'f_RA': r['f_RA_mean'], 'Md30_austenite': Md30})
        return -r['f_RA_mean'], abs(Md30 - 40.0)

    study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(multi_obj, n_trials=n_trials, show_progress_bar=True)
    pareto = [all_trials[t.number] for t in study.best_trials if t.number < len(all_trials)]
    return {'pareto_front': pareto, 'all_trials': all_trials, 'composition': comp, 'Ac1': Ac1, 'Ac3': Ac3}


def recommend_schedule(model, comp, target_RA=0.30, config=None):
    r = optimize_single_objective(model, comp, target_RA, config, 50)
    Ac1, Ac3, Md30 = r['Ac1'], r['Ac3'], compute_Md30(comp)
    lines = [
        "=" * 60, "ANNEALING SCHEDULE RECOMMENDATION", "=" * 60,
        f"Alloy: Fe-{comp.get('Mn',0):.1f}Mn-{comp.get('C',0):.2f}C-{comp.get('Al',0):.1f}Al-{comp.get('Si',0):.1f}Si",
        f"Target: {target_RA:.0%} RA | Ac1={Ac1:.0f}°C | Ac3={Ac3:.0f}°C | Md30={Md30:.0f}°C",
        f"OPTIMAL: T={r['best_T']:.0f}°C, t={r['best_t']:.0f}s ({r['best_t']/60:.1f}min)",
        "Top 5:"
    ]
    for i, t in enumerate(r['top_5']):
        lines.append(f"  #{i+1}: {t['T_celsius']:.0f}°C × {t['t_seconds']:.0f}s → RA={t['f_RA_mean']:.1%} [{t['f_RA_lower']:.1%},{t['f_RA_upper']:.1%}]")
    return "\n".join(lines)
