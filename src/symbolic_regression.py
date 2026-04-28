from typing import Optional, Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_symbolic_equation(model, config=None, n_samples=2000, n_time_points=20):
    try:
        from pysr import PySRRegressor
    except ImportError:
        logger.warning("PySR not installed. Install: pip install pysr")
        return None

    if config is None:
        from config import get_config
        config = get_config()

    import torch
    from data_generator import generate_synthetic_data, prepare_train_val_split
    from features import compute_diffusivity, compute_hollomon_jaffe, compute_JMAK
    from thermodynamics import get_equilibrium_RA, get_driving_force

    df = generate_synthetic_data(n_samples=n_samples, config=config, seed=0)
    device = config.device

    X_list, y_list = [], []
    for sid in df['sample_id'].unique()[:200]:
        sub = df[df['sample_id'] == sid].sort_values('t_seconds')
        Mn, C, Al, Si = sub['Mn'].iloc[0], sub['C'].iloc[0], sub['Al'].iloc[0], sub['Si'].iloc[0]
        T_c = sub['T_celsius'].iloc[0]
        T_K = T_c + 273.15
        f_eq = sub['f_eq'].iloc[0]
        D_Mn = compute_diffusivity(T_K)
        dG = get_driving_force({'Mn': Mn, 'C': C}, T_c, force_fallback=True)

        for _, row in sub.iterrows():
            t = row['t_seconds']
            static = torch.tensor([[(T_K - config.data.T_ref) / config.data.T_scale, Mn, C, Al, Si,
                                     np.log10(D_Mn + 1e-30), dG / 1000.0,
                                     compute_hollomon_jaffe(T_K, max(t, 1.0)) / 20000.0]], dtype=torch.float32).to(device)
            f_eq_t = torch.tensor([[f_eq]], dtype=torch.float32).to(device)
            dG_t = torch.tensor([[dG / 1000.0]], dtype=torch.float32).to(device)
            t_span = torch.linspace(0, float(t), 5).to(device)

            model.eval()
            with torch.no_grad():
                try:
                    pred = model(static, f_eq_t, dG_t, t_span)
                    X_list.append([Mn, C, T_K, t, f_eq, D_Mn, dG])
                    y_list.append(pred[0, -1].cpu().item())
                except Exception:
                    continue

    X = np.array(X_list)
    y = np.array(y_list)

    sr_model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["exp", "log", "sqrt", "abs"],
        populations=20,
        population_size=50,
        maxsize=30,
        parsimony=0.005,
        weight_optimize=0.001,
        turbo=True,
        bumper=True,
        progress=True,
        verbosity=1,
        extra_sympy_mappings={"pow": lambda x, y: x ** y},
        variable_names=["Mn", "C", "T", "t", "f_eq", "D_Mn", "dG"],
    )

    sr_model.fit(X, y)
    logger.info(f"\nSymbolic equations discovered:")
    logger.info(sr_model)

    best_eq = sr_model.get_best()
    logger.info(f"\nBest equation: {best_eq['equation']}")
    logger.info(f"Complexity: {best_eq['complexity']}, Loss: {best_eq['loss']:.6f}")

    return {'model': sr_model, 'best_equation': str(best_eq['equation']),
            'best_loss': float(best_eq['loss']), 'complexity': int(best_eq['complexity']),
            'X': X, 'y': y, 'equations': sr_model.equations_}


def validate_symbolic_equation(sr_result, test_conditions=None):
    if sr_result is None:
        return None
    if test_conditions is None:
        test_conditions = [
            {'Mn': 7, 'C': 0.1, 'T': 923, 't': 3600, 'f_eq': 0.35, 'D_Mn': 1e-17, 'dG': -500},
            {'Mn': 10, 'C': 0.1, 'T': 913, 't': 7200, 'f_eq': 0.45, 'D_Mn': 5e-18, 'dG': -800},
            {'Mn': 5, 'C': 0.2, 'T': 973, 't': 1800, 'f_eq': 0.25, 'D_Mn': 3e-17, 'dG': -300},
        ]
    sr_model = sr_result['model']
    results = []
    for cond in test_conditions:
        X = np.array([[cond['Mn'], cond['C'], cond['T'], cond['t'], cond['f_eq'], cond['D_Mn'], cond['dG']]])
        pred = sr_model.predict(X)[0]
        results.append({'conditions': cond, 'prediction': float(pred), 'physically_valid': 0 <= pred <= cond['f_eq']})
    return results
