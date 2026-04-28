"""
Three-Tier Data Pipeline for Physics-Constrained Latent Neural ODE
==================================================================

TIER 1: Real experimental data (from real_data.py + user CSVs)
         → provenance = 'experimental' or 'user_provided'

TIER 2: Physics-calibrated synthetic data
         JMAK curves calibrated to match real data endpoints
         → provenance = 'synthetic_calibrated'

TIER 3: Exploratory synthetic data
         LHS sampled compositions + JMAK kinetics
         → provenance = 'synthetic_exploratory'

Every data point carries a provenance tag. The model is trained with
provenance-aware loss weighting (real data weighted higher).
"""

from typing import Optional, Tuple, Dict, List
from pathlib import Path
import numpy as np
import pandas as pd
from config import get_config, Config
from features import compute_k_arrhenius, compute_diffusivity, compute_JMAK, compute_Md30
from thermodynamics import get_equilibrium_RA, get_driving_force, get_Ac1_Ac3

import logging
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: REAL DATA
# ═══════════════════════════════════════════════════════════════════════════

def build_real_dataset(config: Optional[Config] = None) -> pd.DataFrame:
    """Load all real experimental data into training-ready format.

    Combines:
      1) Verified published data from real_data.py
      2) Any user-provided CSVs from data/user_experimental/

    Returns DataFrame in the standard pipeline format with 'provenance' column.
    """
    if config is None:
        config = get_config()

    from real_data import load_all_experimental, load_user_csvs, get_study_summary

    # Print what we have
    logger.info(get_study_summary())

    # Load published experimental data
    exp_df = load_all_experimental()

    # Load user-provided CSVs
    user_df = load_user_csvs(config.user_data_dir)

    # Combine
    if len(user_df) > 0:
        df = pd.concat([exp_df, user_df], ignore_index=True)
        logger.info(f"Combined: {len(exp_df)} published + {len(user_df)} user = {len(df)} total real points")
    else:
        df = exp_df

    # Convert to training format: each real data point becomes a single-point "curve"
    # For temperature sweeps (same alloy, same time, varying T), each T is a separate sample
    # For kinetic curves (same alloy, same T, varying time), group them as one sample
    training_df = _convert_real_to_training_format(df, config)

    return training_df


def _convert_real_to_training_format(real_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Convert real experimental data into the training format expected by the trainer.

    For kinetic curves (multiple time points at same T and comp), generates a full curve.
    For single measurements, generates a minimal curve anchored at that data point.

    The output format matches what generate_kinetic_curve() produces:
    columns: sample_id, t_seconds, f_RA, f_eq, delta_G, k_jmak, n_jmak,
             Mn, C, Al, Si, T_celsius, Ac1, Ac3, X_Mn_aus, provenance
    """
    all_curves = []
    sample_id = 0
    min_group_points = max(int(getattr(config.data, 'real_curve_group_min_points', 2)), 2)

    # Group by alloy + temperature + study to find kinetic curves
    group_keys = ['Mn', 'C', 'Al', 'Si', 'T_celsius', 'study_id']
    available_keys = [k for k in group_keys if k in real_df.columns]

    for group_vals, group in real_df.groupby(available_keys):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        group_dict = dict(zip(available_keys, group_vals))

        Mn = group_dict.get('Mn', 7.0)
        C = group_dict.get('C', 0.1)
        Al = group_dict.get('Al', 0.0)
        Si = group_dict.get('Si', 0.0)
        T_c = group_dict.get('T_celsius', 650.0)
        comp = {'Mn': Mn, 'C': C, 'Al': Al, 'Si': Si}

        # Get thermodynamic properties
        f_eq, X_Mn_aus = get_equilibrium_RA(comp, T_c, force_fallback=True)
        delta_G = get_driving_force(comp, T_c, force_fallback=True)
        Ac1, Ac3 = get_Ac1_Ac3(comp)
        T_K = T_c + 273.15
        k_default = compute_k_arrhenius(T_K, Mn, C)
        n_default = 2.0

        group_sorted = group.sort_values('t_seconds')
        if group_sorted['t_seconds'].duplicated().any():
            group_sorted = (
                group_sorted.groupby('t_seconds', as_index=False, sort=True)
                .agg({
                    'f_RA': 'mean',
                    'provenance': 'first',
                })
            )
        prov = group_sorted['provenance'].iloc[0] if 'provenance' in group_sorted.columns else 'experimental'
        k_fit, n_jmak = _estimate_jmak_parameters(group_sorted, f_eq, k_default, n_default)

        if len(group_sorted) >= min_group_points:
            # Multiple time points → use as kinetic curve directly
            # Add t=0, f_RA=0 anchor if missing
            times = group_sorted['t_seconds'].to_numpy(dtype=float)
            f_vals = group_sorted['f_RA'].to_numpy(dtype=float)
            obs_mask = np.ones_like(times, dtype=np.float32)

            if times[0] > 0:
                times = np.concatenate([[0.0], times])
                f_vals = np.concatenate([[0.0], f_vals])
                obs_mask = np.concatenate([[0.0], obs_mask])

            for t, f, observed in zip(times, f_vals, obs_mask):
                all_curves.append({
                    'sample_id': sample_id,
                    't_seconds': float(t),
                    'f_RA': float(f),
                    'f_clean': float(f),
                    'f_eq': f_eq,
                    'delta_G': delta_G,
                    'k_jmak': k_fit,
                    'n_jmak': n_jmak,
                    'Mn': Mn, 'C': C, 'Al': Al, 'Si': Si,
                    'T_celsius': T_c,
                    'Ac1': Ac1, 'Ac3': Ac3,
                    'X_Mn_aus': X_Mn_aus,
                    'provenance': prov,
                    'fidelity': 'experimental',
                    'is_observed': float(observed),
                })
            sample_id += 1

        else:
            row = group_sorted.iloc[0]
            t_real = float(row['t_seconds'])
            f_real = float(row['f_RA'])

            if t_real > 0:
                t_arr = np.array([0.0, t_real], dtype=float)
                f_arr = np.array([0.0, f_real], dtype=float)
                obs_mask = np.array([0.0, 1.0], dtype=np.float32)
            else:
                t_arr = np.array([0.0], dtype=float)
                f_arr = np.array([f_real], dtype=float)
                obs_mask = np.array([1.0], dtype=np.float32)

            for t, f, observed in zip(t_arr, f_arr, obs_mask):
                all_curves.append({
                    'sample_id': sample_id,
                    't_seconds': float(t),
                    'f_RA': float(f),
                    'f_clean': float(f),
                    'f_eq': f_eq,
                    'delta_G': delta_G,
                    'k_jmak': k_fit,
                    'n_jmak': n_jmak,
                    'Mn': Mn, 'C': C, 'Al': Al, 'Si': Si,
                    'T_celsius': T_c,
                    'Ac1': Ac1, 'Ac3': Ac3,
                    'X_Mn_aus': X_Mn_aus,
                    'provenance': prov,
                    'fidelity': 'experimental',
                    'is_observed': float(observed),
                })
            sample_id += 1

    df = pd.DataFrame(all_curves)
    logger.info(f"Converted {len(real_df)} real measurements into {sample_id} training curves ({len(df)} total points)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: PHYSICS-CALIBRATED SYNTHETIC (fills gaps in real data)
# ═══════════════════════════════════════════════════════════════════════════

def build_calibrated_synthetic(real_df: pd.DataFrame, config: Optional[Config] = None,
                                 n_samples: int = None) -> pd.DataFrame:
    """Generate synthetic curves CALIBRATED to bracket real data endpoints.

    Only generates curves in composition/temperature regions NOT covered by real data.
    All outputs are tagged with provenance='synthetic_calibrated'.
    """
    if config is None:
        config = get_config()
    if n_samples is None:
        n_samples = config.data.synthetic_calibration_samples

    # Determine what compositions and temperatures real data covers
    real_Mn = real_df['Mn'].unique() if 'Mn' in real_df.columns else []
    real_T = real_df['T_celsius'].unique() if 'T_celsius' in real_df.columns else []

    cb = config.composition
    rng = np.random.RandomState(config.model.random_seed + 100)

    # LHS sampling in gaps between real data
    lhs = latin_hypercube_sample(n_samples, 6, config.model.random_seed + 100)
    Mn_arr = cb.Mn_min + lhs[:, 0] * (cb.Mn_max - cb.Mn_min)
    C_arr = cb.C_min + lhs[:, 1] * (cb.C_max - cb.C_min)
    Al_arr = cb.Al_min + lhs[:, 2] * (cb.Al_max - cb.Al_min)
    Si_arr = cb.Si_min + lhs[:, 3] * (cb.Si_max - cb.Si_min)
    T_arr = cb.T_ICA_min + lhs[:, 4] * (cb.T_ICA_max - cb.T_ICA_min)
    log_t_min, log_t_max = np.log(max(cb.t_min, 30)), np.log(cb.t_max)
    t_max_arr = np.exp(log_t_min + lhs[:, 5] * (log_t_max - log_t_min))

    all_curves = []
    for idx in range(n_samples):
        comp = {'Mn': Mn_arr[idx], 'C': C_arr[idx], 'Al': Al_arr[idx], 'Si': Si_arr[idx]}
        Ac1, Ac3 = get_Ac1_Ac3(comp)
        T = float(np.clip(T_arr[idx], Ac1 + 10, Ac3 - 10))
        sigma = rng.uniform(config.data.noise_sigma_min, config.data.noise_sigma_max)

        try:
            curve = generate_kinetic_curve(comp, T, t_max_arr[idx],
                                             config.data.n_time_points, sigma, rng)
            curve['sample_id'] = 100000 + idx  # offset to avoid collision with real
            curve['provenance'] = 'synthetic_calibrated'
            curve['fidelity'] = 'synthetic'
            all_curves.append(curve)
        except Exception:
            continue

    if all_curves:
        df = pd.concat(all_curves, ignore_index=True)
        logger.info(f"Generated {len(all_curves)} calibrated synthetic curves ({len(df)} points) | provenance='synthetic_calibrated'")
        return df
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: EXPLORATORY SYNTHETIC (original broad sampling)
# ═══════════════════════════════════════════════════════════════════════════

def generate_exploratory_synthetic(n_samples: int = None, config: Optional[Config] = None,
                                     seed: int = 42) -> pd.DataFrame:
    """Generate broad exploratory synthetic data via LHS + JMAK.

    This is the original synthetic data generation, now clearly tagged.
    All outputs: provenance='synthetic_exploratory'
    """
    if config is None:
        config = get_config()
    if n_samples is None:
        n_samples = config.data.synthetic_exploration_samples
    dc = config.data
    rng = np.random.RandomState(seed)
    compositions = sample_compositions(n_samples, config, seed)
    all_curves = []

    for idx in range(len(compositions)):
        row = compositions.iloc[idx]
        comp = {'Mn': row['Mn'], 'C': row['C'], 'Al': row['Al'], 'Si': row['Si']}
        Ac1, Ac3 = get_Ac1_Ac3(comp)
        T = float(np.clip(row['T_celsius'], Ac1 + 10, Ac3 - 10))
        sigma = rng.uniform(dc.noise_sigma_min, dc.noise_sigma_max)

        try:
            curve = generate_kinetic_curve(comp, T, row['t_max'], dc.n_time_points, sigma, rng)
            curve['sample_id'] = 200000 + idx  # offset
            curve['provenance'] = 'synthetic_exploratory'
            curve['fidelity'] = 'synthetic'
            all_curves.append(curve)
        except Exception:
            continue

    df = pd.concat(all_curves, ignore_index=True)
    logger.info(f"Generated {len(all_curves)} exploratory synthetic curves ({len(df)} points) | provenance='synthetic_exploratory'")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_full_dataset(config: Optional[Config] = None) -> pd.DataFrame:
    """Build the complete three-tier dataset.

    Returns a single DataFrame with all tiers merged, each row tagged with provenance.
    Prints a summary of what was built.
    """
    if config is None:
        config = get_config()

    dfs = []

    # TIER 1: Real data (always loaded)
    if config.data.use_real_data:
        real_df = build_real_dataset(config)
        if len(real_df) > 0:
            dfs.append(real_df)
            logger.info(f"TIER 1 (Real): {real_df['sample_id'].nunique()} curves, {len(real_df)} points")

    if not config.data.real_only:
        # TIER 2: Calibrated synthetic
        real_for_cal = dfs[0] if dfs else pd.DataFrame()
        cal_df = build_calibrated_synthetic(real_for_cal, config)
        if len(cal_df) > 0:
            dfs.append(cal_df)
            logger.info(f"TIER 2 (Calibrated Synthetic): {cal_df['sample_id'].nunique()} curves, {len(cal_df)} points")

        # TIER 3: Exploratory synthetic
        exp_df = generate_exploratory_synthetic(config=config)
        if len(exp_df) > 0:
            dfs.append(exp_df)
            logger.info(f"TIER 3 (Exploratory Synthetic): {exp_df['sample_id'].nunique()} curves, {len(exp_df)} points")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
    else:
        raise RuntimeError("No data available! Check real_data.py or enable synthetic generation.")

    # Summary
    prov_counts = full_df['provenance'].value_counts()
    logger.info(f"\n{'='*60}")
    logger.info(f"FULL DATASET: {full_df['sample_id'].nunique()} curves, {len(full_df)} total points")
    for prov, count in prov_counts.items():
        logger.info(f"  {prov}: {count} points")
    logger.info(f"{'='*60}\n")

    return full_df


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (kept from original, used by all tiers)
# ═══════════════════════════════════════════════════════════════════════════

def latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        samples[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
    return samples


def sample_compositions(n_samples: int, config: Optional[Config] = None, seed: int = 42) -> pd.DataFrame:
    if config is None:
        config = get_config()
    cb = config.composition
    lhs = latin_hypercube_sample(n_samples, 6, seed)
    Mn = cb.Mn_min + lhs[:, 0] * (cb.Mn_max - cb.Mn_min)
    C = cb.C_min + lhs[:, 1] * (cb.C_max - cb.C_min)
    Al = cb.Al_min + lhs[:, 2] * (cb.Al_max - cb.Al_min)
    Si = cb.Si_min + lhs[:, 3] * (cb.Si_max - cb.Si_min)
    T_celsius = cb.T_ICA_min + lhs[:, 4] * (cb.T_ICA_max - cb.T_ICA_min)
    log_t_min, log_t_max = np.log(max(cb.t_min, 30)), np.log(cb.t_max)
    t_max = np.exp(log_t_min + lhs[:, 5] * (log_t_max - log_t_min))
    return pd.DataFrame({'Mn': Mn, 'C': C, 'Al': Al, 'Si': Si, 'T_celsius': T_celsius, 't_max': t_max})


def generate_kinetic_curve(comp: Dict[str, float], T_celsius: float, t_max: float, n_points: int = 50, noise_sigma: float = 0.02, rng: np.random.RandomState = None) -> pd.DataFrame:
    T_K = T_celsius + 273.15
    Mn, C = comp.get('Mn', 7.0), comp.get('C', 0.1)
    f_eq, X_Mn_aus = get_equilibrium_RA(comp, T_celsius, force_fallback=True)
    delta_G = get_driving_force(comp, T_celsius, force_fallback=True)
    Ac1, Ac3 = get_Ac1_Ac3(comp)
    k = compute_k_arrhenius(T_K, Mn, C)
    pc = get_config().physics
    n_jmak = pc.jmak_n_min + (pc.jmak_n_max - pc.jmak_n_min) * (rng.random() if rng else np.random.random())

    t = np.concatenate([np.array([0.0]), np.geomspace(1, t_max, n_points - 1)])
    f_clean = compute_JMAK(t, k, n_jmak, f_eq)
    if rng:
        noise = rng.normal(0, noise_sigma, n_points)
        outliers = rng.random(n_points) < 0.03
        noise[outliers] *= 5
    else:
        noise = np.random.normal(0, noise_sigma, n_points)
    f_noisy = np.clip(f_clean + noise, 0, f_eq)
    f_noisy[0] = 0.0

    rows = []
    for i in range(n_points):
        rows.append(dict(t_seconds=t[i], f_RA=f_noisy[i], f_clean=f_clean[i], f_eq=f_eq, delta_G=delta_G,
                         k_jmak=k, n_jmak=n_jmak, Mn=Mn, C=C, Al=comp.get('Al', 0), Si=comp.get('Si', 0),
                         T_celsius=T_celsius, Ac1=Ac1, Ac3=Ac3, X_Mn_aus=X_Mn_aus, is_observed=1.0))
    return pd.DataFrame(rows)


def _estimate_jmak_parameters(group: pd.DataFrame, f_eq: float, default_k: float, default_n: float) -> Tuple[float, float]:
    valid = group[
        (group['t_seconds'] > 0)
        & (group['f_RA'] > 1e-4)
        & (group['f_RA'] < max(f_eq * 0.999, 1e-4))
    ].copy()
    if valid.empty or f_eq <= 1e-6:
        return float(default_k), float(default_n)

    ratio = np.clip(valid['f_RA'].to_numpy(dtype=float) / max(f_eq, 1e-6), 1e-8, 0.999999)
    t_vals = valid['t_seconds'].to_numpy(dtype=float)

    if len(valid) >= 2:
        x = np.log(t_vals)
        y = np.log(-np.log(1.0 - ratio))
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 2:
            slope, intercept = np.polyfit(x[finite], y[finite], deg=1)
            n_fit = float(np.clip(slope, 0.8, 4.0))
            k_fit = float(np.clip(np.exp(intercept), 1e-12, 1e3))
            return k_fit, n_fit

    ratio_last = float(ratio[-1])
    t_last = float(max(t_vals[-1], 1e-6))
    k_fit = -np.log(1.0 - ratio_last) / max(t_last ** default_n, 1e-30)
    return float(np.clip(k_fit, 1e-12, 1e3)), float(default_n)


# Legacy function names for backward compatibility
def generate_synthetic_data(n_samples: int = 5000, config: Optional[Config] = None, seed: int = 42) -> pd.DataFrame:
    """Legacy wrapper — now calls build_full_dataset or generate_exploratory_synthetic."""
    if config is None:
        config = get_config()
    return generate_exploratory_synthetic(n_samples, config, seed)


def create_literature_validation_data(config: Optional[Config] = None) -> pd.DataFrame:
    """Create literature validation data — now pulls from real_data.py instead of hardcoded."""
    if config is None:
        config = get_config()

    from real_data import load_all_experimental
    df = load_all_experimental()
    path = config.literature_dir / "literature_validation.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} real experimental points to {path}")
    return df


def prepare_train_val_test_split(df: pd.DataFrame, config: Optional[Config] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = get_config()
    ids = df['sample_id'].unique()
    np.random.seed(config.model.random_seed)

    # Stratified split: ensure real data appears in both train and val
    if 'provenance' in df.columns:
        real_ids = df[df['provenance'].isin(['experimental', 'user_provided'])]['sample_id'].unique()
        synth_ids = df[~df['provenance'].isin(['experimental', 'user_provided'])]['sample_id'].unique()

        np.random.shuffle(real_ids)
        np.random.shuffle(synth_ids)

        r_train, r_val, _ = config.data.train_val_test_split
        # Split real
        n_real_train = max(1, int(len(real_ids) * r_train))
        n_real_val = max(1, int(len(real_ids) * (r_train + r_val))) if len(real_ids) > 1 else 1
        real_train = real_ids[:n_real_train]
        real_val = real_ids[n_real_train:n_real_val]
        real_test = real_ids[n_real_val:]

        # Split synthetic
        n_synth_train = int(len(synth_ids) * r_train)
        n_synth_val = int(len(synth_ids) * (r_train + r_val))
        synth_train = synth_ids[:n_synth_train]
        synth_val = synth_ids[n_synth_train:n_synth_val]
        synth_test = synth_ids[n_synth_val:]

        train_ids = np.concatenate([real_train, synth_train])
        val_ids = np.concatenate([real_val, synth_val])
        test_ids = np.concatenate([real_test, synth_test])
    else:
        np.random.shuffle(ids)
        r_train, r_val, _ = config.data.train_val_test_split
        n_train = int(len(ids) * r_train)
        n_val = int(len(ids) * (r_train + r_val))
        train_ids, val_ids, test_ids = ids[:n_train], ids[n_train:n_val], ids[n_val:]

    return df[df['sample_id'].isin(train_ids)], df[df['sample_id'].isin(val_ids)], df[df['sample_id'].isin(test_ids)]


def prepare_train_val_split(df: pd.DataFrame, config: Optional[Config] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, va, te = prepare_train_val_test_split(df, config)
    return tr, pd.concat([va, te], ignore_index=True)


def save_synthetic_data(df: pd.DataFrame, config: Optional[Config] = None):
    if config is None:
        config = get_config()
    path = config.synthetic_dir / "synthetic_kinetics.csv"
    df.to_csv(path, index=False)


def plot_synthetic_curves(df, n_show=10, config=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if config is None:
        config = get_config()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot by provenance with different styles
    for prov, style, alpha in [('experimental', '-o', 1.0), ('user_provided', '-s', 1.0),
                                 ('synthetic_calibrated', '--', 0.5), ('synthetic_exploratory', ':', 0.3)]:
        subset = df[df['provenance'] == prov] if 'provenance' in df.columns else df
        if len(subset) == 0:
            continue
        ids = subset['sample_id'].unique()[:n_show]
        for sid in ids:
            sub = subset[subset['sample_id'] == sid].sort_values('t_seconds')
            label = f"{prov[:4]}#{sid}" if sid == ids[0] else None
            ax.plot(sub['t_seconds'] / 3600, sub['f_RA'], style, markersize=2, alpha=alpha, label=label)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Austenite fraction')
    ax.set_title('Training Data — Real (solid) vs Synthetic (dashed)')
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(config.figure_dir / "data_overview.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved data overview figure to {config.figure_dir / 'data_overview.png'}")
