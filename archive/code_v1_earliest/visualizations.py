from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from config import get_config, Config


def _setup(config=None):
    if config is None:
        config = get_config()
    v = config.visualization
    plt.rcParams.update({
        'font.family': v.font_family, 'font.size': v.font_size, 'axes.labelsize': v.font_size,
        'axes.titlesize': v.font_size + 1, 'xtick.labelsize': v.tick_size, 'ytick.labelsize': v.tick_size,
        'legend.fontsize': v.legend_size, 'figure.dpi': v.dpi, 'savefig.dpi': v.dpi, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8, 'lines.linewidth': v.line_width, 'lines.markersize': v.marker_size,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def _save(fig, name, config=None):
    if config is None:
        config = get_config()
    for fmt in config.visualization.save_formats:
        fig.savefig(config.figure_dir / f"{name}.{fmt}", dpi=config.visualization.dpi, bbox_inches='tight')
    plt.close(fig)


def plot_kinetic_curves_with_uq(predictions, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    c = config.visualization.colors
    fig, ax = plt.subplots(figsize=config.visualization.figsize_double)
    for i, (label, d) in enumerate(predictions.items()):
        color = c[i % len(c)]
        ax.fill_between(d['t_hours'], d['f_RA_lower'], d['f_RA_upper'], alpha=0.2, color=color, label=f'{label} 95% CI')
        ax.plot(d['t_hours'], d['f_RA_mean'], color=color, lw=1.5)
        if 'f_RA_true' in d and d['f_RA_true'] is not None:
            ax.scatter(d['t_hours'], d['f_RA_true'], color=color, s=20, marker='o', zorder=5, edgecolors='white', linewidths=0.5)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Austenite fraction')
    ax.set_title('Kinetics with Uncertainty')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(loc='lower right', framealpha=0.9)
    _save(fig, 'fig1_kinetic_curves_uq', config)


def plot_parity(f_true, f_pred, f_pred_std=None, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    fig, ax = plt.subplots(figsize=config.visualization.figsize_single)
    lim = [0, max(f_true.max(), f_pred.max()) * 1.1]
    ax.plot(lim, lim, '--', color='#888', lw=0.8)
    x = np.linspace(*lim, 100)
    ax.fill_between(x, x * 0.9, x * 1.1, alpha=0.1, color='gray')
    if f_pred_std is not None:
        ax.errorbar(f_true, f_pred, yerr=1.96 * f_pred_std, fmt='o', ms=3, color=config.visualization.colors[0], ecolor='lightblue', elinewidth=0.5, capsize=0, alpha=0.7)
    else:
        ax.scatter(f_true, f_pred, s=10, color=config.visualization.colors[0], alpha=0.5)
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(f_true, f_pred))
    r2 = r2_score(f_true, f_pred)
    ax.text(0.05, 0.95, f'RMSE={rmse:.4f}\nR²={r2:.4f}', transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('True RA')
    ax.set_ylabel('Predicted RA')
    ax.set_title('Parity Plot')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    _save(fig, 'fig2_parity', config)


def plot_mn_effect(sensitivity_results, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    fig, ax = plt.subplots(figsize=config.visualization.figsize_single)
    if 'Mn' in sensitivity_results:
        d = sensitivity_results['Mn']
        v = ~np.isnan(d['f_RA'])
        ax.fill_between(d['values'][v], d.get('f_RA_lower', d['f_RA'])[v], d.get('f_RA_upper', d['f_RA'])[v], alpha=0.2, color=config.visualization.colors[0])
        ax.plot(d['values'][v], d['f_RA'][v], '-o', color=config.visualization.colors[0], ms=3)
    ax.set_xlabel('Mn (wt%)')
    ax.set_ylabel('Predicted RA')
    ax.set_title('Mn Effect on Austenite Reversion')
    _save(fig, 'fig3_mn_effect', config)


def plot_temperature_effect(sensitivity_results, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    fig, ax = plt.subplots(figsize=config.visualization.figsize_single)
    if 'T' in sensitivity_results:
        d = sensitivity_results['T']
        v = ~np.isnan(d['f_RA'])
        ax.plot(d['values'][v], d['f_RA'][v], '-s', color=config.visualization.colors[1], ms=3)
        if 'f_RA_lower' in d:
            ax.fill_between(d['values'][v], d['f_RA_lower'][v], d['f_RA_upper'][v], alpha=0.2, color=config.visualization.colors[1])
    ax.set_xlabel('T (°C)')
    ax.set_ylabel('Predicted RA at t=3600s')
    ax.set_title('Temperature Dependence')
    _save(fig, 'fig4_temperature_effect', config)


def plot_phase_diagram_section(config=None):
    _setup(config)
    if config is None:
        config = get_config()
    from thermodynamics import get_Ac1_Ac3
    fig, ax = plt.subplots(figsize=config.visualization.figsize_double)
    Mn_r = np.linspace(4, 12, 50)
    Ac1s, Ac3s = zip(*[get_Ac1_Ac3({'Mn': m, 'C': 0.1, 'Al': 0.0}) for m in Mn_r])
    Ac1s, Ac3s = np.array(Ac1s), np.array(Ac3s)
    c = config.visualization.colors
    ax.fill_between(Mn_r, Ac1s, Ac3s, alpha=0.3, color=c[2], label='α+γ')
    ax.fill_between(Mn_r, Ac3s, 950, alpha=0.15, color=c[5], label='γ')
    ax.fill_between(Mn_r, 400, Ac1s, alpha=0.15, color=c[1], label='α')
    ax.plot(Mn_r, Ac1s, '-', color=c[1], lw=2, label='Ac1')
    ax.plot(Mn_r, Ac3s, '-', color=c[0], lw=2, label='Ac3')
    ax.axhspan(625, 700, alpha=0.1, color='gold', label='Typical ICA')
    ax.set_xlabel('Mn (wt%)')
    ax.set_ylabel('T (°C)')
    ax.set_title('Fe-xMn-0.1C Phase Diagram')
    ax.set_xlim(4, 12)
    ax.set_ylim(400, 950)
    ax.legend(fontsize=7)
    _save(fig, 'fig5_phase_diagram', config)


def plot_optimization_results(opt_results, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    fig, axes = plt.subplots(1, 2, figsize=config.visualization.figsize_double)
    c = config.visualization.colors
    if 'all_trials' in opt_results and opt_results['all_trials']:
        trials = opt_results['all_trials']
        losses = [t['loss'] for t in trials]
        best = np.minimum.accumulate(losses)
        axes[0].plot(range(1, len(losses)+1), losses, 'o', color=c[5], alpha=0.3, ms=3)
        axes[0].plot(range(1, len(best)+1), best, '-', color=c[0], lw=2)
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('|f_RA - target|')
        axes[0].set_title('(a) Convergence')

        Ts = [t['T_celsius'] for t in trials]
        ts = [t['t_seconds']/60 for t in trials]
        fs = [t.get('f_RA', 0) for t in trials]
        sc = axes[1].scatter(Ts, ts, c=fs, cmap='RdYlGn', s=15, alpha=0.7, edgecolors='gray', linewidths=0.3)
        plt.colorbar(sc, ax=axes[1], label='RA')
        if 'best_T' in opt_results:
            axes[1].scatter(opt_results['best_T'], opt_results['best_t']/60, marker='*', s=200, color='red', edgecolors='black', zorder=10)
        axes[1].set_xlabel('T (°C)')
        axes[1].set_ylabel('t (min)')
        axes[1].set_title('(b) Landscape')
    plt.tight_layout()
    _save(fig, 'fig6_optimization', config)


def plot_shap_summary(shap_result, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    fig, ax = plt.subplots(figsize=config.visualization.figsize_single)
    if shap_result and 'feature_importance' in shap_result:
        fi = shap_result['feature_importance']
        names = [f['feature'] for f in fi]
        vals = [f['importance'] for f in fi]
        idx = np.argsort(vals)
        ax.barh(range(len(idx)), [vals[i] for i in idx], color=config.visualization.colors[0], alpha=0.8)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([names[i] for i in idx])
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title('Feature Importance')
    _save(fig, 'fig7_shap', config)


def plot_training_history(history, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    fig, axes = plt.subplots(2, 2, figsize=config.visualization.figsize_double)
    fig.suptitle('Training', fontweight='bold')
    panels = [
        ('train_loss', 'val_loss', 'Total', axes[0, 0]),
        ('train_data', 'val_data', 'Data (Huber+MSE)', axes[0, 1]),
        ('train_physics', 'val_physics', 'Physics (JMAK)', axes[1, 0]),
        ('train_bound', 'val_bound', 'Bound (f≤f_eq)', axes[1, 1]),
    ]
    c = config.visualization.colors
    for tk, vk, title, ax in panels:
        if tk in history and history[tk]:
            eps = range(1, len(history[tk])+1)
            ax.semilogy(eps, history[tk], color=c[0], alpha=0.7, label='Train')
            if vk in history and history[vk]:
                ax.semilogy(eps, history[vk], color=c[1], alpha=0.7, label='Val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    _save(fig, 'fig8_training', config)


def plot_pareto_front(pareto_data, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    fig, ax = plt.subplots(figsize=config.visualization.figsize_single)
    c = config.visualization.colors
    if 'all_trials' in pareto_data:
        trials = pareto_data['all_trials']
        ax.scatter([t['f_RA'] for t in trials], [t.get('Md30_austenite', 0) for t in trials], c=c[5], alpha=0.3, s=10)
        if 'pareto_front' in pareto_data:
            pf = pareto_data['pareto_front']
            ax.scatter([p['f_RA'] for p in pf], [p.get('Md30_austenite', 0) for p in pf], c=c[0], s=50, marker='D', edgecolors='black', zorder=5)
    ax.axhspan(20, 60, alpha=0.1, color='green', label='Optimal Md30')
    ax.set_xlabel('RA fraction')
    ax.set_ylabel('Md30 (°C)')
    ax.set_title('Pareto Front')
    ax.legend(fontsize=7)
    _save(fig, 'fig6b_pareto', config)


def plot_nfe_evolution(history, config=None):
    _setup(config)
    if config is None:
        config = get_config()
    fig, ax = plt.subplots(figsize=config.visualization.figsize_single)
    if 'nfe' in history and history['nfe']:
        eps = range(1, len(history['nfe'])+1)
        ax.plot(eps, history['nfe'], color=config.visualization.colors[2], lw=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg NFE / batch')
        ax.set_title('ODE Solver Efficiency')
    _save(fig, 'fig9_nfe', config)


def fig0_real_data_overview(config=None):
    """Scatter plot of ALL experimental data: X=T, Y=f_RA, color=Mn, size=time."""
    _setup(config)
    if config is None:
        config = get_config()
    from real_data import load_all_experimental, EXPERIMENTAL_STUDIES
    df = load_all_experimental()
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=config.visualization.figsize_double)
    # Color by Mn content
    sc = ax.scatter(
        df['T_celsius'], df['f_RA_pct'],
        c=df['Mn'], cmap='viridis', s=np.clip(np.log1p(df['t_seconds']) * 5, 10, 80),
        alpha=0.75, edgecolors='white', linewidths=0.3, zorder=5
    )
    cb = plt.colorbar(sc, ax=ax, label='Mn (wt%)', pad=0.02)
    # Label each study
    for sid in df['study_id'].unique():
        sub = df[df['study_id'] == sid]
        cx, cy = sub['T_celsius'].mean(), sub['f_RA_pct'].mean()
        label = sid.split('_')[0]
        ax.annotate(label, (cx, cy), fontsize=5, alpha=0.6, ha='center')
    n_pts = len(df)
    n_studies = len(EXPERIMENTAL_STUDIES)
    ax.set_xlabel('Annealing Temperature (°C)')
    ax.set_ylabel('Retained Austenite (%)')
    ax.set_title(f'Experimental Database: {n_pts} points from {n_studies} studies')
    ax.set_xlim(0, 1050)
    ax.set_ylim(-2, 70)
    _save(fig, 'fig0_real_data_overview', config)


def fig_provenance_comparison(train_df=None, config=None):
    """Side-by-side showing model data distribution: real vs synthetic."""
    _setup(config)
    if config is None:
        config = get_config()
    if train_df is None:
        p = config.synthetic_dir / 'train.csv'
        if not p.exists():
            return
        train_df = pd.read_csv(p)
    if 'provenance' not in train_df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=config.visualization.figsize_double)
    c = config.visualization.colors
    # Panel A: T distribution by provenance
    for i, (prov, color) in enumerate([
        ('experimental', c[0]), ('synthetic_calibrated', c[2]), ('synthetic_exploratory', c[5])
    ]):
        sub = train_df[train_df['provenance'] == prov]
        if len(sub) > 0:
            axes[0].hist(sub['T_celsius'], bins=20, alpha=0.5, color=color,
                        label=f'{prov} ({len(sub)})', density=True)
    axes[0].set_xlabel('T (°C)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(a) Temperature Distribution')
    axes[0].legend(fontsize=6)
    # Panel B: f_RA distribution by provenance
    for i, (prov, color) in enumerate([
        ('experimental', c[0]), ('synthetic_calibrated', c[2]), ('synthetic_exploratory', c[5])
    ]):
        sub = train_df[train_df['provenance'] == prov]
        if len(sub) > 0:
            axes[1].hist(sub['f_RA'], bins=20, alpha=0.5, color=color,
                        label=f'{prov}', density=True)
    axes[1].set_xlabel('Austenite Fraction')
    axes[1].set_ylabel('Density')
    axes[1].set_title('(b) RA Distribution')
    axes[1].legend(fontsize=6)
    plt.tight_layout()
    _save(fig, 'fig_provenance_comparison', config)


def generate_all_figures(training_history=None, sensitivity_results=None, shap_result=None, opt_results=None,
                         pareto_data=None, predictions=None, f_true=None, f_pred=None, config=None):
    if config is None:
        config = get_config()
    # Always generate data overview plots
    try:
        fig0_real_data_overview(config)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f'fig0 skipped: {e}')
    try:
        fig_provenance_comparison(config=config)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f'provenance fig skipped: {e}')
    plot_phase_diagram_section(config)
    if training_history:
        plot_training_history(training_history, config)
        plot_nfe_evolution(training_history, config)
    if sensitivity_results:
        plot_mn_effect(sensitivity_results, config)
        plot_temperature_effect(sensitivity_results, config)
    if shap_result:
        plot_shap_summary(shap_result, config)
    if opt_results:
        plot_optimization_results(opt_results, config)
    if pareto_data:
        plot_pareto_front(pareto_data, config)
    if predictions:
        plot_kinetic_curves_with_uq(predictions, config)
    if f_true is not None and f_pred is not None:
        plot_parity(f_true, f_pred, config=config)
