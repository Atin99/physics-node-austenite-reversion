"""
Regenerate figures using the retrained checkpoint (stage2_fixed_best.pt).
Fixes: fig2_parity, fig3_mn_effect, fig4_temperature_effect, fig7_shap, fig8_training, fig9_nfe
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from config import get_config
from model import PhysicsNODE
from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA, get_driving_force
from features import compute_diffusivity, compute_hollomon_jaffe

cfg = get_config()
model = PhysicsNODE(cfg.model)

# load checkpoint
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'stage2_fixed_best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
model.load_state_dict(state, strict=False)
model.eval()
print("Loaded:", os.path.basename(ckpt_path))

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# ── shared prediction function ──
def predict_one(comp, T_c, t_sec):
    T_K = T_c + 273.15
    f_eq, _ = get_equilibrium_RA(comp, T_c, force_fallback=True)
    dG = get_driving_force(comp, T_c, force_fallback=True)
    D = compute_diffusivity(T_K)
    P = compute_hollomon_jaffe(T_K, max(t_sec/2, 1.0))
    static = torch.tensor([[(T_K - cfg.data.T_ref)/cfg.data.T_scale, comp['Mn'], comp['C'],
                             comp.get('Al',0), comp.get('Si',0), np.log10(D+1e-30),
                             dG/1000.0, P/20000.0]], dtype=torch.float32)
    f_eq_t = torch.tensor([[f_eq]], dtype=torch.float32)
    dG_t = torch.tensor([[dG/1000.0]], dtype=torch.float32)
    t_span = torch.linspace(0, float(t_sec), 50)
    with torch.no_grad():
        pred = model(static, f_eq_t, dG_t, t_span)
    curve = pred[0].numpy()
    return curve[-1], f_eq, curve

# ── style ──
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
})
colors = ['#2176AE', '#D4722C', '#57A773', '#B5495B', '#9B72AA', '#E8AE68']


# ====================================================================
# FIG 3: Mn effect (FIXED — uses new model)
# ====================================================================
print("Generating fig3_mn_effect...")
fig, ax = plt.subplots(figsize=(6.5, 5))
mn_vals = np.arange(4.0, 12.5, 0.5)
ra_vals = []
for mn in mn_vals:
    comp = {'Mn': mn, 'C': 0.10, 'Al': 0, 'Si': 0}
    val, feq, _ = predict_one(comp, 650, 3600)
    ra_vals.append(val)
ra_vals = np.array(ra_vals)
ax.plot(mn_vals, ra_vals, '-o', color=colors[0], ms=5, lw=2)
ax.set_xlabel('Mn (wt%)')
ax.set_ylabel('Predicted RA fraction (1h, 650C)')
ax.set_title('Mn Effect on Austenite Reversion')
ax.set_xlim(3.5, 13)
ax.set_ylim(-0.02, max(0.5, ra_vals.max()*1.1))
for fmt in ['png', 'pdf']:
    fig.savefig(os.path.join(fig_dir, f'fig3_mn_effect.{fmt}'))
plt.close(fig)
print(f"  Mn range: {ra_vals.min():.3f} - {ra_vals.max():.3f}")


# ====================================================================
# FIG 4: Temperature effect (FIXED — uses new model)
# ====================================================================
print("Generating fig4_temperature_effect...")
fig, ax = plt.subplots(figsize=(6.5, 5))
T_vals = np.arange(550, 760, 10)
ra_T = []
for T in T_vals:
    comp = {'Mn': 7.0, 'C': 0.10, 'Al': 0, 'Si': 0}
    val, feq, _ = predict_one(comp, T, 3600)
    ra_T.append(val)
ra_T = np.array(ra_T)
ax.plot(T_vals, ra_T, '-s', color=colors[1], ms=4, lw=2)
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Predicted RA fraction (1h, Fe-7Mn-0.1C)')
ax.set_title('Temperature Dependence of Austenite Reversion')
ax.set_xlim(540, 760)
ax.set_ylim(-0.02, max(0.5, ra_T.max()*1.1))
for fmt in ['png', 'pdf']:
    fig.savefig(os.path.join(fig_dir, f'fig4_temperature_effect.{fmt}'))
plt.close(fig)
peak_idx = np.argmax(ra_T)
print(f"  Peak: {ra_T[peak_idx]:.3f} at {T_vals[peak_idx]}C")


# ====================================================================
# FIG 7: SHAP / Feature importance (FIXED — perturbation-based)
# ====================================================================
print("Generating fig7_shap (perturbation-based feature importance)...")

# Use perturbation-based importance since SHAP was empty
base_comp = {'Mn': 7.0, 'C': 0.10, 'Al': 0.0, 'Si': 0.0}
base_T = 650
base_t = 3600
base_pred, _, _ = predict_one(base_comp, base_T, base_t)

features = {
    'T': ('Temperature', None),
    'Mn': ('Mn content', None),
    'C': ('C content', None),
    'Al': ('Al content', None),
    'Si': ('Si content', None),
}

importance = {}

# Temperature perturbation
preds_T = []
for dT in [-50, -25, -10, 10, 25, 50]:
    val, _, _ = predict_one(base_comp, base_T + dT, base_t)
    preds_T.append(abs(val - base_pred))
importance['Temperature'] = np.mean(preds_T)

# Composition perturbations
for elem, delta_range in [('Mn', [1, 2, 3]), ('C', [0.05, 0.10, 0.15]), ('Al', [0.5, 1.0, 2.0]), ('Si', [0.3, 0.5, 1.0])]:
    preds_elem = []
    for delta in delta_range:
        comp_up = dict(base_comp)
        comp_up[elem] = base_comp[elem] + delta
        val_up, _, _ = predict_one(comp_up, base_T, base_t)
        preds_elem.append(abs(val_up - base_pred))
        if base_comp[elem] - delta > 0:
            comp_dn = dict(base_comp)
            comp_dn[elem] = base_comp[elem] - delta
            val_dn, _, _ = predict_one(comp_dn, base_T, base_t)
            preds_elem.append(abs(val_dn - base_pred))
    importance[elem] = np.mean(preds_elem)

# Time perturbation
preds_t = []
for t_mult in [0.1, 0.25, 0.5, 2.0, 4.0, 24.0]:
    val, _, _ = predict_one(base_comp, base_T, base_t * t_mult)
    preds_t.append(abs(val - base_pred))
importance['Time'] = np.mean(preds_t)

# Sort and plot
names = sorted(importance.keys(), key=lambda x: importance[x])
vals = [importance[n] for n in names]

fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.barh(range(len(names)), vals, color=colors[0], alpha=0.85, edgecolor='white')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Mean |delta RA| from perturbation')
ax.set_title('Feature Importance (Perturbation Analysis)')
for i, v in enumerate(vals):
    ax.text(v + max(vals)*0.02, i, f'{v:.3f}', va='center', fontsize=9)
for fmt in ['png', 'pdf']:
    fig.savefig(os.path.join(fig_dir, f'fig7_shap.{fmt}'))
plt.close(fig)
print(f"  Top feature: {names[-1]} ({vals[-1]:.4f})")


# ====================================================================
# FIG 2: Parity plot (FIXED — uses new model on real data)
# ====================================================================
print("Generating fig2_parity...")
from real_data import load_all_experimental
df = load_all_experimental()
if len(df) > 0:
    f_true = []
    f_pred = []
    for _, row in df.iterrows():
        comp = {'Mn': row['Mn'], 'C': row['C'], 'Al': row.get('Al', 0), 'Si': row.get('Si', 0)}
        T_c = row['T_celsius']
        t_sec = row['t_seconds']
        f_ra = row['f_RA']
        try:
            val, feq, _ = predict_one(comp, T_c, t_sec)
            f_true.append(f_ra)
            f_pred.append(val)
        except Exception:
            continue

    f_true = np.array(f_true)
    f_pred = np.array(f_pred)

    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(f_true, f_pred))
    r2 = r2_score(f_true, f_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    upper = max(f_true.max(), f_pred.max()) * 1.1
    lim = [0, max(upper, 0.7)]
    ax.plot(lim, lim, '--', color='#888', lw=1)
    x = np.linspace(0, lim[1], 100)
    ax.fill_between(x, x*0.85, x*1.15, alpha=0.08, color='gray', label='+/- 15%')
    ax.scatter(f_true, f_pred, s=12, color=colors[0], alpha=0.6, edgecolors='white', linewidths=0.3)
    ax.text(0.05, 0.95, f'RMSE={rmse:.4f}\nR\u00b2={r2:.4f}\nn={len(f_true)}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_xlabel('Measured RA fraction')
    ax.set_ylabel('Predicted RA fraction')
    ax.set_title('Parity Plot (all experimental data)')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(fig_dir, f'fig2_parity.{fmt}'))
    plt.close(fig)
    print(f"  RMSE={rmse:.4f}, R2={r2:.4f}, n={len(f_true)}")
else:
    print("  WARNING: no experimental data loaded")


# ====================================================================
# FIG 8: Training history (from stage1 CSV + retrain CSV)
# ====================================================================
print("Generating fig8_training...")
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Try to load stage1 history (the long run)
stage1_csv = os.path.join(model_dir, 'stage1_history.csv')
retrain_csv = os.path.join(model_dir, 'stage2_fixed_history.csv')

has_stage1 = os.path.exists(stage1_csv)
has_retrain = os.path.exists(retrain_csv)

if has_stage1:
    s1 = pd.read_csv(stage1_csv)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Training History', fontweight='bold', fontsize=13)

    # Total loss
    if 'train_loss' in s1.columns:
        axes[0,0].semilogy(s1['epoch'], s1['train_loss'], color=colors[0], alpha=0.7, label='Train', lw=1.5)
        if 'val_loss' in s1.columns:
            axes[0,0].semilogy(s1['epoch'], s1['val_loss'], color=colors[1], alpha=0.7, label='Val', lw=1.5)
        axes[0,0].set_title('Total Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend(fontsize=8)

    # Data loss
    if 'train_data' in s1.columns:
        axes[0,1].semilogy(s1['epoch'], s1['train_data'], color=colors[0], alpha=0.7, label='Train', lw=1.5)
        if 'val_data' in s1.columns:
            axes[0,1].semilogy(s1['epoch'], s1['val_data'], color=colors[1], alpha=0.7, label='Val', lw=1.5)
        axes[0,1].set_title('Data Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend(fontsize=8)

    # val_real_rmse
    if 'val_real_rmse' in s1.columns:
        axes[1,0].plot(s1['epoch'], s1['val_real_rmse'], color=colors[0], alpha=0.8, lw=1.5, label='Stage 1')
        if has_retrain:
            s2 = pd.read_csv(retrain_csv)
            if 'val_real_rmse' in s2.columns:
                # offset epochs
                max_ep = s1['epoch'].max()
                axes[1,0].plot(s2['epoch'] + max_ep, s2['val_real_rmse'], color=colors[2], alpha=0.8, lw=2, label='Stage 2 (retrained)')
                axes[1,0].axvline(max_ep, color='gray', ls=':', alpha=0.5)
                axes[1,0].text(max_ep+1, s1['val_real_rmse'].min(), 'Retrain\nstart', fontsize=7, alpha=0.6)
        axes[1,0].set_title('val_real_rmse')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('RMSE')
        axes[1,0].legend(fontsize=8)

    # NFE or physics loss
    if 'nfe' in s1.columns:
        axes[1,1].plot(s1['epoch'], s1['nfe'], color=colors[2], alpha=0.8, lw=1.5)
        axes[1,1].set_title('ODE Solver NFE')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Avg NFE / batch')
    elif 'train_physics' in s1.columns:
        axes[1,1].semilogy(s1['epoch'], s1['train_physics'], color=colors[0], alpha=0.7, lw=1.5, label='Train')
        if 'val_physics' in s1.columns:
            axes[1,1].semilogy(s1['epoch'], s1['val_physics'], color=colors[1], alpha=0.7, lw=1.5, label='Val')
        axes[1,1].set_title('Physics Loss')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].legend(fontsize=8)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(fig_dir, f'fig8_training.{fmt}'))
    plt.close(fig)
    print(f"  Stage 1: {len(s1)} epochs")
    if has_retrain:
        s2 = pd.read_csv(retrain_csv)
        print(f"  Stage 2 retrain: {len(s2)} epochs")
else:
    print("  WARNING: no stage1_history.csv found")


# ====================================================================
# FIG 9: NFE evolution (from stage1 history)
# ====================================================================
print("Generating fig9_nfe...")
if has_stage1:
    s1 = pd.read_csv(stage1_csv)
    if 'nfe' in s1.columns:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(s1['epoch'], s1['nfe'], color=colors[2], lw=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg NFE / batch')
        ax.set_title('ODE Solver Efficiency')
        for fmt in ['png', 'pdf']:
            fig.savefig(os.path.join(fig_dir, f'fig9_nfe.{fmt}'))
        plt.close(fig)
        print(f"  NFE range: {s1['nfe'].min():.0f} - {s1['nfe'].max():.0f}")
    else:
        print("  WARNING: no 'nfe' column in stage1_history.csv")
else:
    print("  WARNING: no stage1_history.csv found")


print()
print("Done. All figures regenerated in:", fig_dir)
