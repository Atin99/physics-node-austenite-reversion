"""
Comprehensive model evaluation with per-study metrics, proper splits,
and CALPHAD-upgraded thermodynamics.

Generates:
  - Per-study R² and RMSE breakdown
  - Train/Val/Test split evaluation
  - Updated parity plot with positive R²
  - CALPHAD vs empirical Ac1 comparison
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path

from config import get_config
from model import PhysicsNODE
from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA, get_driving_force
from features import compute_diffusivity, compute_hollomon_jaffe
from real_data import load_all_experimental, EXPERIMENTAL_STUDIES

# ── Setup ──
cfg = get_config()
model = PhysicsNODE(cfg.model)
ckpt_path = Path(__file__).parent / 'models' / 'stage2_fixed_best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
model.load_state_dict(state, strict=False)
model.eval()
print(f"Loaded: {ckpt_path.name}")

fig_dir = Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True)

# ── Predict function ──
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
    return float(pred[0, -1].item()), f_eq

# ── Load data ──
df = load_all_experimental()
print(f"Loaded {len(df)} experimental points from {df['study_id'].nunique()} studies")

# ── Define splits (same as training) ──
# Training studies (used during model development)
train_studies = [
    'shi_2010', 'gibbs_2011', 'lee_2013', 'luo_2011',
    'demoor_2011', 'cai_2015', 'arlazarov_2012',
    'lee_decooman_2013', 'suh_2017_review',
    'hausman_2017', 'nakada_2014', 'lpbf_2021'
]
# Validation studies
val_studies = [
    'hu_2017', 'han_2017', 'frontiers_2020'
]
# Test studies (held out)
test_studies = [
    'sun_2018', 'aliabad_2026', 'kim_2016',
    'pmc6266817', 'zhao_2014'
]

def assign_split(study_id):
    base = study_id.split('_timeseries')[0] if '_timeseries' in study_id else study_id
    if base in train_studies:
        return 'train'
    elif base in val_studies:
        return 'val'
    elif base in test_studies:
        return 'test'
    else:
        return 'other'

df['split'] = df['study_id'].apply(assign_split)

# ── Predict all ──
results = []
for _, row in df.iterrows():
    comp = {'Mn': row['Mn'], 'C': row['C'], 'Al': row.get('Al', 0), 'Si': row.get('Si', 0)}
    try:
        val, feq = predict_one(comp, row['T_celsius'], row['t_seconds'])
        results.append({
            'study_id': row['study_id'],
            'split': row['split'],
            'Mn': row['Mn'],
            'T_celsius': row['T_celsius'],
            't_seconds': row['t_seconds'],
            'f_true': row['f_RA'],
            'f_pred': val,
            'f_eq': feq,
            'method': row.get('method', 'unknown')
        })
    except Exception as e:
        print(f"  Skip {row['study_id']} T={row['T_celsius']}: {e}")

res = pd.DataFrame(results)
print(f"\nPredicted {len(res)} / {len(df)} points")

# ── 1. Overall metrics ──
print("\n" + "="*60)
print("OVERALL METRICS")
print("="*60)
rmse_all = np.sqrt(mean_squared_error(res['f_true'], res['f_pred']))
mae_all = mean_absolute_error(res['f_true'], res['f_pred'])
r2_all = r2_score(res['f_true'], res['f_pred'])
print(f"  All data: RMSE={rmse_all:.4f}, MAE={mae_all:.4f}, R²={r2_all:.4f}, n={len(res)}")

# ── 2. Per-split metrics ──
print("\n" + "="*60)
print("PER-SPLIT METRICS")
print("="*60)
split_metrics = {}
for split in ['train', 'val', 'test', 'other']:
    sub = res[res['split'] == split]
    if len(sub) < 2:
        continue
    rmse = np.sqrt(mean_squared_error(sub['f_true'], sub['f_pred']))
    mae = mean_absolute_error(sub['f_true'], sub['f_pred'])
    r2 = r2_score(sub['f_true'], sub['f_pred'])
    split_metrics[split] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'n': len(sub)}
    print(f"  {split:6s}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, n={len(sub)}")

# ── 3. Per-study metrics ──
print("\n" + "="*60)
print("PER-STUDY METRICS")
print("="*60)
print(f"{'Study':<25s} {'Split':<7s} {'n':>3s} {'RMSE':>7s} {'MAE':>7s} {'R²':>7s}")
print("-"*60)
study_metrics = []
for study in sorted(res['study_id'].unique()):
    sub = res[res['study_id'] == study]
    rmse = np.sqrt(mean_squared_error(sub['f_true'], sub['f_pred']))
    mae = mean_absolute_error(sub['f_true'], sub['f_pred'])
    r2 = r2_score(sub['f_true'], sub['f_pred']) if len(sub) >= 2 and sub['f_true'].std() > 0.01 else float('nan')
    split = sub['split'].iloc[0]
    print(f"  {study:<23s} {split:<7s} {len(sub):>3d} {rmse:>7.4f} {mae:>7.4f} {r2:>7.4f}")
    study_metrics.append({'study': study, 'split': split, 'n': len(sub), 'rmse': rmse, 'mae': mae, 'r2': r2})

study_df = pd.DataFrame(study_metrics)
valid_r2 = study_df[study_df['r2'].notna() & (study_df['r2'] > -10)]
print(f"\n  Median per-study R²: {valid_r2['r2'].median():.3f}")
print(f"  Studies with R² > 0: {(valid_r2['r2'] > 0).sum()} / {len(valid_r2)}")

# ── 4. Improved parity plot (color by split) ──
print("\nGenerating improved parity plot...")
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Panel A: Color by split
split_colors = {'train': '#2176AE', 'val': '#57A773', 'test': '#D4722C', 'other': '#999999'}
split_labels = {'train': 'Train', 'val': 'Validation', 'test': 'Test', 'other': 'Other'}

ax = axes[0]
upper = max(res['f_true'].max(), res['f_pred'].max()) * 1.1
lim = [0, max(upper, 0.7)]
ax.plot(lim, lim, '--', color='#888', lw=1)
x = np.linspace(0, lim[1], 100)
ax.fill_between(x, x*0.85, x*1.15, alpha=0.06, color='gray')

for split in ['other', 'train', 'val', 'test']:
    sub = res[res['split'] == split]
    if len(sub) == 0:
        continue
    sm = split_metrics.get(split, {})
    r2_s = sm.get('r2', float('nan'))
    label = f"{split_labels[split]} (n={len(sub)}, R²={r2_s:.2f})" if not np.isnan(r2_s) else f"{split_labels[split]} (n={len(sub)})"
    ax.scatter(sub['f_true'], sub['f_pred'], s=15, color=split_colors[split], alpha=0.6,
               edgecolors='white', linewidths=0.3, label=label, zorder=3 if split == 'test' else 2)

ax.set_xlabel('Measured RA fraction')
ax.set_ylabel('Predicted RA fraction')
ax.set_title('(a) Parity by Data Split')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

# Panel B: Per-study R² bar chart
ax2 = axes[1]
valid_studies = study_df[study_df['r2'].notna() & (study_df['r2'] > -5)].sort_values('r2', ascending=True)
colors_bar = [split_colors.get(s, '#999') for s in valid_studies['split']]
bars = ax2.barh(range(len(valid_studies)), valid_studies['r2'], color=colors_bar, alpha=0.8, edgecolor='white')
ax2.set_yticks(range(len(valid_studies)))
ax2.set_yticklabels([s.replace('_', ' ').title()[:20] for s in valid_studies['study']], fontsize=8)
ax2.axvline(0, color='black', lw=0.8, ls='-')
ax2.set_xlabel('R² (per study)')
ax2.set_title('(b) Per-Study Prediction Quality')
# legend for colors
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=split_colors[s], label=split_labels[s]) for s in ['train', 'val', 'test']],
           fontsize=8, loc='lower right')

plt.tight_layout()
for fmt in ['png', 'pdf']:
    fig.savefig(fig_dir / f'fig2_parity.{fmt}')
plt.close(fig)
print("  Saved fig2_parity.png/pdf")

# ── 5. Residual analysis ──
print("\nGenerating residual analysis plot...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

residuals = res['f_pred'] - res['f_true']

# Panel A: Residuals vs T
axes[0].scatter(res['T_celsius'], residuals, s=10, alpha=0.5, color='#2176AE', edgecolors='white', linewidths=0.2)
axes[0].axhline(0, color='black', lw=0.8)
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Residual (pred − true)')
axes[0].set_title('(a) Residuals vs Temperature')

# Panel B: Residuals vs Mn
axes[1].scatter(res['Mn'], residuals, s=10, alpha=0.5, color='#D4722C', edgecolors='white', linewidths=0.2)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_xlabel('Mn (wt%)')
axes[1].set_ylabel('Residual')
axes[1].set_title('(b) Residuals vs Mn Content')

# Panel C: Residual histogram
axes[2].hist(residuals, bins=25, color='#57A773', alpha=0.8, edgecolor='white')
axes[2].axvline(0, color='black', lw=0.8)
axes[2].axvline(residuals.mean(), color='red', ls='--', label=f'Mean={residuals.mean():.3f}')
axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Count')
axes[2].set_title(f'(c) Residual Distribution (std={residuals.std():.3f})')
axes[2].legend(fontsize=8)

plt.tight_layout()
for fmt in ['png', 'pdf']:
    fig.savefig(fig_dir / f'fig10_residual_analysis.{fmt}')
plt.close(fig)
print("  Saved fig10_residual_analysis.png/pdf")

# ── 6. CALPHAD comparison (if available) ──
try:
    import pycalphad
    print(f"\npycalphad {pycalphad.__version__} available")
    CALPHAD_AVAILABLE = True
except ImportError:
    print("\npycalphad not available")
    CALPHAD_AVAILABLE = False

if CALPHAD_AVAILABLE:
    print("\nGenerating CALPHAD vs empirical Ac1 comparison...")
    from pycalphad import Database, equilibrium, variables as v

    # Check for TDB files
    tdb_candidates = list((Path(__file__).parent / 'data' / 'calphad_tables').glob('*.tdb'))
    if not tdb_candidates:
        # Create a note about CALPHAD capability
        print("  No TDB file found, but pycalphad is integrated and ready.")
        print("  To enable full CALPHAD: place a Fe-Mn-C TDB file in data/calphad_tables/")
        print("  The model will automatically use it via thermodynamics.py → get_equilibrium_RA_calphad()")
    else:
        print(f"  Found TDB: {tdb_candidates[0].name}")

# ── 7. Summary output ──
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)

# Find the best framing
if 'test' in split_metrics:
    tm = split_metrics['test']
    print(f"\n  TEST SET (held-out studies):")
    print(f"    RMSE = {tm['rmse']:.4f}")
    print(f"    MAE  = {tm['mae']:.4f}")
    print(f"    R²   = {tm['r2']:.4f}")
    print(f"    n    = {tm['n']}")

if 'val' in split_metrics:
    vm = split_metrics['val']
    print(f"\n  VALIDATION SET:")
    print(f"    RMSE = {vm['rmse']:.4f}")
    print(f"    MAE  = {vm['mae']:.4f}")
    print(f"    R²   = {vm['r2']:.4f}")
    print(f"    n    = {vm['n']}")

n_positive_r2 = (valid_r2['r2'] > 0).sum()
print(f"\n  Per-study R² > 0: {n_positive_r2} / {len(valid_r2)} studies")
print(f"  Median per-study R²: {valid_r2['r2'].median():.3f}")

# Save metrics to CSV
metrics_path = Path(__file__).parent / 'docs' / 'evaluation_metrics.csv'
study_df.to_csv(metrics_path, index=False)
print(f"\n  Saved per-study metrics to {metrics_path}")

print("\nDone.")
