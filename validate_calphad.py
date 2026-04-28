"""
CALPHAD Validation: Compare empirical Ac1/f_eq with pycalphad equilibrium calculations.

Generates:
  - fig11_calphad_comparison.png/pdf : Empirical vs CALPHAD Ac1 and f_eq
  - Printed comparison table for all compositions in the dataset
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from config import get_config
from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA

cfg = get_config()
fig_dir = Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True)

# ---- CALPHAD setup ----
try:
    from pycalphad import Database, equilibrium, variables as v
    tdb_path = Path(__file__).parent / 'src' / 'data' / 'calphad_tables' / 'FeMnC.tdb'
    db = Database(str(tdb_path))
    print(f"Loaded CALPHAD database: {tdb_path.name}")
    print(f"  Phases: {list(db.phases.keys())}")
    CALPHAD_OK = True
except Exception as e:
    print(f"CALPHAD not available: {e}")
    CALPHAD_OK = False


def calphad_fcc_fraction(db, Mn_wt, C_wt, T_celsius):
    """Compute FCC (austenite) mole fraction at equilibrium using pycalphad."""
    T_K = T_celsius + 273.15

    # Convert wt% to mole fraction
    # Atomic masses: Fe=55.845, Mn=54.938, C=12.011
    w_Fe = 100.0 - Mn_wt - C_wt
    n_Fe = w_Fe / 55.845
    n_Mn = Mn_wt / 54.938
    n_C = C_wt / 12.011
    total = n_Fe + n_Mn + n_C
    x_Mn = n_Mn / total
    x_C = n_C / total

    phases = ['FCC_A1', 'BCC_A2', 'CEMENTITE']
    comps = ['FE', 'MN', 'C', 'VA']
    conds = {v.T: T_K, v.P: 101325, v.X('MN'): x_Mn, v.X('C'): x_C}

    try:
        eq = equilibrium(db, comps, phases, conds)
        # Extract phase fractions
        phase_names = eq.Phase.values.squeeze()
        phase_fracs = eq.NP.values.squeeze()

        fcc_frac = 0.0
        for name, frac in zip(phase_names, phase_fracs):
            if isinstance(name, bytes):
                name = name.decode()
            name = str(name).strip()
            if name == 'FCC_A1' and not np.isnan(frac):
                fcc_frac += float(frac)

        return fcc_frac
    except Exception as e:
        return float('nan')


def calphad_Ac1(db, Mn_wt, C_wt, T_start=400, T_end=900, step=5):
    """Find Ac1 (first appearance of FCC) by scanning temperature."""
    for T in range(T_start, T_end, step):
        f = calphad_fcc_fraction(db, Mn_wt, C_wt, T)
        if not np.isnan(f) and f > 0.01:
            # Refine
            for T2 in range(T - step, T + 1, 1):
                f2 = calphad_fcc_fraction(db, Mn_wt, C_wt, T2)
                if not np.isnan(f2) and f2 > 0.01:
                    return T2
            return T
    return float('nan')


if CALPHAD_OK:
    # ---- Compositions from dataset ----
    compositions = [
        {'Mn': 5.0, 'C': 0.20, 'label': 'Fe-5Mn-0.2C'},
        {'Mn': 5.8, 'C': 0.12, 'label': 'Fe-5.8Mn-0.12C'},
        {'Mn': 6.0, 'C': 0.10, 'label': 'Fe-6Mn-0.1C'},
        {'Mn': 7.0, 'C': 0.10, 'label': 'Fe-7Mn-0.1C'},
        {'Mn': 7.9, 'C': 0.04, 'label': 'Fe-7.9Mn-0.04C'},
        {'Mn': 8.0, 'C': 0.05, 'label': 'Fe-8Mn-0.05C'},
        {'Mn': 9.0, 'C': 0.05, 'label': 'Fe-9Mn-0.05C'},
        {'Mn': 10.0, 'C': 0.10, 'label': 'Fe-10Mn-0.1C'},
        {'Mn': 11.0, 'C': 0.10, 'label': 'Fe-11Mn-0.1C'},
        {'Mn': 12.0, 'C': 0.10, 'label': 'Fe-12Mn-0.1C'},
    ]

    print("\n" + "="*80)
    print("CALPHAD vs EMPIRICAL Ac1 COMPARISON")
    print("="*80)
    print(f"{'Composition':<22s} {'Empirical Ac1':>13s} {'CALPHAD Ac1':>12s} {'Diff':>6s}")
    print("-"*60)

    empirical_ac1 = []
    calphad_ac1_list = []

    for comp in compositions:
        # Empirical
        emp_Ac1, emp_Ac3 = get_Ac1_Ac3(comp)
        empirical_ac1.append(emp_Ac1)

        # CALPHAD
        cal_Ac1 = calphad_Ac1(db, comp['Mn'], comp['C'])
        calphad_ac1_list.append(cal_Ac1)

        diff = cal_Ac1 - emp_Ac1 if not np.isnan(cal_Ac1) else float('nan')
        print(f"  {comp['label']:<20s} {emp_Ac1:>10.1f} C  {cal_Ac1:>9.1f} C  {diff:>+5.1f}")

    # ---- f_eq comparison at selected T ----
    print("\n" + "="*80)
    print("CALPHAD vs EMPIRICAL f_eq COMPARISON (at 650 C)")
    print("="*80)
    print(f"{'Composition':<22s} {'Empirical f_eq':>14s} {'CALPHAD f_eq':>13s} {'Diff':>7s}")
    print("-"*60)

    T_test = 650
    emp_feq = []
    cal_feq = []

    for comp in compositions:
        f_emp, _ = get_equilibrium_RA(comp, T_test, force_fallback=True)
        f_cal = calphad_fcc_fraction(db, comp['Mn'], comp['C'], T_test)
        emp_feq.append(f_emp)
        cal_feq.append(f_cal)

        diff = f_cal - f_emp if not np.isnan(f_cal) else float('nan')
        print(f"  {comp['label']:<20s} {f_emp:>11.3f}    {f_cal:>10.3f}    {diff:>+6.3f}")

    # ---- Generate comparison figure ----
    print("\nGenerating CALPHAD comparison figure...")
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 200, 'savefig.bbox': 'tight',
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Ac1 comparison
    ax = axes[0]
    labels = [c['label'].replace('Fe-', '') for c in compositions]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, empirical_ac1, width, label='Empirical (this work)',
                   color='#2176AE', alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + width/2, calphad_ac1_list, width, label='CALPHAD (Djurovic 2011)',
                   color='#D4722C', alpha=0.8, edgecolor='white')
    ax.set_ylabel('Ac1 Temperature (C)')
    ax.set_title('(a) Ac1: Empirical vs CALPHAD')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)

    # Panel B: f_eq comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, emp_feq, width, label='Empirical (this work)',
                    color='#2176AE', alpha=0.8, edgecolor='white')
    bars4 = ax2.bar(x + width/2, cal_feq, width, label='CALPHAD (Djurovic 2011)',
                    color='#D4722C', alpha=0.8, edgecolor='white')
    ax2.set_ylabel('Equilibrium Austenite Fraction')
    ax2.set_title(f'(b) f_eq at {T_test} C: Empirical vs CALPHAD')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(fig_dir / f'fig11_calphad_comparison.{fmt}')
    plt.close(fig)
    print("  Saved fig11_calphad_comparison.png/pdf")

    # ---- Phase diagram section (FCC fraction vs T for Fe-7Mn-0.1C) ----
    print("\nGenerating CALPHAD phase fraction vs temperature...")
    T_range = np.arange(400, 901, 10)
    comp_ref = {'Mn': 7.0, 'C': 0.10}
    fcc_fracs = []
    emp_fracs = []

    for T in T_range:
        f_cal = calphad_fcc_fraction(db, comp_ref['Mn'], comp_ref['C'], T)
        f_emp, _ = get_equilibrium_RA(comp_ref, T, force_fallback=True)
        fcc_fracs.append(f_cal)
        emp_fracs.append(f_emp)

    fig2, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(T_range, fcc_fracs, 'o-', color='#D4722C', markersize=3,
             label='CALPHAD (pycalphad + Djurovic 2011)')
    ax3.plot(T_range, emp_fracs, 's-', color='#2176AE', markersize=3,
             label='Empirical (recalibrated)')
    ax3.set_xlabel('Temperature (C)')
    ax3.set_ylabel('Equilibrium Austenite Fraction')
    ax3.set_title('Fe-7Mn-0.1C: Phase Fraction vs Temperature')
    ax3.legend()
    ax3.set_xlim(400, 900)
    ax3.set_ylim(-0.05, 1.05)
    ax3.axhline(0, color='gray', lw=0.5)
    ax3.axhline(1, color='gray', lw=0.5)

    for fmt in ['png', 'pdf']:
        fig2.savefig(fig_dir / f'fig12_calphad_phase_fraction.{fmt}')
    plt.close(fig2)
    print("  Saved fig12_calphad_phase_fraction.png/pdf")

    print("\nCALPHAD validation complete.")
else:
    print("Skipping CALPHAD validation (pycalphad not available)")
