"""
Backend validation: tests the trained model against known literature data.
Checks prediction accuracy, physical sanity, and edge cases.
Does NOT save anything. Just prints results.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np

from config import get_config
from model import PhysicsNODE
from thermodynamics import get_Ac1_Ac3, get_equilibrium_RA, get_driving_force
from features import compute_diffusivity, compute_hollomon_jaffe, compute_Md30

cfg = get_config()
model = PhysicsNODE(cfg.model)

# load best checkpoint
candidates = [
    os.path.join(os.path.dirname(__file__), 'models', 'stage2_fixed_best.pt'),
    os.path.join(os.path.dirname(__file__), 'models', 'final_best_stage2_fixed.pt'),
    os.path.join(os.path.dirname(__file__), 'models', 'physics_node_best.pt'),
]
loaded = False
for p in candidates:
    if os.path.exists(p):
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
        model.load_state_dict(state, strict=False)
        print("Loaded: %s (%.0f KB)" % (os.path.basename(p), os.path.getsize(p)/1024))
        loaded = True
        break

if not loaded:
    print("ERROR: no checkpoint found")
    sys.exit(1)

model.eval()
n_params = sum(p.numel() for p in model.parameters())
print("Parameters: %d" % n_params)
print()


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


def predict_trajectory(comp, T_c, t_sec):
    _, _, curve = predict_one(comp, T_c, t_sec)
    return curve


# ============================================================
# TEST 1: predictions vs known literature values
# ============================================================
print("=" * 70)
print("TEST 1: PREDICTIONS vs KNOWN LITERATURE VALUES")
print("=" * 70)

test_cases = [
    ({'Mn':7.1,'C':0.10,'Al':0,'Si':0}, 650, 604800, 0.435, 'Gibbs 2011 650C 1wk'),
    ({'Mn':7.1,'C':0.10,'Al':0,'Si':0}, 625, 604800, 0.428, 'Gibbs 2011 625C 1wk'),
    ({'Mn':7.1,'C':0.10,'Al':0,'Si':0}, 600, 604800, 0.343, 'Gibbs 2011 600C 1wk'),
    ({'Mn':5.0,'C':0.20,'Al':0,'Si':0}, 650, 86400, 0.35, 'Luo 2011 650C 24h'),
    ({'Mn':5.0,'C':0.20,'Al':0,'Si':0}, 650, 3600, 0.10, 'Luo 2011 650C 1h'),
    ({'Mn':5.0,'C':0.12,'Al':1.0,'Si':0}, 650, 1800, 0.39, 'PMC6266817 CR peak'),
    ({'Mn':9.0,'C':0.05,'Al':0,'Si':0}, 675, 3600, 0.35, 'Han 2014 peak'),
    ({'Mn':7.9,'C':0.07,'Al':0.05,'Si':0.14}, 640, 3600, 0.38, 'Zhao 2014 peak'),
    ({'Mn':6.0,'C':0.10,'Al':3.0,'Si':0}, 750, 3600, 0.45, 'Suh 2017 peak'),
    ({'Mn':12.0,'C':0.05,'Al':0,'Si':0}, 650, 3600, 0.25, 'Sun 2018 peak'),
]

print("%-25s %7s %9s %7s %6s %5s" % ('Study', 'Actual', 'Pred', 'Error', 'f_eq', 'OK?'))
print("-" * 65)

errors_abs = []
for comp, T, t, actual, study in test_cases:
    pred_val, f_eq, _ = predict_one(comp, T, t)
    err = pred_val - actual
    errors_abs.append(abs(err))
    ok = "PASS" if abs(err) < 0.15 else "FAIL"
    print("%-25s %7.3f %9.3f %+7.3f %6.3f %5s" % (study, actual, pred_val, err, f_eq, ok))

mae = np.mean(errors_abs)
rmse = np.sqrt(np.mean([e**2 for e in errors_abs]))
print()
print("MAE:  %.4f" % mae)
print("RMSE: %.4f" % rmse)
print("Max error: %.4f" % max(errors_abs))
print()


# ============================================================
# TEST 2: physical sanity checks
# ============================================================
print("=" * 70)
print("TEST 2: PHYSICAL SANITY CHECKS")
print("=" * 70)

comp_test = {'Mn': 7.0, 'C': 0.10, 'Al': 0, 'Si': 0}

# 2a: monotonicity - RA should not decrease during isothermal hold
print("\n2a. Monotonicity check (RA should never decrease during hold)")
violations = 0
total_checks = 0
for T in [575, 600, 625, 650, 675, 700]:
    curve = predict_trajectory(comp_test, T, 86400)
    diffs = np.diff(curve)
    neg = np.sum(diffs < -1e-6)
    total_checks += len(diffs)
    violations += neg
    status = "PASS" if neg == 0 else "FAIL (%d violations)" % neg
    print("  T=%dC: min(df)=%.6f max(df)=%.6f  %s" % (T, diffs.min(), diffs.max(), status))
print("  Total: %d/%d steps checked, %d violations" % (total_checks, total_checks, violations))

# 2b: boundary check - RA should be in [0, f_eq]
print("\n2b. Boundary check (0 <= RA <= f_eq)")
bound_violations = 0
for T in [575, 600, 625, 650, 675, 700]:
    curve = predict_trajectory(comp_test, T, 86400)
    f_eq, _ = get_equilibrium_RA(comp_test, T, force_fallback=True)
    neg_count = np.sum(curve < -1e-6)
    over_count = np.sum(curve > f_eq * 1.05)
    bound_violations += neg_count + over_count
    status = "PASS" if (neg_count + over_count) == 0 else "FAIL"
    print("  T=%dC: min=%.4f max=%.4f f_eq=%.4f  %s" % (T, curve.min(), curve.max(), f_eq, status))
print("  Total boundary violations: %d" % bound_violations)

# 2c: temperature dependence - peak RA should be at intermediate T, not extremes
print("\n2c. Temperature dependence (should show peak at intermediate T)")
t_sweep_results = []
for T in range(550, 760, 10):
    val, f_eq, _ = predict_one(comp_test, T, 3600)
    t_sweep_results.append((T, val, f_eq))
    
peak_T = max(t_sweep_results, key=lambda x: x[1])
print("  Peak RA: %.3f at %dC (f_eq=%.3f)" % (peak_T[1], peak_T[0], peak_T[2]))
# check that RA at 550 < peak and RA at 750 < peak
ra_550 = [x for x in t_sweep_results if x[0] == 550][0][1]
ra_750 = [x for x in t_sweep_results if x[0] == 750][0][1]
if ra_550 < peak_T[1] and ra_750 < peak_T[1] and 600 <= peak_T[0] <= 720:
    print("  PASS: peak at reasonable temperature, lower at extremes")
else:
    print("  WARNING: unexpected temperature dependence")
print("  T sweep (1h hold):")
for T, ra, feq in t_sweep_results[::2]:
    bar = "#" * int(ra * 100)
    print("    %dC: %.3f  %s" % (T, ra, bar))

# 2d: composition dependence - higher Mn should shift things
print("\n2d. Composition sensitivity (Mn effect at 650C, 1h)")
for mn in [4.0, 6.0, 8.0, 10.0, 12.0]:
    comp_mn = {'Mn': mn, 'C': 0.10, 'Al': 0, 'Si': 0}
    val, f_eq, _ = predict_one(comp_mn, 650, 3600)
    print("  Mn=%.1f: RA=%.3f  f_eq=%.3f" % (mn, val, f_eq))

# 2e: time dependence - should be sigmoidal
print("\n2e. Time dependence (Fe-7Mn-0.1C at 650C)")
for t_sec in [60, 300, 1800, 3600, 14400, 86400, 604800]:
    val, _, _ = predict_one(comp_test, 650, t_sec)
    t_label = "%ds" % t_sec if t_sec < 3600 else "%.1fh" % (t_sec/3600)
    if t_sec >= 86400:
        t_label = "%.1fd" % (t_sec/86400)
    print("  t=%-8s RA=%.4f" % (t_label, val))


# ============================================================
# TEST 3: edge cases
# ============================================================
print()
print("=" * 70)
print("TEST 3: EDGE CASES")
print("=" * 70)

# 3a: below Ac1 - should give near-zero RA
print("\n3a. Below Ac1 (should give near-zero RA)")
comp_low = {'Mn': 5.0, 'C': 0.10, 'Al': 0, 'Si': 0}
Ac1, Ac3 = get_Ac1_Ac3(comp_low)
print("  Ac1=%.0fC, Ac3=%.0fC" % (Ac1, Ac3))
val_below, _, _ = predict_one(comp_low, Ac1 - 50, 3600)
print("  T=%dC (Ac1-50): RA=%.4f  %s" % (Ac1-50, val_below, "PASS" if val_below < 0.05 else "CONCERN"))

# 3b: very short time - should give near-initial
print("\n3b. Very short time (t=1s, should give near-zero)")
val_short, _, _ = predict_one(comp_test, 650, 1)
print("  t=1s: RA=%.4f  %s" % (val_short, "PASS" if val_short < 0.05 else "CONCERN"))

# 3c: very long time - should approach f_eq
print("\n3c. Very long time (t=7 days)")
val_long, f_eq_long, _ = predict_one(comp_test, 650, 604800)
ratio = val_long / f_eq_long if f_eq_long > 0 else 0
print("  t=7d: RA=%.4f f_eq=%.4f ratio=%.2f  %s" % (val_long, f_eq_long, ratio, "PASS" if ratio > 0.5 else "CONCERN"))

# 3d: high Al alloy
print("\n3d. High-Al alloy (Fe-9.4Mn-0.2C-4.3Al)")
comp_hial = {'Mn': 9.4, 'C': 0.20, 'Al': 4.3, 'Si': 0}
val_hial, f_eq_hial, _ = predict_one(comp_hial, 800, 3600)
print("  800C/1h: RA=%.3f f_eq=%.3f (actual from PMC11173901: 0.599)" % (val_hial, f_eq_hial))

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("Prediction MAE against literature: %.4f (%.1f%%)" % (mae, mae*100))
print("Prediction RMSE against literature: %.4f (%.1f%%)" % (rmse, rmse*100))
print("Monotonicity violations: %d" % violations)
print("Boundary violations: %d" % bound_violations)
print("Physics constraints: %s" % ("ALL SATISFIED" if violations == 0 and bound_violations == 0 else "ISSUES FOUND"))
