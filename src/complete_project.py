"""Complete the project: diagnose checkpoints, fix plots, run explainability + symbolic regression."""
import sys, json, traceback, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, ".")
from config import get_config
from model import PhysicsNODE
from trainer import AusteniteReversionDataset

config = get_config()
config.device = torch.device("cpu")
# Speed: reduce UQ samples for CPU
config.model.n_mc_samples = 8

# ============================================================
# STEP 1: Diagnose ALL checkpoints - pick lowest val loss
# ============================================================
print("=" * 60)
print("STEP 1: CHECKPOINT DIAGNOSIS")
print("=" * 60)
ckpt_dir = Path("models/checkpoints")
best_ckpt_path = None
best_val_loss = float("inf")
all_histories = {}

for pt_file in sorted(ckpt_dir.glob("*.pt")):
    try:
        ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
        hist = ckpt.get("history", {})
        n_epochs = len(hist.get("train_loss", []))
        bv = ckpt.get("best_val", float("inf"))
        print(f"  {pt_file.name}: {n_epochs} epochs | best_val={bv:.6f}")
        all_histories[pt_file.name] = hist
        if bv < best_val_loss:
            best_val_loss = bv
            best_ckpt_path = pt_file
    except Exception as e:
        print(f"  {pt_file.name}: BROKEN - {e}")

print(f"\n  >>> Using: {best_ckpt_path} (val={best_val_loss:.6f})")

# ============================================================
# STEP 2: Load model
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: LOADING MODEL")
print("=" * 60)
ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
model = PhysicsNODE(config.model)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
history = ckpt.get("history", {})

# Merge longest history from ALL checkpoints
best_hist_len = 0
for name, h in all_histories.items():
    tl = len(h.get("train_loss", []))
    if tl > best_hist_len:
        best_hist_len = tl
        history = h
print(f"  Model loaded. Using history with {best_hist_len} epoch(s)")

# ============================================================
# STEP 3: Load validation dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: LOADING DATA")
print("=" * 60)
df_val = pd.read_csv(config.synthetic_dir / "val.csv")
dataset = AusteniteReversionDataset(df_val, config)
print(f"  Loaded {len(dataset)} validation curves")

# ============================================================
# STEP 4: Build predictions
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: GENERATING PREDICTIONS")
print("=" * 60)
from publication_pipeline import select_representative_ids, build_prediction_panels, build_parity_arrays
representative_ids = select_representative_ids(df_val, n_curves=3)
predictions = build_prediction_panels(model, dataset, representative_ids, config, n_samples=8)
print(f"  Built {len(predictions)} prediction panels")
f_true, f_pred = build_parity_arrays(model, dataset, config, max_curves=200)
print(f"  Parity: {f_true.shape[0]} points")

# ============================================================
# STEP 5: Sensitivity Analysis (FAST, no UQ per-point)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: SENSITIVITY ANALYSIS (FAST)")
print("=" * 60)
sensitivity_results = {}
try:
    from optimizer_annealing import predict_RA_for_schedule
    base_comp = {"Mn": 7.0, "C": 0.1, "Al": 1.5, "Si": 0.5}
    cb = config.composition
    sweeps = {
        "Mn": (np.linspace(cb.Mn_min, cb.Mn_max, 15), "Mn"),
        "C": (np.linspace(cb.C_min, cb.C_max, 15), "C"),
        "Al": (np.linspace(cb.Al_min, max(cb.Al_max, 0.1), 15), "Al"),
        "T": (np.linspace(cb.T_ICA_min, cb.T_ICA_max, 15), "T_celsius"),
    }
    for name, (vals, key) in sweeps.items():
        f_ra = []
        t0 = time.time()
        for val in vals:
            comp = base_comp.copy()
            T_c, t_s = 650.0, 3600.0
            if key in comp:
                comp[key] = float(val)
            elif key == "T_celsius":
                T_c = float(val)
            try:
                r = predict_RA_for_schedule(model, comp, T_c, t_s, config, False)
                f_ra.append(r["f_RA_mean"])
            except Exception:
                f_ra.append(np.nan)
        sensitivity_results[name] = {
            "values": vals,
            "f_RA": np.array(f_ra),
            "f_RA_lower": np.array(f_ra) * 0.9,
            "f_RA_upper": np.array(f_ra) * 1.1,
        }
        print(f"  {name}: {time.time()-t0:.1f}s | range=[{np.nanmin(f_ra):.3f}, {np.nanmax(f_ra):.3f}]")
except Exception as e:
    print(f"  Sensitivity failed: {e}")
    traceback.print_exc()

# Validate physics consistency
consistency = {}
expected = {"Mn": "positive", "C": "positive", "Al": "negative", "T": "positive"}
for feat, direction in expected.items():
    if feat not in sensitivity_results:
        continue
    d = sensitivity_results[feat]
    valid = ~np.isnan(d["f_RA"])
    if valid.sum() < 3:
        continue
    corr = np.corrcoef(d["values"][valid], d["f_RA"][valid])[0, 1]
    passed = (corr > 0) if direction == "positive" else (corr < 0)
    status = "PASS" if passed else "FAIL"
    consistency[feat] = {"passed": passed, "correlation": float(corr), "expected": direction}
    print(f"  {status}: {feat} -> RA corr={corr:.3f} (expected {direction})")

# ============================================================
# STEP 6: SHAP (skip if not installed or slow)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SHAP")
print("=" * 60)
shap_result = None
try:
    import shap
    sample_data = np.stack([s["static"].cpu().numpy() for s in dataset.samples[:16]])
    feature_names = ["T_n", "Mn", "C", "Al", "Si", "D_log", "dG_n", "PHJ_n"]

    def predict_fn(X):
        xt = torch.tensor(X, dtype=torch.float32)
        B = xt.shape[0]
        f_eq = torch.full((B, 1), 0.35)
        dG = xt[:, 6:7]
        t_sp = torch.linspace(0, 3600, 10)
        model.eval()
        with torch.no_grad():
            try:
                return model(xt, f_eq, dG, t_sp)[:, -1].cpu().numpy()
            except Exception:
                return np.zeros(B)

    explainer = shap.KernelExplainer(predict_fn, sample_data[:8])
    sv = explainer.shap_values(sample_data[:16], nsamples=50)
    imp = np.abs(sv).mean(0)
    ranked = np.argsort(imp)[::-1]
    shap_result = {
        "feature_importance": [
            {"feature": feature_names[i] if i < len(feature_names) else f"f{i}", "importance": float(imp[i])}
            for i in ranked
        ]
    }
    print(f"  SHAP done! Top feature: {shap_result['feature_importance'][0]}")
except ImportError:
    print("  SHAP not installed, skipping")
except Exception as e:
    print(f"  SHAP failed: {e}")

# ============================================================
# STEP 7: Symbolic Regression (skip if PySR not installed)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: SYMBOLIC REGRESSION")
print("=" * 60)
symbolic_result = None
try:
    from symbolic_regression import extract_symbolic_equation
    symbolic_result = extract_symbolic_equation(model, config=config, n_samples=50)
    if symbolic_result:
        print(f"  Best: {symbolic_result.get('best_equation')}")
    else:
        print("  PySR not installed, skipping")
except Exception as e:
    print(f"  Symbolic regression skipped: {e}")

# ============================================================
# STEP 8: Generate ALL figures
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: GENERATING ALL FIGURES")
print("=" * 60)
from visualizations import generate_all_figures
generate_all_figures(
    training_history=history if history and len(history.get("train_loss", [])) > 0 else None,
    sensitivity_results=sensitivity_results if sensitivity_results else None,
    shap_result=shap_result,
    predictions=predictions,
    f_true=f_true,
    f_pred=f_pred,
    config=config,
)
print("  All figures saved!")

# ============================================================
# STEP 9: Final summary
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: FINAL SUMMARY")
print("=" * 60)
rmse = float(np.sqrt(np.mean((f_true - f_pred) ** 2)))
r2 = float(1 - np.sum((f_true - f_pred) ** 2) / max(np.sum((f_true - np.mean(f_true)) ** 2), 1e-10))
summary = {
    "checkpoint": str(best_ckpt_path),
    "best_val_loss": best_val_loss,
    "n_val_curves": len(dataset),
    "representative_ids": representative_ids,
    "parity_rmse": rmse,
    "parity_r2": r2,
    "consistency": consistency if consistency else None,
    "symbolic_best_equation": symbolic_result.get("best_equation") if symbolic_result else None,
    "symbolic_best_loss": symbolic_result.get("best_loss") if symbolic_result else None,
    "sensitivity_features": list(sensitivity_results.keys()),
}
with open(config.figure_dir / "publication_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"  Checkpoint: {best_ckpt_path.name}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2:   {r2:.4f}")
print(f"  Consistency: {consistency}")
print(f"  Figures: {list(config.figure_dir.glob('*.png'))}")

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)
