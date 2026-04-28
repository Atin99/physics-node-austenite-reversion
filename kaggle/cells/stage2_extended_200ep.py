"""
STAGE 2 EXTENDED RETRAINING - 200 epochs, patient convergence.

Previous run (60 epochs, 31 min) was STILL improving at epoch 60:
  val_real_rmse went 0.175 -> 0.161 and never plateaued.
With 4h GPU budget, 200 epochs @ ~30s/epoch = ~100 min, well within budget.

Key changes:
  - max_epochs: 60 -> 200
  - early_stopping_patience: 10 -> 50
  - learning_rate: 8e-5 -> 3e-5 (finer convergence since we have more epochs)
  - scheduler: cosine warm restarts (T_0=40) for periodic exploration
  - Full R2 evaluation at the end
"""
import gc
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

def log(message=""):
    print(message, flush=True)

def ensure_pkg(pkg_name: str):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])

ensure_pkg("torchdiffeq")
ensure_pkg("scikit-learn")

import pandas as pd
import torch
from torch.utils.data import DataLoader

WORK_DIR = Path("/kaggle/working/stage2_extended")
ARTIFACT_DIR = Path("/kaggle/working/stage2_extended_artifacts")
REQUIRED_FILES = ["config.py", "data_generator.py", "losses.py", "model.py", "thermodynamics.py", "trainer.py"]
START_UTC = datetime.now(timezone.utc).isoformat()
START_WALL = time.time()
RUN_VERSION = "2026-04-28-stage2-extended-200ep"

# ── Workspace setup ──

def is_repo_root(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_FILES)

def score_repo(path: Path) -> tuple:
    lower = str(path).lower()
    score = 0
    if "before_kaggle" in lower: score += 100
    if "final_project_4" in lower: score += 40
    if "medium_mn_neural_ode" in lower: score += 20
    if "with_kaggle" in lower: score -= 50
    return (score, -len(str(path)))

def locate_source():
    dir_candidates = []
    for cfg_path in glob.glob("/kaggle/input/**/config.py", recursive=True):
        repo = Path(cfg_path).parent
        if is_repo_root(repo):
            dir_candidates.append(repo)
    dir_candidates = sorted(set(dir_candidates), key=score_repo, reverse=True)
    if dir_candidates:
        return dir_candidates[0]
    raise FileNotFoundError("Could not find source project input.")

def prepare_workspace():
    src = locate_source()
    log(f"Source -> {src}")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src if is_repo_root(src) else src, WORK_DIR)
    os.chdir(WORK_DIR)
    sys.path.insert(0, str(WORK_DIR))

prepare_workspace()

for mod_name in ["config", "thermodynamics", "data_generator", "losses", "model", "trainer", "features", "real_data"]:
    if mod_name in sys.modules:
        del sys.modules[mod_name]

from config import get_config
from data_generator import build_full_dataset, prepare_train_val_test_split
from model import PhysicsNODE
from thermodynamics import precompute_thermo_tables, get_Ac1_Ac3, get_equilibrium_RA, get_driving_force
from features import compute_diffusivity, compute_hollomon_jaffe
from trainer import AusteniteReversionDataset, Trainer, _collate_fn, set_seed

# ── Helpers ──

def real_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["provenance"].isin(["experimental", "user_provided"])].copy()

def sanitize_strict_time_curves(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for sid, g in df.groupby("sample_id", sort=False):
        g = g.sort_values("t_seconds").copy()
        rows = []
        for t, tg in g.groupby("t_seconds", sort=True):
            base = tg.iloc[0].copy()
            base["f_RA"] = float(tg["f_RA"].mean())
            rows.append(base)
        g2 = pd.DataFrame(rows).sort_values("t_seconds").copy()
        mask = g2["t_seconds"].diff().fillna(1.0) > 0
        g2 = g2[mask].copy()
        if len(g2) >= 2:
            cleaned.append(g2)
    if not cleaned:
        return df.iloc[0:0].copy()
    return pd.concat(cleaned, ignore_index=True)

def make_eval_loader(df, config, batch_size=1):
    ds = AusteniteReversionDataset(df, config)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)

def evaluate_split(trainer, df, name, config):
    if len(df) == 0:
        return {"name": name, "empty": True}
    loader = make_eval_loader(df, config, batch_size=1)
    result = trainer._validate(loader)
    if isinstance(result, tuple) and len(result) == 3:
        losses, violation_rate, metrics = result
    else:
        losses, violation_rate = result
        metrics = {}
    return {
        "name": name,
        "loss_total": float(losses.get("total", 0.0)),
        "loss_data": float(losses.get("data", 0.0)),
        "violation_rate": float(violation_rate),
        "metrics": metrics,
        "curves": int(df["sample_id"].nunique()),
        "points": int(len(df)),
    }


# ── Training loop ──

class SafeLiveTrainer(Trainer):
    def __init__(self, *args, stage_name="stage2", **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_name = stage_name
        self.best_metric_name = "val_real_rmse"
        self.best_checkpoint_metric = float("inf")
        self.best_val_metrics = {}

    def _train_epoch(self, loader, epoch):
        self.model.train()
        totals = {'total': 0.0, 'data': 0.0, 'physics': 0.0, 'monotone': 0.0, 'bound': 0.0}
        nfe_total, n, skipped = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        total_batches = len(loader)

        for step, batch in enumerate(loader, start=1):
            static = batch['static'].to(self.device)
            f_true = batch['traj'].to(self.device)
            t_span = batch['t_span'].to(self.device)
            f_eq = batch['f_eq'].to(self.device)
            dG = batch['dG_norm'].to(self.device)
            k_j = batch['k_jmak'].to(self.device)
            n_j = batch['n_jmak'].to(self.device)

            try:
                f_pred = self.model(static, f_eq, dG, t_span)
                ml = min(f_pred.shape[1], f_true.shape[1])
                shared = None
                for m in self.model.ode_func.net:
                    if hasattr(m, 'weight'):
                        shared = m
                        break
                loss_d = self.criterion(
                    f_pred[:, :ml], f_true[:, :ml], f_eq, t_span[:, :ml],
                    k_j, n_j, self.model, shared, epoch
                )
                loss = loss_d["total"]
                if (not torch.is_tensor(loss)) or (not loss.requires_grad):
                    skipped += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mc.gradient_clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as exc:
                skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                if step <= 5 or step % 20 == 0:
                    log(f"[{self.stage_name}] skipped batch {step}/{total_batches}: {exc}")
                continue

            for key in totals:
                if key in loss_d:
                    totals[key] += float(loss_d[key].detach().item())
            nfe_total += float(self.model.ode_func.nfe)
            self.model.ode_func.nfe = 0
            n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}, nfe_total / max(n, 1), skipped

    def train(self, train_loader, val_loader, verbose=True):
        log("=" * 70)
        log(f"Stage 2 EXTENDED 200ep | {self.device} | epochs={self.mc.max_epochs} | lr={self.mc.learning_rate} | patience={self.mc.early_stopping_patience}")
        log("=" * 70)
        history = []
        patience = 0
        best_metric = float("inf")

        for epoch in range(1, self.mc.max_epochs + 1):
            train_loss, nfe, skipped = self._train_epoch(train_loader, epoch)
            result = self._validate(val_loader)
            if isinstance(result, tuple) and len(result) == 3:
                val_loss, viol, metrics = result
            else:
                val_loss, viol = result
                metrics = {}

            metric = float(metrics.get("real_rmse", val_loss.get("total", 1e9)))
            self.scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss.get("total", 0.0)),
                "train_data": float(train_loss.get("data", 0.0)),
                "val_loss": float(val_loss.get("total", 0.0)),
                "val_rmse": float(metrics.get("rmse", 0.0)),
                "val_real_rmse": float(metrics.get("real_rmse", 0.0)),
                "val_endpoint_mae": float(metrics.get("endpoint_mae", 0.0)),
                "val_real_endpoint_mae": float(metrics.get("real_endpoint_mae", 0.0)),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "violations": float(viol),
                "nfe": float(nfe),
                "skipped": int(skipped),
            }
            history.append(row)

            is_best = ""
            if metric < best_metric:
                best_metric = metric
                patience = 0
                self.best_checkpoint_metric = metric
                self.best_val_metrics = metrics
                self.save_checkpoint("best")
                is_best = " *BEST*"
            else:
                patience += 1

            log(f"E{epoch:3d} | T:{row['train_loss']:.5f} V:{row['val_loss']:.5f} | RMSE:{row['val_rmse']:.4f} RealRMSE:{row['val_real_rmse']:.4f} | NFE:{row['nfe']:.0f} | pat={patience}/{self.mc.early_stopping_patience}{is_best}")

            if patience >= self.mc.early_stopping_patience:
                log(f"Early stopping at epoch {epoch}")
                break

            # Save history every 10 epochs
            if epoch % 10 == 0:
                pd.DataFrame(history).to_csv(ARTIFACT_DIR / "stage2_extended_history.csv", index=False)

        self.save_checkpoint("last")
        return history


# ── Config ──
config = get_config()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.model.use_homoscedastic = False
config.model.use_gradnorm = False
config.model.checkpoint_metric = "val_real_rmse"
config.model.batch_size = 24
config.model.max_epochs = 120
config.model.early_stopping_patience = 24
config.model.learning_rate = 2.5e-4
config.model.scheduler_type = "cosine"
config.model.scheduler_eta_min = 5e-6
config.model.use_amp = False
config.model.adjoint = False
config.model.rtol = 5e-3
config.model.atol = 1e-4
config.model.max_num_steps = 512
config.model.swa_start_epoch = config.model.max_epochs + 10
config.model.gradient_clip_val = 1.0
config.data.synthetic_calibration_samples = 250
config.data.synthetic_exploration_samples = 700
config.data.real_data_weight = 5.0
config.data.provenance_aware_loss = True
config.data.real_curve_group_min_points = 2

set_seed(config.model.random_seed)

log("=" * 90)
log("STAGE 2 EXTENDED RETRAINING (200 epochs, patience=50)")
log("=" * 90)
log(f"Run version: {RUN_VERSION}")
log(f"UTC start: {START_UTC}")

precompute_thermo_tables(config, n_comp=16, n_temp=24)
full_df = build_full_dataset(config)
train_df, val_df, test_df = prepare_train_val_test_split(full_df, config)
train_real_df = sanitize_strict_time_curves(real_only(train_df))
val_real_df = sanitize_strict_time_curves(real_only(val_df))
test_real_df = sanitize_strict_time_curves(real_only(test_df))

log(f"train_real: {train_real_df['sample_id'].nunique()} curves, {len(train_real_df)} points")
log(f"val_real: {val_real_df['sample_id'].nunique()} curves, {len(val_real_df)} points")
log(f"test_real: {test_real_df['sample_id'].nunique()} curves, {len(test_real_df)} points")

# ── Load starting checkpoint ──
# Priority: stage2_fixed_best.pt (current best) > stage1_best.pt > any .pt
start_ckpt = None

# 1. Check in the workspace (copied from input dataset)
for name in ["stage2_fixed_best.pt", "stage1_best.pt"]:
    candidate = WORK_DIR / name
    if candidate.exists():
        start_ckpt = candidate
        break

# 2. Check in /kaggle/input (uploaded datasets)
if start_ckpt is None:
    for name in ["stage2_fixed_best.pt", "stage1_best.pt", "final_best_real.pt"]:
        for hit in Path("/kaggle/input").rglob(name):
            start_ckpt = hit
            break
        if start_ckpt:
            break

# 3. Last resort: any .pt file in input
if start_ckpt is None:
    for pt in Path("/kaggle/input").rglob("*.pt"):
        start_ckpt = pt
        log(f"  Found checkpoint (fallback): {pt}")
        break

assert start_ckpt is not None and start_ckpt.exists(), \
    "No checkpoint found. Upload stage2_fixed_best.pt as a Kaggle dataset."
log(f"Starting from checkpoint: {start_ckpt}")

# ── Stage 2 config (the key changes) ──
stage2_config = deepcopy(config)
stage2_config.model.learning_rate = 3e-5        # finer convergence (was 8e-5)
stage2_config.model.max_epochs = 200             # 200 epochs (~100 min on T4)
stage2_config.model.early_stopping_patience = 9999  # NO early stopping — train all 200
stage2_config.model.batch_size = 1               # per-curve training
stage2_config.model.scheduler_type = "cosine_warm_restarts"
stage2_config.model.scheduler_T_0 = 40           # restart every 40 epochs
stage2_config.model.scheduler_T_mult = 2         # double restart period
stage2_config.model.scheduler_eta_min = 1e-6
stage2_config.model.swa_start_epoch = stage2_config.model.max_epochs + 10
stage2_config.model.use_amp = False
stage2_config.model.gradient_clip_val = 0.5      # tighter clipping for stability

# ── Load and train ──
stage2_model = PhysicsNODE(stage2_config.model)
ckpt = torch.load(start_ckpt, map_location=stage2_config.device, weights_only=False)
state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
stage2_model.load_state_dict(state, strict=False)
for p in stage2_model.parameters():
    p.requires_grad_(True)

train_ds = AusteniteReversionDataset(train_real_df, stage2_config)
val_ds = AusteniteReversionDataset(val_real_df, stage2_config)
stage2_train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate_fn, num_workers=0)
stage2_val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate_fn, num_workers=0)

stage2_trainer = SafeLiveTrainer(stage2_model, stage2_config, use_gradnorm=False, use_homoscedastic=False, stage_name="stage2_ext")
stage2_history = stage2_trainer.train(stage2_train_loader, stage2_val_loader, verbose=True)

# ── Save artifacts ──
stage2_best_path = stage2_config.checkpoint_dir / "physics_node_best.pt"
stage2_last_path = stage2_config.checkpoint_dir / "physics_node_last.pt"
if stage2_best_path.exists():
    shutil.copy2(stage2_best_path, ARTIFACT_DIR / "stage2_extended_best.pt")
if stage2_last_path.exists():
    shutil.copy2(stage2_last_path, ARTIFACT_DIR / "stage2_extended_last.pt")

pd.DataFrame(stage2_history).to_csv(ARTIFACT_DIR / "stage2_extended_history.csv", index=False)

# ── Full evaluation ──
log("\n" + "=" * 70)
log("FULL EVALUATION")
log("=" * 70)

final_trainer = stage2_trainer
final_config = stage2_config
final_trainer.load_checkpoint("best")

eval_results = {
    "val_full": evaluate_split(final_trainer, val_df, "val_full", final_config),
    "test_full": evaluate_split(final_trainer, test_df, "test_full", final_config),
    "val_real": evaluate_split(final_trainer, val_real_df, "val_real", final_config),
    "test_real": evaluate_split(final_trainer, test_real_df, "test_real", final_config),
}

for name, result in eval_results.items():
    log(f"  {name}: {json.dumps(result, indent=2, default=str)}")

# ── Point-level R2 evaluation ──
log("\n" + "=" * 70)
log("POINT-LEVEL R2 EVALUATION")
log("=" * 70)

stage2_model.eval()
cfg = stage2_config

def predict_point(comp, T_c, t_sec):
    T_K = T_c + 273.15
    f_eq, _ = get_equilibrium_RA(comp, T_c, force_fallback=True)
    dG = get_driving_force(comp, T_c, force_fallback=True)
    D = compute_diffusivity(T_K)
    P = compute_hollomon_jaffe(T_K, max(t_sec/2, 1.0))
    static = torch.tensor([[(T_K - cfg.data.T_ref)/cfg.data.T_scale, comp['Mn'], comp['C'],
                             comp.get('Al',0), comp.get('Si',0), np.log10(D+1e-30),
                             dG/1000.0, P/20000.0]], dtype=torch.float32).to(cfg.device)
    f_eq_t = torch.tensor([[f_eq]], dtype=torch.float32).to(cfg.device)
    dG_t = torch.tensor([[dG/1000.0]], dtype=torch.float32).to(cfg.device)
    t_span = torch.linspace(0, float(t_sec), 50).to(cfg.device)
    with torch.no_grad():
        pred = stage2_model(static, f_eq_t, dG_t, t_span)
    return float(pred[0, -1].cpu().item())

try:
    from real_data import load_all_experimental
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    exp_df = load_all_experimental()
    preds, trues = [], []
    for _, row in exp_df.iterrows():
        comp = {'Mn': row['Mn'], 'C': row['C'], 'Al': row.get('Al', 0), 'Si': row.get('Si', 0)}
        try:
            val = predict_point(comp, row['T_celsius'], row['t_seconds'])
            preds.append(val)
            trues.append(row['f_RA'])
        except:
            continue

    preds, trues = np.array(preds), np.array(trues)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    log(f"\n  ALL EXPERIMENTAL DATA (n={len(preds)}):")
    log(f"    RMSE = {rmse:.4f}")
    log(f"    MAE  = {mae:.4f}")
    log(f"    R2   = {r2:.4f}")
except Exception as e:
    log(f"  Point-level evaluation failed: {e}")

# ── Summary ──
summary = {
    "started_utc": START_UTC,
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "runtime_minutes": round((time.time() - START_WALL) / 60.0, 2),
    "run_version": RUN_VERSION,
    "source_checkpoint": str(start_ckpt),
    "device": str(final_config.device),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "stage2_config": {
        "learning_rate": stage2_config.model.learning_rate,
        "max_epochs": stage2_config.model.max_epochs,
        "early_stopping_patience": stage2_config.model.early_stopping_patience,
        "scheduler": stage2_config.model.scheduler_type,
        "gradient_clip": stage2_config.model.gradient_clip_val,
    },
    "best_metric_name": final_trainer.best_metric_name,
    "best_metric_value": float(final_trainer.best_checkpoint_metric),
    "best_val_metrics": final_trainer.best_val_metrics,
    "evaluation": eval_results,
    "total_epochs_trained": len(stage2_history),
}

(ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
zip_path = shutil.make_archive("/kaggle/working/stage2_extended_artifacts", "zip", ARTIFACT_DIR)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

log("\n" + "=" * 90)
log("STAGE 2 EXTENDED RETRAINING COMPLETE")
log("=" * 90)
log(f"Best {final_trainer.best_metric_name}: {final_trainer.best_checkpoint_metric:.6f}")
log(f"Total epochs: {len(stage2_history)}")
log(f"Runtime: {(time.time() - START_WALL) / 60:.1f} min")
log(f"Artifacts: {ARTIFACT_DIR}")
log(f"Zip: {zip_path}")
