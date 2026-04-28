"""
Stage 2 fine-tuning ONLY with fixed thermodynamics.
Expected runtime: 15-20 min on T4 GPU.

Upload dataset: retrain_v2_stage2_only.zip
This contains the source code + data + stage1 checkpoint.
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
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

def log(msg=""):
    print(msg, flush=True)

def ensure_pkg(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

ensure_pkg("torchdiffeq")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

WORK_DIR = Path("/kaggle/working/retrain_v2")
ARTIFACT_DIR = Path("/kaggle/working/retrain_v2_artifacts")
REQUIRED_FILES = ["config.py", "data_generator.py", "losses.py", "model.py", "thermodynamics.py", "trainer.py"]
START_UTC = datetime.now(timezone.utc).isoformat()
START_WALL = time.time()

# ============================================================
# SETUP
# ============================================================
def is_repo_root(p):
    return p.is_dir() and all((p / n).exists() for n in REQUIRED_FILES)

def find_source():
    # first try to find extracted zip contents
    for cfg in glob.glob("/kaggle/input/**/config.py", recursive=True):
        repo = Path(cfg).parent
        if is_repo_root(repo):
            return repo
    # try extracting zips
    for zp in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        tmp = Path("/kaggle/working/_tmp_extract")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(tmp)
        for cfg in tmp.rglob("config.py"):
            repo = cfg.parent
            if is_repo_root(repo):
                return repo
    raise FileNotFoundError("No valid source found in /kaggle/input/")

def find_checkpoint():
    # look for stage1_best.pt in input
    for pt in glob.glob("/kaggle/input/**/stage1_best.pt", recursive=True):
        return Path(pt)
    # also check extracted tmp
    for pt in glob.glob("/kaggle/working/_tmp_extract/**/stage1_best.pt", recursive=True):
        return Path(pt)
    raise FileNotFoundError("No stage1_best.pt found")

log("Finding source code...")
src = find_source()
log(f"Source: {src}")

log("Finding checkpoint...")
ckpt_path = find_checkpoint()
log(f"Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size/1024:.0f} KB)")

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
if ARTIFACT_DIR.exists():
    shutil.rmtree(ARTIFACT_DIR)
ARTIFACT_DIR.mkdir(parents=True)
shutil.copytree(src, WORK_DIR)

os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

for mod in ["config", "thermodynamics", "data_generator", "losses", "model", "trainer", "features"]:
    if mod in sys.modules:
        del sys.modules[mod]

from config import get_config
from data_generator import build_full_dataset, prepare_train_val_test_split
from model import PhysicsNODE
from thermodynamics import precompute_thermo_tables
from trainer import AusteniteReversionDataset, Trainer, _collate_fn, set_seed

# ============================================================
# CONFIG
# ============================================================
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
config.model.swa_start_epoch = 999
config.model.gradient_clip_val = 1.0
config.data.synthetic_calibration_samples = 250
config.data.synthetic_exploration_samples = 700
config.data.real_data_weight = 5.0
config.data.provenance_aware_loss = True
config.data.real_curve_group_min_points = 2

set_seed(config.model.random_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

log("=" * 70)
log("STAGE 2 FINE-TUNING WITH FIXED THERMODYNAMICS")
log("=" * 70)
log(f"Device: {config.device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATA
# ============================================================
log("\n[1/4] Precomputing thermodynamic tables...")
precompute_thermo_tables(config, n_comp=20, n_temp=30)

log("\n[2/4] Building dataset...")
full_df = build_full_dataset(config)
train_df, val_df, test_df = prepare_train_val_test_split(full_df, config)

def real_only(df):
    return df[df["provenance"].isin(["experimental", "user_provided"])].copy()

def sanitize(df):
    cleaned = []
    for sid, g in df.groupby("sample_id", sort=False):
        g = g.sort_values("t_seconds").copy()
        rows = []
        for t, tg in g.groupby("t_seconds", sort=True):
            base = tg.iloc[0].copy()
            base["f_RA"] = float(tg["f_RA"].mean())
            rows.append(base)
        g2 = pd.DataFrame(rows).sort_values("t_seconds")
        mask = g2["t_seconds"].diff().fillna(1.0) > 0
        g2 = g2[mask]
        if len(g2) >= 2:
            cleaned.append(g2)
    return pd.concat(cleaned, ignore_index=True) if cleaned else df.iloc[0:0].copy()

train_real = sanitize(real_only(train_df))
val_real = sanitize(real_only(val_df))
test_real = sanitize(real_only(test_df))

log(f"  train_real: {train_real['sample_id'].nunique()} curves, {len(train_real)} points")
log(f"  val_real:   {val_real['sample_id'].nunique()} curves, {len(val_real)} points")
log(f"  test_real:  {test_real['sample_id'].nunique()} curves, {len(test_real)} points")

# ============================================================
# STAGE 2
# ============================================================
log("\n[3/4] Loading stage 1 checkpoint and starting stage 2...")

s2_config = deepcopy(config)
s2_config.model.learning_rate = 8e-5
s2_config.model.max_epochs = 60
s2_config.model.early_stopping_patience = 12
s2_config.model.batch_size = 1
s2_config.model.scheduler_type = "cosine"
s2_config.model.scheduler_eta_min = 1e-6
s2_config.model.swa_start_epoch = 999

model = PhysicsNODE(s2_config.model)
ckpt = torch.load(ckpt_path, map_location=s2_config.device, weights_only=False)
state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
model.load_state_dict(state, strict=False)
model.to(s2_config.device)
for p in model.parameters():
    p.requires_grad_(True)
log(f"  Loaded checkpoint, {sum(p.numel() for p in model.parameters())} params")

train_ds = AusteniteReversionDataset(train_real, s2_config)
val_ds = AusteniteReversionDataset(val_real, s2_config)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate_fn, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate_fn, num_workers=0)

class Stage2Trainer(Trainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.best_metric_name = "val_real_rmse"
        self.best_checkpoint_metric = float("inf")
        self.best_val_metrics = {}

    def _train_epoch(self, loader, epoch):
        self.model.train()
        totals = {'total': 0.0, 'data': 0.0, 'physics': 0.0, 'monotone': 0.0, 'bound': 0.0}
        nfe_total, n, skipped = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        total_batches = len(loader)
        for step, batch in enumerate(loader, 1):
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
                loss_d = self.criterion(f_pred[:,:ml], f_true[:,:ml], f_eq, t_span[:,:ml], k_j, n_j, self.model, shared, epoch)
                loss = loss_d["total"]
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    skipped += 1
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.mc.gradient_clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            except Exception as exc:
                skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                if step <= 3:
                    log(f"  skip batch {step}: {exc}")
                continue
            for k in totals:
                if k in loss_d:
                    totals[k] += float(loss_d[k].detach())
            nfe_total += float(self.model.ode_func.nfe)
            self.model.ode_func.nfe = 0
            n += 1
        return {k: v/max(n,1) for k,v in totals.items()}, nfe_total/max(n,1), skipped

    def train(self, train_loader, val_loader, verbose=True):
        log(f"  Training: {self.mc.max_epochs} epochs, lr={self.mc.learning_rate}, patience={self.mc.early_stopping_patience}")
        history = []
        patience = 0
        best = float("inf")
        for epoch in range(1, self.mc.max_epochs + 1):
            t0 = time.time()
            train_loss, nfe, skipped = self._train_epoch(train_loader, epoch)
            result = self._validate(val_loader)
            val_loss, viol, metrics = result if len(result) == 3 else (*result, {})
            metric = float(metrics.get("real_rmse", val_loss.get("total", 1e9)))
            self.scheduler.step()
            dt = time.time() - t0
            row = {
                "epoch": epoch,
                "train_loss": float(train_loss.get("total", 0)),
                "val_loss": float(val_loss.get("total", 0)),
                "val_rmse": float(metrics.get("rmse", 0)),
                "val_real_rmse": float(metrics.get("real_rmse", 0)),
                "val_endpoint_mae": float(metrics.get("endpoint_mae", 0)),
                "val_real_endpoint_mae": float(metrics.get("real_endpoint_mae", 0)),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "nfe": float(nfe),
                "skipped": skipped,
                "time_s": dt,
            }
            history.append(row)
            marker = " *" if metric < best else ""
            log(f"  E{epoch:3d} | T:{row['train_loss']:.5f} V:{row['val_loss']:.5f} | RMSE:{row['val_rmse']:.5f} RealRMSE:{row['val_real_rmse']:.5f} | EndMAE:{row['val_endpoint_mae']:.5f} RealEnd:{row['val_real_endpoint_mae']:.5f} | {dt:.0f}s{marker}")
            if metric < best:
                best = metric
                patience = 0
                self.best_checkpoint_metric = metric
                self.best_val_metrics = metrics
                self.save_checkpoint("best")
            else:
                patience += 1
                if patience >= self.mc.early_stopping_patience:
                    log(f"  Early stop at epoch {epoch}")
                    break
        self.save_checkpoint("last")
        return history

trainer = Stage2Trainer(model, s2_config, use_gradnorm=False, use_homoscedastic=False)
history = trainer.train(train_loader, val_loader)

# ============================================================
# EVALUATE + SAVE
# ============================================================
log("\n[4/4] Evaluating and saving artifacts...")
trainer.load_checkpoint("best")

def eval_split(trainer, df, name, cfg):
    if len(df) == 0:
        return {"name": name, "empty": True}
    loader = DataLoader(AusteniteReversionDataset(df, cfg), batch_size=1, shuffle=False, collate_fn=_collate_fn, num_workers=0)
    result = trainer._validate(loader)
    losses, viol, metrics = result if len(result) == 3 else (*result, {})
    out = {"name": name, "rmse": float(metrics.get("rmse", 0)), "real_rmse": float(metrics.get("real_rmse", 0)),
           "mae": float(metrics.get("mae", 0)), "real_mae": float(metrics.get("real_mae", 0)),
           "endpoint_mae": float(metrics.get("endpoint_mae", 0)), "real_endpoint_mae": float(metrics.get("real_endpoint_mae", 0))}
    log(f"  {name}: rmse={out['rmse']:.5f} real_rmse={out['real_rmse']:.5f} endpoint_mae={out['endpoint_mae']:.5f}")
    return out

evals = {
    "val_real": eval_split(trainer, val_real, "val_real", s2_config),
    "test_real": eval_split(trainer, test_real, "test_real", s2_config),
    "val_full": eval_split(trainer, val_df, "val_full", s2_config),
    "test_full": eval_split(trainer, test_df, "test_full", s2_config),
}

# save checkpoint
best_pt = s2_config.checkpoint_dir / "physics_node_best.pt"
last_pt = s2_config.checkpoint_dir / "physics_node_last.pt"
if best_pt.exists():
    shutil.copy2(best_pt, ARTIFACT_DIR / "stage2_fixed_best.pt")
if last_pt.exists():
    shutil.copy2(last_pt, ARTIFACT_DIR / "stage2_fixed_last.pt")

pd.DataFrame(history).to_csv(ARTIFACT_DIR / "stage2_history.csv", index=False)

summary = {
    "started_utc": START_UTC,
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "runtime_minutes": round((time.time() - START_WALL) / 60, 2),
    "device": str(s2_config.device),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "stage1_checkpoint": str(ckpt_path),
    "best_metric": float(trainer.best_checkpoint_metric),
    "best_val_metrics": trainer.best_val_metrics,
    "evaluation": evals,
}
(ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

zip_path = shutil.make_archive("/kaggle/working/retrain_v2_artifacts", "zip", ARTIFACT_DIR)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

log("\n" + "=" * 70)
log("DONE")
log("=" * 70)
log(json.dumps(evals, indent=2))
log(f"Best val_real_rmse: {trainer.best_checkpoint_metric:.6f}")
log(f"Artifacts: {zip_path}")
