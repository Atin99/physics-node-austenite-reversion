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

def log(message=""):
    print(message, flush=True)

def ensure_pkg(pkg_name: str):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])

ensure_pkg("torchdiffeq")

import pandas as pd
import torch
from torch.utils.data import DataLoader

WORK_DIR = Path("/kaggle/working/final_project_4_stage2_retry_fixed")
ARTIFACT_DIR = Path("/kaggle/working/final_project_4_stage2_fixed_artifacts")
REQUIRED_FILES = ["config.py", "data_generator.py", "losses.py", "model.py", "thermodynamics.py", "trainer.py"]
START_UTC = datetime.now(timezone.utc).isoformat()
START_WALL = time.time()
RUN_VERSION = "2026-04-28-stage2-retry-fixed"

def is_repo_root(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_FILES)

def score_repo(path: Path) -> tuple:
    lower = str(path).lower()
    score = 0
    if "before_kaggle" in lower:
        score += 100
    if "final_project_4" in lower:
        score += 40
    if "medium_mn_neural_ode" in lower:
        score += 20
    if "with_kaggle" in lower:
        score -= 50
    return (score, -len(str(path)))

def find_repo_roots(root: Path):
    hits = []
    for cfg_path in root.rglob("config.py"):
        candidate = cfg_path.parent
        if is_repo_root(candidate):
            hits.append(candidate)
    if not hits:
        raise FileNotFoundError("Could not find a valid project root.")
    return sorted(set(hits), key=score_repo, reverse=True)

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

    repo_root = src if is_repo_root(src) else find_repo_roots(src)[0]
    shutil.copytree(repo_root, WORK_DIR)

    os.chdir(WORK_DIR)
    sys.path.insert(0, str(WORK_DIR))

prepare_workspace()

for mod_name in ["config", "thermodynamics", "data_generator", "losses", "model", "trainer"]:
    if mod_name in sys.modules:
        del sys.modules[mod_name]

from config import get_config
from data_generator import build_full_dataset, prepare_train_val_test_split
from model import PhysicsNODE
from thermodynamics import precompute_thermo_tables
from trainer import AusteniteReversionDataset, Trainer, _collate_fn, set_seed

def describe_df(name: str, df: pd.DataFrame) -> dict:
    curves = int(df["sample_id"].nunique()) if len(df) else 0
    points = int(len(df))
    out = {"name": name, "curves": curves, "points": points}
    if "provenance" in df.columns and len(df):
        out["points_by_provenance"] = {str(k): int(v) for k, v in df["provenance"].value_counts().to_dict().items()}
        out["curves_by_provenance"] = {str(k): int(v) for k, v in df.groupby("provenance")["sample_id"].nunique().to_dict().items()}
    return out

def real_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["provenance"].isin(["experimental", "user_provided"])].copy()

def sanitize_strict_time_curves(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for sid, g in df.groupby("sample_id", sort=False):
        g = g.sort_values("t_seconds").copy()

        # collapse duplicate times by averaging f_RA and keeping first metadata row
        rows = []
        for t, tg in g.groupby("t_seconds", sort=True):
            base = tg.iloc[0].copy()
            base["f_RA"] = float(tg["f_RA"].mean())
            rows.append(base)
        g2 = pd.DataFrame(rows).sort_values("t_seconds").copy()

        # enforce strictly increasing time
        mask = g2["t_seconds"].diff().fillna(1.0) > 0
        g2 = g2[mask].copy()

        if len(g2) >= 2:
            cleaned.append(g2)

    if not cleaned:
        return df.iloc[0:0].copy()
    return pd.concat(cleaned, ignore_index=True)

def make_eval_loader(df: pd.DataFrame, config, batch_size=1):
    ds = AusteniteReversionDataset(df, config)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0)

def evaluate_split(trainer: Trainer, df: pd.DataFrame, name: str, config):
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
        "loss_physics": float(losses.get("physics", 0.0)),
        "loss_monotone": float(losses.get("monotone", 0.0)),
        "loss_bound": float(losses.get("bound", 0.0)),
        "violation_rate": float(violation_rate),
        "metrics": metrics,
        "curves": int(df["sample_id"].nunique()),
        "points": int(len(df)),
    }

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
                    f_pred[:, :ml],
                    f_true[:, :ml],
                    f_eq,
                    t_span[:, :ml],
                    k_j,
                    n_j,
                    self.model,
                    shared,
                    epoch
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
                log(f"[{self.stage_name}] skipped batch {step}/{total_batches}: {exc}")
                continue

            for key in totals:
                if key in loss_d:
                    totals[key] += float(loss_d[key].detach().item())
            nfe_total += float(self.model.ode_func.nfe)
            self.model.ode_func.nfe = 0
            n += 1

            if step == total_batches or step % 8 == 0:
                log(f"[{self.stage_name}] epoch {epoch:03d} batch {step:03d}/{total_batches:03d} avg_total={totals['total']/max(n,1):.5f} avg_data={totals['data']/max(n,1):.5f} avg_nfe={nfe_total/max(n,1):.1f} skipped={skipped}")

        return {k: v / max(n, 1) for k, v in totals.items()}, nfe_total / max(n, 1), skipped

    def train(self, train_loader, val_loader, verbose=True):
        log("=" * 60)
        log(f"Stage 2 FIXED on {self.device} | epochs={self.mc.max_epochs} | lr={self.mc.learning_rate}")
        log("=" * 60)
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
                "val_loss": float(val_loss.get("total", 0.0)),
                "val_rmse": float(metrics.get("rmse", 0.0)),
                "val_real_rmse": float(metrics.get("real_rmse", 0.0)),
                "val_endpoint_mae": float(metrics.get("endpoint_mae", 0.0)),
                "val_real_endpoint_mae": float(metrics.get("real_endpoint_mae", 0.0)),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "violations": float(viol),
                "nfe": float(nfe),
                "skipped_batches": int(skipped),
            }
            history.append(row)

            log(f"E {epoch:3d} | T:{row['train_loss']:.5f} V:{row['val_loss']:.5f} | RMSE:{row['val_rmse']:.5f} RealRMSE:{row['val_real_rmse']:.5f} | End:{row['val_endpoint_mae']:.5f} RealEnd:{row['val_real_endpoint_mae']:.5f} | NFE:{row['nfe']:.0f} | Skip:{row['skipped_batches']} | LR:{row['lr']:.1e}")

            if metric < best_metric:
                best_metric = metric
                patience = 0
                self.best_checkpoint_metric = metric
                self.best_val_metrics = metrics
                self.save_checkpoint("best")
            else:
                patience += 1
                if patience >= self.mc.early_stopping_patience:
                    log(f"Early stopping at epoch {epoch}")
                    break

        self.save_checkpoint("last")
        return history

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
log("KAGGLE STAGE-2 ONLY RETRY FIXED")
log("=" * 90)
log(f"Run version: {RUN_VERSION}")
log(f"UTC start: {START_UTC}")

precompute_thermo_tables(config, n_comp=16, n_temp=24)
full_df = build_full_dataset(config)
train_df, val_df, test_df = prepare_train_val_test_split(full_df, config)
train_real_df = sanitize_strict_time_curves(real_only(train_df))
val_real_df = sanitize_strict_time_curves(real_only(val_df))
test_real_df = sanitize_strict_time_curves(real_only(test_df))

log(f"train_real curves after sanitize: {train_real_df['sample_id'].nunique()} | points={len(train_real_df)}")
log(f"val_real curves after sanitize: {val_real_df['sample_id'].nunique()} | points={len(val_real_df)}")
log(f"test_real curves after sanitize: {test_real_df['sample_id'].nunique()} | points={len(test_real_df)}")

bundle_dir = Path("/kaggle/working/recovered_stage1_bundle")
stage1_best = bundle_dir / "stage1_best.pt"
assert stage1_best.exists(), f"Missing recovered checkpoint: {stage1_best}"
log(f"Recovered stage1 checkpoint: {stage1_best}")

stage2_config = deepcopy(config)
stage2_config.model.learning_rate = 8e-5
stage2_config.model.max_epochs = 60
stage2_config.model.early_stopping_patience = 10
stage2_config.model.batch_size = 1
stage2_config.model.scheduler_type = "cosine"
stage2_config.model.scheduler_eta_min = 1e-6
stage2_config.model.swa_start_epoch = stage2_config.model.max_epochs + 10
stage2_config.model.use_amp = False

stage2_model = PhysicsNODE(stage2_config.model)
ckpt = torch.load(stage1_best, map_location=stage2_config.device, weights_only=False)
stage2_model.load_state_dict(ckpt["model"])
for p in stage2_model.parameters():
    p.requires_grad_(True)

train_ds = AusteniteReversionDataset(train_real_df, stage2_config)
val_ds = AusteniteReversionDataset(val_real_df, stage2_config)
stage2_train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate_fn, num_workers=0)
stage2_val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate_fn, num_workers=0)

stage2_trainer = SafeLiveTrainer(stage2_model, stage2_config, use_gradnorm=False, use_homoscedastic=False, stage_name="stage2")
stage2_history = stage2_trainer.train(stage2_train_loader, stage2_val_loader, verbose=True)

stage2_best_path = stage2_config.checkpoint_dir / "physics_node_best.pt"
stage2_last_path = stage2_config.checkpoint_dir / "physics_node_last.pt"
if stage2_best_path.exists():
    shutil.copy2(stage2_best_path, ARTIFACT_DIR / "final_best_real.pt")
if stage2_last_path.exists():
    shutil.copy2(stage2_last_path, ARTIFACT_DIR / "final_last_real.pt")

pd.DataFrame(stage2_history).to_csv(ARTIFACT_DIR / "stage2_history.csv", index=False)

final_trainer = stage2_trainer
final_config = stage2_config
final_trainer.load_checkpoint("best")

eval_results = {
    "val_full": evaluate_split(final_trainer, val_df, "val_full", final_config),
    "test_full": evaluate_split(final_trainer, test_df, "test_full", final_config),
    "val_real": evaluate_split(final_trainer, val_real_df, "val_real", final_config),
    "test_real": evaluate_split(final_trainer, test_real_df, "test_real", final_config),
}

summary = {
    "started_utc": START_UTC,
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "runtime_minutes": round((time.time() - START_WALL) / 60.0, 2),
    "run_version": RUN_VERSION,
    "stage1_checkpoint": str(stage1_best),
    "device": str(final_config.device),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "dataset": {
        "full": describe_df("full", full_df),
        "train_real": describe_df("train_real", train_real_df),
        "val_real": describe_df("val_real", val_real_df),
        "test_real": describe_df("test_real", test_real_df),
    },
    "best_metric_name": final_trainer.best_metric_name,
    "best_metric_value": float(final_trainer.best_checkpoint_metric),
    "best_val_metrics": final_trainer.best_val_metrics,
    "evaluation": eval_results,
}
(ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

zip_path = shutil.make_archive("/kaggle/working/final_project_4_stage2_fixed_artifacts", "zip", ARTIFACT_DIR)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

log("=" * 90)
log("STAGE 2 FIXED RETRY COMPLETE")
log("=" * 90)
log(json.dumps(summary["evaluation"], indent=2))
log(f"Best checkpoint metric ({final_trainer.best_metric_name}): {final_trainer.best_checkpoint_metric:.6f}")
log(f"Artifact folder: {ARTIFACT_DIR}")
log(f"Artifact zip: {zip_path}")
