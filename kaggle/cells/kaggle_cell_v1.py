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


def log(message=""):
    print(message, flush=True)


def ensure_pkg(pkg_name: str):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])


ensure_pkg("torchdiffeq")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

WORK_DIR = Path("/kaggle/working/final_project_4_run")
TMP_EXTRACT = Path("/kaggle/working/final_project_4_extract")
ARTIFACT_DIR = Path("/kaggle/working/final_project_4_artifacts")
REQUIRED_FILES = ["config.py", "data_generator.py", "losses.py", "model.py", "thermodynamics.py", "trainer.py"]
START_UTC = datetime.now(timezone.utc).isoformat()
START_WALL = time.time()
RUN_VERSION = "2026-04-27-fix3"


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
        raise FileNotFoundError("Could not find a valid project root containing config.py/trainer.py/model.py.")
    return sorted(set(hits), key=score_repo, reverse=True)


def locate_source():
    zip_candidates = sorted(glob.glob("/kaggle/input/**/*.zip", recursive=True))
    dir_candidates = []
    for cfg_path in glob.glob("/kaggle/input/**/config.py", recursive=True):
        repo = Path(cfg_path).parent
        if is_repo_root(repo):
            dir_candidates.append(repo)
    dir_candidates = sorted(set(dir_candidates), key=score_repo, reverse=True)

    ranked_zips = []
    for zip_path in zip_candidates:
        lower = os.path.basename(zip_path).lower()
        score = 0
        if "before_kaggle" in lower:
            score += 100
        if "final_project_4" in lower:
            score += 40
        if "medium_mn_neural_ode" in lower:
            score += 20
        if "with_kaggle" in lower:
            score -= 50
        ranked_zips.append((score, zip_path))
    ranked_zips.sort(reverse=True)

    if ranked_zips and ranked_zips[0][0] >= (score_repo(dir_candidates[0])[0] if dir_candidates else -999):
        return ("zip", Path(ranked_zips[0][1]))
    if dir_candidates:
        return ("dir", Path(dir_candidates[0]))
    raise FileNotFoundError(
        "Attach a Kaggle dataset containing either the clean `before_kaggle` project folder "
        "or a zip that contains it."
    )


def prepare_workspace():
    kind, src = locate_source()
    log(f"Source: {kind} -> {src}")

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    if TMP_EXTRACT.exists():
        shutil.rmtree(TMP_EXTRACT)
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)

    WORK_DIR.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if kind == "zip":
        TMP_EXTRACT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(TMP_EXTRACT)
        repo_root = find_repo_roots(TMP_EXTRACT)[0]
    else:
        repo_root = src
        if not is_repo_root(repo_root):
            repo_root = find_repo_roots(repo_root)[0]

    shutil.copytree(repo_root, WORK_DIR)

    user_dir = WORK_DIR / "data" / "user_experimental"
    user_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for csv_path in sorted(glob.glob("/kaggle/input/**/user_experimental/*.csv", recursive=True)):
        dst = user_dir / Path(csv_path).name
        shutil.copy2(csv_path, dst)
        copied += 1

    os.chdir(WORK_DIR)
    sys.path.insert(0, str(WORK_DIR))
    return copied


user_csv_count = prepare_workspace()
for mod_name in ["config", "thermodynamics", "data_generator", "losses", "model", "trainer", "publication_pipeline", "visualizations"]:
    if mod_name in sys.modules:
        del sys.modules[mod_name]

from config import get_config
from data_generator import build_full_dataset, prepare_train_val_test_split
from model import PhysicsNODE
from thermodynamics import precompute_thermo_tables
from trainer import AusteniteReversionDataset, Trainer, _collate_fn, create_data_loaders, set_seed


def describe_df(name: str, df: pd.DataFrame) -> dict:
    curves = int(df["sample_id"].nunique()) if len(df) else 0
    points = int(len(df))
    out = {"name": name, "curves": curves, "points": points}
    if "provenance" in df.columns and len(df):
        out["points_by_provenance"] = {str(k): int(v) for k, v in df["provenance"].value_counts().to_dict().items()}
        out["curves_by_provenance"] = {
            str(k): int(v)
            for k, v in df.groupby("provenance")["sample_id"].nunique().to_dict().items()
        }
    return out


def real_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["provenance"].isin(["experimental", "user_provided"])].copy()


def make_eval_loader(df: pd.DataFrame, config, batch_size=None):
    ds = AusteniteReversionDataset(df, config)
    return DataLoader(
        ds,
        batch_size=batch_size or config.model.batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
        pin_memory=config.device.type == "cuda",
    )


def assert_strict_time_order(df: pd.DataFrame, name: str):
    bad = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        times = group.sort_values("t_seconds")["t_seconds"].to_numpy(dtype=float)
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            bad.append({
                "sample_id": int(sample_id),
                "times": times[:10].tolist(),
            })
            if len(bad) >= 5:
                break
    if bad:
        raise RuntimeError(
            f"{name} contains non-strict time grids. Refusing to train. Examples: {json.dumps(bad, indent=2)}"
        )


def assert_expected_counts(full_df: pd.DataFrame, user_csv_count: int):
    if user_csv_count != 0:
        return
    exp_points = int((full_df["provenance"] == "experimental").sum())
    total_points = int(len(full_df))
    if exp_points != 227 or total_points != 57227:
        raise RuntimeError(
            "Dataset signature mismatch. Expected 227 experimental points and 57227 total points, "
            f"got experimental={exp_points}, total={total_points}. This usually means Kaggle is still using the old dataset version."
        )


class LiveTrainer(Trainer):
    def __init__(self, *args, stage_name="stage", heartbeat_batches=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_name = stage_name
        self.heartbeat_batches = max(int(heartbeat_batches), 1)

    def _train_epoch(self, loader, epoch):
        self.model.train()
        totals = {'total': 0.0, 'data': 0.0, 'physics': 0.0, 'monotone': 0.0, 'bound': 0.0}
        nfe_total, n, skipped = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        accum_steps = 0
        shared_layer = self._get_shared_layer()
        epoch_t0 = time.time()
        total_batches = len(loader)

        for step, batch in enumerate(loader, start=1):
            static = batch['static'].to(self.device)
            f_true = batch['traj'].to(self.device)
            t_span = batch['t_span'].to(self.device)
            point_mask = batch['point_mask'].to(self.device)
            obs_mask = batch['obs_mask'].to(self.device)
            lengths = batch['lengths']
            f_eq = batch['f_eq'].to(self.device)
            dG = batch['dG_norm'].to(self.device)
            k_j = batch['k_jmak'].to(self.device)
            n_j = batch['n_jmak'].to(self.device)

            with autocast('cuda', enabled=self.mc.use_amp and torch.cuda.is_available()):
                try:
                    f_pred = self.model(static, f_eq, dG, t_span, lengths=lengths)
                    ml = min(f_pred.shape[1], f_true.shape[1])
                    loss_d = self.criterion(
                        f_pred[:, :ml],
                        f_true[:, :ml],
                        f_eq,
                        t_span[:, :ml],
                        k_j,
                        n_j,
                        self.model,
                        shared_layer,
                        epoch,
                        point_mask=point_mask[:, :ml],
                        obs_mask=obs_mask[:, :ml],
                    )
                    provenance = batch.get('provenance', ['unknown'] * static.shape[0])
                    real_mask = torch.tensor(
                        [p in ('experimental', 'user_provided') for p in provenance],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if real_mask.sum() > 0 and getattr(self.config.data, 'provenance_aware_loss', False):
                        weight = 1.0 + (self.config.data.real_data_weight - 1.0) * real_mask.mean()
                        loss_d['total'] = loss_d['total'] * weight
                except Exception as exc:
                    skipped += 1
                    log(f"[{self.stage_name}] skipped batch {step}/{total_batches}: {exc}")
                    continue

            self.scaler.scale(loss_d['total']).backward()
            accum_steps += 1
            if accum_steps >= self.mc.accumulate_grad_batches:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.mc.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accum_steps = 0

            for key in totals:
                if key in loss_d:
                    totals[key] += float(loss_d[key].detach().item())
            nfe_total += float(self.model.ode_func.nfe)
            self.model.ode_func.nfe = 0
            n += 1

            if step % self.heartbeat_batches == 0 or step == total_batches:
                elapsed_min = (time.time() - epoch_t0) / 60.0
                log(
                    f"[{self.stage_name}] epoch {epoch:03d} batch {step:03d}/{total_batches:03d} "
                    f"avg_total={totals['total']/max(n,1):.5f} avg_data={totals['data']/max(n,1):.5f} "
                    f"avg_nfe={nfe_total/max(n,1):.1f} skipped={skipped} elapsed={elapsed_min:.1f}m"
                )

        if accum_steps > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.mc.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return {key: value / max(n, 1) for key, value in totals.items()}, nfe_total / max(n, 1), skipped


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)


def save_history(history: dict, path: Path):
    pd.DataFrame(history).to_csv(path, index_label="epoch")


def plot_history(history: dict, path: Path, title: str):
    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = np.arange(1, len(history_df) + 1)
    axes[0].plot(epochs, history_df["train_loss"], label="train_total")
    axes[0].plot(epochs, history_df["val_loss"], label="val_total")
    axes[0].set_title("Weighted objective")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history_df["val_rmse"], label="val_rmse")
    if "val_real_rmse" in history_df:
        axes[1].plot(epochs, history_df["val_real_rmse"], label="val_real_rmse")
    axes[1].set_title("Observed-point RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history_df["val_endpoint_mae"], label="val_endpoint_mae")
    if "val_real_endpoint_mae" in history_df:
        axes[2].plot(epochs, history_df["val_real_endpoint_mae"], label="val_real_endpoint_mae")
    axes[2].set_title("Endpoint MAE")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(trainer: Trainer, df: pd.DataFrame, name: str, config):
    if len(df) == 0:
        return {"name": name, "empty": True}
    loader = make_eval_loader(df, config, batch_size=min(config.model.batch_size, 64))
    losses, violation_rate, metrics = trainer._validate(loader)
    result = {
        "name": name,
        "loss_total": float(losses["total"]),
        "loss_data": float(losses["data"]),
        "loss_physics": float(losses["physics"]),
        "loss_monotone": float(losses["monotone"]),
        "loss_bound": float(losses["bound"]),
        "violation_rate": float(violation_rate),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "real_mae": float(metrics["real_mae"]),
        "real_rmse": float(metrics["real_rmse"]),
        "endpoint_mae": float(metrics["endpoint_mae"]),
        "real_endpoint_mae": float(metrics["real_endpoint_mae"]),
        "n_observed": int(metrics["n_observed"]),
        "n_real_observed": int(metrics["n_real_observed"]),
        "curves": int(df["sample_id"].nunique()),
        "points": int(len(df)),
    }
    log(
        f"[eval] {name}: rmse={result['rmse']:.5f} real_rmse={result['real_rmse']:.5f} "
        f"endpoint_mae={result['endpoint_mae']:.5f} real_endpoint_mae={result['real_endpoint_mae']:.5f}"
    )
    return result


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
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

log("=" * 90)
log("KAGGLE GPU TRAINING RUN")
log("=" * 90)
log(f"Run version: {RUN_VERSION}")
log(f"UTC start: {START_UTC}")
log(f"Work dir: {WORK_DIR}")
log(f"User CSVs copied: {user_csv_count}")
log(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory / 1e9:.2f} GB")
log(
    f"Training config: batch={config.model.batch_size} epochs={config.model.max_epochs} "
    f"adjoint={config.model.adjoint} rtol={config.model.rtol} atol={config.model.atol} "
    f"max_steps={config.model.max_num_steps}"
)

log("\n[1/6] Precomputing thermodynamic tables...")
precompute_thermo_tables(config, n_comp=16, n_temp=24)

log("\n[2/6] Building dataset...")
full_df = build_full_dataset(config)
train_df, val_df, test_df = prepare_train_val_test_split(full_df, config)
train_real_df = real_only(train_df)
val_real_df = real_only(val_df)
test_real_df = real_only(test_df)

assert_expected_counts(full_df, user_csv_count)
assert_strict_time_order(full_df, "full_df")
assert_strict_time_order(train_df, "train_df")
assert_strict_time_order(val_df, "val_df")
assert_strict_time_order(test_df, "test_df")

for name, df in [("full", full_df), ("train", train_df), ("val", val_df), ("test", test_df), ("train_real", train_real_df), ("val_real", val_real_df), ("test_real", test_real_df)]:
    summary = describe_df(name, df)
    log(json.dumps(summary, indent=2))

config.synthetic_dir.mkdir(parents=True, exist_ok=True)
full_df.to_csv(config.synthetic_dir / "full_dataset.csv", index=False)
train_df.to_csv(config.synthetic_dir / "train.csv", index=False)
val_df.to_csv(config.synthetic_dir / "val.csv", index=False)
test_df.to_csv(config.synthetic_dir / "test.csv", index=False)
train_real_df.to_csv(config.synthetic_dir / "train_real.csv", index=False)
val_real_df.to_csv(config.synthetic_dir / "val_real.csv", index=False)
test_real_df.to_csv(config.synthetic_dir / "test_real.csv", index=False)

log("\n[3/6] Stage 1 training: full dataset with honest validation metrics...")
stage1_train_loader, stage1_val_loader = create_data_loaders(train_df, val_df, config)
stage1_model = PhysicsNODE(config.model)
stage1_trainer = LiveTrainer(stage1_model, config, stage_name="stage1")
stage1_history = stage1_trainer.train(stage1_train_loader, stage1_val_loader, verbose=True)
stage1_best_path = config.checkpoint_dir / "physics_node_best.pt"
stage1_last_path = config.checkpoint_dir / "physics_node_last.pt"
copy_if_exists(stage1_best_path, ARTIFACT_DIR / "stage1_best.pt")
copy_if_exists(stage1_last_path, ARTIFACT_DIR / "stage1_last.pt")
save_history(stage1_history, ARTIFACT_DIR / "stage1_history.csv")
plot_history(stage1_history, ARTIFACT_DIR / "stage1_history.png", "Stage 1")

final_model = stage1_model
final_trainer = stage1_trainer
final_config = config
final_stage_name = "stage1"

can_run_stage2 = train_real_df["sample_id"].nunique() >= 20 and val_real_df["sample_id"].nunique() >= 5
if can_run_stage2:
    log("\n[4/6] Stage 2 fine-tuning: real data only...")
    stage2_config = deepcopy(config)
    stage2_config.model.learning_rate = 8e-5
    stage2_config.model.max_epochs = 60
    stage2_config.model.early_stopping_patience = 15
    stage2_config.model.batch_size = min(32, config.model.batch_size)
    stage2_config.model.scheduler_type = "cosine"
    stage2_config.model.scheduler_eta_min = 1e-6
    stage2_config.model.swa_start_epoch = stage2_config.model.max_epochs + 10

    stage2_model = PhysicsNODE(stage2_config.model)
    stage2_model.load_state_dict(torch.load(stage1_best_path, map_location=stage2_config.device, weights_only=False)["model"])
    stage2_train_loader, stage2_val_loader = create_data_loaders(train_real_df, val_real_df, stage2_config)
    stage2_trainer = LiveTrainer(stage2_model, stage2_config, stage_name="stage2")
    stage2_history = stage2_trainer.train(stage2_train_loader, stage2_val_loader, verbose=True)
    stage2_best_path = stage2_config.checkpoint_dir / "physics_node_best.pt"
    stage2_last_path = stage2_config.checkpoint_dir / "physics_node_last.pt"
    copy_if_exists(stage2_best_path, ARTIFACT_DIR / "final_best_real.pt")
    copy_if_exists(stage2_last_path, ARTIFACT_DIR / "final_last_real.pt")
    save_history(stage2_history, ARTIFACT_DIR / "stage2_history.csv")
    plot_history(stage2_history, ARTIFACT_DIR / "stage2_history.png", "Stage 2")

    final_model = stage2_model
    final_trainer = stage2_trainer
    final_config = stage2_config
    final_stage_name = "stage2"
else:
    log("\n[4/6] Stage 2 fine-tuning skipped: not enough held-out real curves for a meaningful real-only pass.")

log("\n[5/6] Loading best checkpoint and running held-out evaluation...")
final_trainer.load_checkpoint("best")
eval_results = {
    "val_full": evaluate_split(final_trainer, val_df, "val_full", final_config),
    "test_full": evaluate_split(final_trainer, test_df, "test_full", final_config),
    "val_real": evaluate_split(final_trainer, val_real_df, "val_real", final_config),
    "test_real": evaluate_split(final_trainer, test_real_df, "test_real", final_config),
}

log("\n[6/6] Writing artifacts...")
summary = {
    "started_utc": START_UTC,
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "runtime_minutes": round((time.time() - START_WALL) / 60.0, 2),
    "work_dir": str(WORK_DIR),
    "artifact_dir": str(ARTIFACT_DIR),
    "final_stage": final_stage_name,
    "device": str(final_config.device),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "user_csv_count": int(user_csv_count),
    "dataset": {
        "full": describe_df("full", full_df),
        "train": describe_df("train", train_df),
        "val": describe_df("val", val_df),
        "test": describe_df("test", test_df),
        "train_real": describe_df("train_real", train_real_df),
        "val_real": describe_df("val_real", val_real_df),
        "test_real": describe_df("test_real", test_real_df),
    },
    "config": {
        "batch_size": int(final_config.model.batch_size),
        "max_epochs": int(final_config.model.max_epochs),
        "learning_rate": float(final_config.model.learning_rate),
        "rtol": float(final_config.model.rtol),
        "atol": float(final_config.model.atol),
        "adjoint": bool(final_config.model.adjoint),
        "use_amp": bool(final_config.model.use_amp),
        "checkpoint_metric": final_config.model.checkpoint_metric,
        "synthetic_calibration_samples": int(config.data.synthetic_calibration_samples),
        "synthetic_exploration_samples": int(config.data.synthetic_exploration_samples),
        "real_data_weight": float(config.data.real_data_weight),
    },
    "best_metric_name": final_trainer.best_metric_name,
    "best_metric_value": float(final_trainer.best_checkpoint_metric),
    "best_val_metrics": final_trainer.best_val_metrics,
    "evaluation": eval_results,
}
summary_path = ARTIFACT_DIR / "run_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
copy_if_exists(final_config.checkpoint_dir / "physics_node_best.pt", ARTIFACT_DIR / f"{final_stage_name}_best_checkpoint.pt")
copy_if_exists(final_config.checkpoint_dir / "physics_node_last.pt", ARTIFACT_DIR / f"{final_stage_name}_last_checkpoint.pt")

zip_path = shutil.make_archive("/kaggle/working/final_project_4_artifacts", "zip", ARTIFACT_DIR)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

log("\n" + "=" * 90)
log("RUN COMPLETE")
log("=" * 90)
log(json.dumps(summary["evaluation"], indent=2))
log(f"Best checkpoint metric ({final_trainer.best_metric_name}): {final_trainer.best_checkpoint_metric:.6f}")
log(f"Artifact folder: {ARTIFACT_DIR}")
log(f"Artifact zip: {zip_path}")
