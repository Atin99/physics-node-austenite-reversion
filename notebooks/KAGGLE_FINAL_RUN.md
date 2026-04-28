# Kaggle Final Run

Use this with the `medium_mn_neural_ode` codebase, not `medium_mn_neural_ode_final`.

Why:

- this version already completed a stable Kaggle GPU run
- it has the weighted sampler and per-sample `t_span` handling
- it matches the artifacts you already inspected

## Notebook Setup

Create a brand-new Kaggle notebook with:

- Accelerator: `GPU`
- Preferred GPU: `T4 x1` or `P100`
- Internet: `On` only for the `torchdiffeq` install

## Input Setup

Best option:

1. Zip the folder `medium_mn_neural_ode` as `medium_mn_neural_ode.zip`
2. Create a private Kaggle dataset from that zip
3. Add that dataset as notebook input

Optional user CSVs:

- If you have user experimental CSVs, create a second Kaggle dataset whose files live under `user_experimental/`
- Add that dataset too

Do not run the old notebook cells. Use only the two cells below.

## Cell 1: Setup

```python
import os
import sys
import glob
import zipfile
import shutil
import subprocess
import importlib
from pathlib import Path

import torch

WORK_DIR = Path("/kaggle/working/medium_mn_neural_ode_final_run")
TMP_EXTRACT = Path("/kaggle/working/_src_extract")

def ensure_pkg(pkg_name: str):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])
        importlib.import_module(pkg_name)

ensure_pkg("torchdiffeq")

def find_repo_root_in_tree(root: Path) -> Path:
    hits = []
    for cfg in root.rglob("config.py"):
        repo = cfg.parent
        needed = ["trainer.py", "model.py", "data_generator.py", "thermodynamics.py"]
        if all((repo / name).exists() for name in needed):
            hits.append(repo)
    if not hits:
        raise FileNotFoundError("Could not find repo root containing config.py/trainer.py/model.py")
    hits = sorted(hits, key=lambda p: len(str(p)))
    return hits[0]

def locate_source():
    zip_candidates = sorted(glob.glob("/kaggle/input/**/*.zip", recursive=True))
    for path in zip_candidates:
        if "medium_mn_neural_ode" in os.path.basename(path).lower():
            return ("zip", Path(path))

    folder_candidates = sorted(glob.glob("/kaggle/input/**/config.py", recursive=True))
    for cfg in folder_candidates:
        repo = Path(cfg).parent
        needed = ["trainer.py", "model.py", "data_generator.py", "thermodynamics.py"]
        if all((repo / name).exists() for name in needed):
            return ("dir", repo)

    raise FileNotFoundError(
        "Source not found. Attach a dataset containing medium_mn_neural_ode.zip "
        "or the extracted medium_mn_neural_ode folder."
    )

kind, src = locate_source()
print("Source:", kind, src)

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
if TMP_EXTRACT.exists():
    shutil.rmtree(TMP_EXTRACT)

WORK_DIR.parent.mkdir(parents=True, exist_ok=True)

if kind == "zip":
    TMP_EXTRACT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(TMP_EXTRACT)
    repo_root = find_repo_root_in_tree(TMP_EXTRACT)
    shutil.copytree(repo_root, WORK_DIR)
else:
    shutil.copytree(src, WORK_DIR)

# Optional user CSVs from a second Kaggle dataset.
user_dir = WORK_DIR / "data" / "user_experimental"
user_dir.mkdir(parents=True, exist_ok=True)
user_csvs = sorted(glob.glob("/kaggle/input/**/user_experimental/*.csv", recursive=True))
for csv_path in user_csvs:
    dst = user_dir / Path(csv_path).name
    shutil.copy2(csv_path, dst)

os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

print("Work dir:", WORK_DIR)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("VRAM (GB):", round(props.total_memory / 1e9, 2))
print("User CSVs copied:", len(list(user_dir.glob("*.csv"))))
```

## Cell 2: Final Training Run

```python
import os
import sys
import json
import time
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Patch the hidden-state clamp before importing project modules.
# ------------------------------------------------------------------
model_py = Path("model.py")
model_text = model_py.read_text(encoding="utf-8")
old_line = "        return x\n"
new_line = "        return torch.clamp(x, min=-10.0, max=10.0)\n"
if new_line not in model_text:
    if old_line not in model_text:
        raise RuntimeError("Could not find the expected `return x` line in model.py")
    model_py.write_text(model_text.replace(old_line, new_line, 1), encoding="utf-8")
    print("Applied clamp patch to model.py")
else:
    print("Clamp patch already present")

for mod in ["config", "thermodynamics", "data_generator", "losses", "model", "trainer"]:
    if mod in sys.modules:
        del sys.modules[mod]

from config import get_config
from thermodynamics import precompute_thermo_tables
from data_generator import build_full_dataset, prepare_train_val_split
from trainer import Trainer, set_seed, create_data_loaders
from model import PhysicsNODE

ARTIFACT_DIR = Path("/kaggle/working/final_run_artifacts")
if ARTIFACT_DIR.exists():
    shutil.rmtree(ARTIFACT_DIR)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

start_wall = time.time()
start_utc = datetime.utcnow().isoformat() + "Z"

config = get_config()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Final run settings tuned for one reliable Kaggle pass.
config.model.rtol = 1e-2
config.model.atol = 1e-3
config.model.batch_size = 128
config.model.max_epochs = 150
config.model.early_stopping_patience = 150
config.model.swa_start_epoch = 80
config.model.swag_collect_freq = 5
config.model.scheduler_type = "cosine"
config.model.scheduler_eta_min = 1e-6
config.model.use_amp = False
config.model.accumulate_grad_batches = 1

set_seed(config.model.random_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

print("=" * 70)
print("FINAL KAGGLE RUN")
print("=" * 70)
print("Device:", config.device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Epochs:", config.model.max_epochs)
print("Batch size:", config.model.batch_size)
print("Solver tolerances:", config.model.rtol, config.model.atol)
print("Scheduler:", config.model.scheduler_type)
print("SWAG start:", config.model.swa_start_epoch)

precompute_thermo_tables(config, n_comp=20, n_temp=30)
df = build_full_dataset(config)
df_train, df_val = prepare_train_val_split(df, config)

config.synthetic_dir.mkdir(parents=True, exist_ok=True)
df_train.to_csv(config.synthetic_dir / "train.csv", index=False)
df_val.to_csv(config.synthetic_dir / "val.csv", index=False)

dataset_summary = {
    "total_points": int(len(df)),
    "total_curves": int(df["sample_id"].nunique()),
    "train_points": int(len(df_train)),
    "train_curves": int(df_train["sample_id"].nunique()),
    "val_points": int(len(df_val)),
    "val_curves": int(df_val["sample_id"].nunique()),
    "points_by_provenance": {str(k): int(v) for k, v in df["provenance"].value_counts().to_dict().items()},
}
print(json.dumps(dataset_summary, indent=2))

class FinalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history.setdefault("train_plain", [])
        self.history.setdefault("val_plain", [])
        self.best_val_plain = float("inf")

    @staticmethod
    def _plain_metric(loss_d):
        return float(
            loss_d["data"]
            + 0.1 * loss_d["physics"]
            + 0.5 * loss_d["monotone"]
            + 0.5 * loss_d["bound"]
        )

    def save_checkpoint(self, tag="best"):
        path = self.config.checkpoint_dir / f"physics_node_{tag}.pt"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val": self.best_val_loss,
            "best_val_plain": self.best_val_plain,
            "history": self.history,
            "config": {
                "hidden_dims": self.mc.hidden_dims,
                "latent_dim": self.mc.latent_dim,
                "augmented_dim": self.mc.augmented_dim,
                "rtol": self.mc.rtol,
                "atol": self.mc.atol,
                "batch_size": self.mc.batch_size,
                "max_epochs": self.mc.max_epochs,
                "scheduler_type": self.mc.scheduler_type,
                "swa_start_epoch": self.mc.swa_start_epoch,
            },
        }
        if self.swag and self.swag.n_collected > 0:
            state["swag_mean"] = self.swag.mean
            state["swag_sq_mean"] = self.swag.sq_mean
        torch.save(state, path)
        return path

    def train(self, train_loader, val_loader, verbose=True):
        print("=" * 70)
        print(f"Training on {self.device} | Epochs: {self.mc.max_epochs} | Batch: {self.mc.batch_size} | LR: {self.mc.learning_rate}")
        print(f"AMP: {self.mc.use_amp} | Adjoint: {self.mc.adjoint} | SWAG: {self.swag is not None}")
        print("=" * 70)
        t0 = time.time()

        for epoch in range(1, self.mc.max_epochs + 1):
            train_loss, nfe = self._train_epoch(train_loader, epoch)
            val_loss, viol = self._validate(val_loader)
            self.scheduler.step()

            train_plain = self._plain_metric(train_loss)
            val_plain = self._plain_metric(val_loss)

            self.history["train_loss"].append(train_loss["total"])
            self.history["val_loss"].append(val_loss["total"])
            self.history["train_data"].append(train_loss["data"])
            self.history["train_physics"].append(train_loss["physics"])
            self.history["train_mono"].append(train_loss["monotone"])
            self.history["train_bound"].append(train_loss["bound"])
            self.history["val_data"].append(val_loss["data"])
            self.history["val_physics"].append(val_loss["physics"])
            self.history["val_mono"].append(val_loss["monotone"])
            self.history["val_bound"].append(val_loss["bound"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.history["violations"].append(viol)
            self.history["nfe"].append(nfe)
            self.history["train_plain"].append(train_plain)
            self.history["val_plain"].append(val_plain)

            if self.swag and epoch >= self.mc.swa_start_epoch and epoch % self.mc.swag_collect_freq == 0:
                self.swag.collect()

            if val_plain < self.best_val_plain:
                self.best_val_plain = val_plain
                self.best_val_loss = float(val_loss["total"])
                self.patience_counter = 0
                self.save_checkpoint("best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.mc.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if verbose:
                print(
                    f"E {epoch:3d} | "
                    f"T_total:{train_loss['total']:.5f} "
                    f"V_total:{val_loss['total']:.5f} "
                    f"V_plain:{val_plain:.6f} | "
                    f"NFE:{nfe:.0f} | Viol:{viol:.2%} | "
                    f"LR:{self.optimizer.param_groups[0]['lr']:.1e}"
                )

        print(f"Done in {(time.time() - t0) / 60:.1f}m | Best plain val: {self.best_val_plain:.6f}")
        self.save_checkpoint("last")
        return self.history

model = PhysicsNODE(config.model)
tr_loader, va_loader = create_data_loaders(df_train, df_val, config)
trainer = FinalTrainer(model, config)
history = trainer.train(tr_loader, va_loader, verbose=True)

ckpt_dir = config.checkpoint_dir
best_plain_src = ckpt_dir / "physics_node_best.pt"
last_src = ckpt_dir / "physics_node_last.pt"
best_plain_dst = ckpt_dir / "physics_node_best_plain.pt"
last_swag_dst = ckpt_dir / f"physics_node_{len(history['val_plain'])}ep_swag.pt"

if best_plain_src.exists():
    shutil.copy2(best_plain_src, best_plain_dst)
if last_src.exists():
    shutil.copy2(last_src, last_swag_dst)

history_df = pd.DataFrame(history)
history_df.index = np.arange(1, len(history_df) + 1)
history_df.index.name = "epoch"
history_csv = ARTIFACT_DIR / "training_history.csv"
history_df.to_csv(history_csv)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = history_df.index.values
axes[0].plot(epochs, history_df["train_loss"], label="train_total")
axes[0].plot(epochs, history_df["val_loss"], label="val_total")
axes[0].set_title("Weighted Total Objective")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs, history_df["train_plain"], label="train_plain")
axes[1].plot(epochs, history_df["val_plain"], label="val_plain")
axes[1].set_title("Plain Validation Metric")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plot_path = ARTIFACT_DIR / "training_curves.png"
plt.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.close(fig)

summary = {
    "started_utc": start_utc,
    "finished_utc": datetime.utcnow().isoformat() + "Z",
    "runtime_minutes": round((time.time() - start_wall) / 60, 2),
    "device": str(config.device),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "epochs_completed": int(len(history["val_plain"])),
    "selection_metric": "val_plain = val_data + 0.1*val_physics + 0.5*val_mono + 0.5*val_bound",
    "best_val_plain": float(trainer.best_val_plain),
    "best_val_total_at_best_plain": float(trainer.best_val_loss),
    "last_val_plain": float(history["val_plain"][-1]),
    "last_val_total": float(history["val_loss"][-1]),
    "last_lr": float(history["lr"][-1]),
    "final_violations": float(history["violations"][-1]),
    "swag_collected": int(trainer.swag.n_collected if trainer.swag else 0),
    "dataset": dataset_summary,
    "config": {
        "batch_size": int(config.model.batch_size),
        "max_epochs": int(config.model.max_epochs),
        "rtol": float(config.model.rtol),
        "atol": float(config.model.atol),
        "scheduler_type": config.model.scheduler_type,
        "scheduler_eta_min": float(config.model.scheduler_eta_min),
        "swa_start_epoch": int(config.model.swa_start_epoch),
        "swag_collect_freq": int(config.model.swag_collect_freq),
        "adjoint": bool(config.model.adjoint),
        "use_amp": bool(config.model.use_amp),
    },
    "artifacts": {
        "best_plain_checkpoint": str(best_plain_dst) if best_plain_dst.exists() else None,
        "final_swag_checkpoint": str(last_swag_dst) if last_swag_dst.exists() else None,
        "history_csv": str(history_csv),
        "training_plot": str(plot_path),
        "train_csv": str(config.synthetic_dir / "train.csv"),
        "val_csv": str(config.synthetic_dir / "val.csv"),
    },
}

summary_path = ARTIFACT_DIR / "run_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

for path in [best_plain_dst, last_swag_dst, config.synthetic_dir / "train.csv", config.synthetic_dir / "val.csv"]:
    if path and Path(path).exists():
        shutil.copy2(path, ARTIFACT_DIR / Path(path).name)

zip_base = "/kaggle/working/final_run_artifacts"
zip_path = shutil.make_archive(zip_base, "zip", ARTIFACT_DIR)

print("\n" + "=" * 70)
print("FINAL RUN COMPLETE")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("Artifact folder:", ARTIFACT_DIR)
print("Artifact zip:", zip_path)
```

## What You Should Expect

- about 3.2 to 3.6 hours on a Kaggle T4 based on the earlier 150-epoch run
- a `physics_node_best_plain.pt` checkpoint chosen by a fixed plain validation metric
- a `physics_node_150ep_swag.pt` style checkpoint copied from the final epoch with SWAG statistics
- a zipped artifact bundle at `/kaggle/working/final_run_artifacts.zip`

## What To Keep For The Report

- `run_summary.json`
- `training_history.csv`
- `training_curves.png`
- `physics_node_best_plain.pt`
- `physics_node_150ep_swag.pt` or the final-epoch SWAG checkpoint name generated in the run
