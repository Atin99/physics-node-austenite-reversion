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

WORK_DIR = Path("/kaggle/working/final_project_4_recover_run")
TMP_EXTRACT = Path("/kaggle/working/final_project_4_recover_extract")
ARTIFACT_DIR = Path("/kaggle/working/final_project_4_recover_artifacts")
RECOVER_EXTRACT = Path("/kaggle/working/final_project_4_recovered_input")
REQUIRED_FILES = ["config.py", "data_generator.py", "losses.py", "model.py", "thermodynamics.py", "trainer.py"]
START_UTC = datetime.now(timezone.utc).isoformat()
START_WALL = time.time()
RUN_VERSION = "2026-04-28-recover-stage1"


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
    raise FileNotFoundError("Could not find the source project input.")


def locate_recovered_input():
    direct_hits = [Path(p).parent for p in glob.glob("/kaggle/input/**/stage1_best.pt", recursive=True)]
    direct_hits += [Path(p).parent for p in glob.glob("/kaggle/working/**/stage1_best.pt", recursive=True)]
    if direct_hits:
        return ("dir", sorted(set(direct_hits), key=lambda p: len(str(p)))[0])

    zip_hits = [Path(p) for p in glob.glob("/kaggle/input/**/*.zip", recursive=True) if "stage1" in os.path.basename(p).lower() or "recover" in os.path.basename(p).lower()]
    if zip_hits:
        return ("zip", sorted(zip_hits, key=lambda p: len(str(p)))[0])

    raise FileNotFoundError("Could not find recovered stage1 input. Add the recovered bundle zip or extracted files as a Kaggle input.")


def prepare_workspace():
    kind, src = locate_source()
    log(f"Source: {kind} -> {src}")

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    if TMP_EXTRACT.exists():
        shutil.rmtree(TMP_EXTRACT)
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if kind == "zip":
        TMP_EXTRACT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(TMP_EXTRACT)
        repo_root = find_repo_roots(TMP_EXTRACT)[0]
    else:
        repo_root = src if is_repo_root(src) else find_repo_roots(src)[0]

    shutil.copytree(repo_root, WORK_DIR)

    os.chdir(WORK_DIR)
    sys.path.insert(0, str(WORK_DIR))


def prepare_recovered_dir():
    kind, src = locate_recovered_input()
    log(f"Recovered input: {kind} -> {src}")
    if RECOVER_EXTRACT.exists():
        shutil.rmtree(RECOVER_EXTRACT)
    RECOVER_EXTRACT.mkdir(parents=True, exist_ok=True)

    if kind == "dir":
        for file in src.iterdir():
            if file.is_file():
                shutil.copy2(file, RECOVER_EXTRACT / file.name)
    else:
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(RECOVER_EXTRACT)

    best = next(RECOVER_EXTRACT.rglob("stage1_best.pt"), None)
    last = next(RECOVER_EXTRACT.rglob("stage1_last.pt"), None)
    if best is None:
        raise FileNotFoundError("Recovered stage1_best.pt not found after extraction.")
    return best, last


prepare_workspace()
stage1_best_src, stage1_last_src = prepare_recovered_dir()

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


def evaluate_split(trainer: Trainer, df: pd.DataFrame, name: str, config):
    if len(df) == 0:
        return {"name": name, "empty": True}
    loader = make_eval_loader(df, config, batch_size=min(config.model.batch_size, 64))
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
log("KAGGLE STAGE-1 RECOVERY RUN")
log("=" * 90)
log(f"Run version: {RUN_VERSION}")
log(f"UTC start: {START_UTC}")
log(f"Recovered checkpoint: {stage1_best_src}")

log("\n[1/4] Precomputing thermodynamic tables...")
precompute_thermo_tables(config, n_comp=16, n_temp=24)

log("\n[2/4] Building dataset...")
full_df = build_full_dataset(config)
train_df, val_df, test_df = prepare_train_val_test_split(full_df, config)
train_real_df = real_only(train_df)
val_real_df = real_only(val_df)
test_real_df = real_only(test_df)

log("\n[3/4] Loading recovered stage-1 checkpoint...")
config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(stage1_best_src, config.checkpoint_dir / "physics_node_best.pt")
if stage1_last_src is not None:
    shutil.copy2(stage1_last_src, config.checkpoint_dir / "physics_node_last.pt")

model = PhysicsNODE(config.model)
trainer = Trainer(model, config, use_gradnorm=False, use_homoscedastic=False)
trainer.load_checkpoint("best")

log("\n[4/4] Evaluating and writing artifacts...")
eval_results = {
    "val_full": evaluate_split(trainer, val_df, "val_full", config),
    "test_full": evaluate_split(trainer, test_df, "test_full", config),
    "val_real": evaluate_split(trainer, val_real_df, "val_real", config),
    "test_real": evaluate_split(trainer, test_real_df, "test_real", config),
}

for file_name in ["stage1_best.pt", "stage1_last.pt", "stage1_history.csv", "stage1_history.png"]:
    src = next(RECOVER_EXTRACT.rglob(file_name), None)
    if src is not None:
        shutil.copy2(src, ARTIFACT_DIR / file_name)

summary = {
    "started_utc": START_UTC,
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "runtime_minutes": round((time.time() - START_WALL) / 60.0, 2),
    "run_version": RUN_VERSION,
    "recovered_checkpoint": str(stage1_best_src),
    "device": str(config.device),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "dataset": {
        "full": describe_df("full", full_df),
        "train": describe_df("train", train_df),
        "val": describe_df("val", val_df),
        "test": describe_df("test", test_df),
        "train_real": describe_df("train_real", train_real_df),
        "val_real": describe_df("val_real", val_real_df),
        "test_real": describe_df("test_real", test_real_df),
    },
    "evaluation": eval_results,
}

(ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
zip_path = shutil.make_archive("/kaggle/working/final_project_4_recover_artifacts", "zip", ARTIFACT_DIR)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

log("\n" + "=" * 90)
log("RECOVERY COMPLETE")
log("=" * 90)
log(json.dumps(summary["evaluation"], indent=2))
log(f"Artifact folder: {ARTIFACT_DIR}")
log(f"Artifact zip: {zip_path}")
