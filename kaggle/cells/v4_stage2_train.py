"""
STAGE 2 FINE-TUNING CELL — Paste this as a SINGLE cell in a Kaggle GPU notebook.
Upload kaggle_stage2_upload.zip as a Kaggle dataset first.
"""
import gc, glob, importlib, json, math, os, shutil, subprocess, sys, time, zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

def log(msg=""): print(msg, flush=True)
def ensure_pkg(p):
    try: importlib.import_module(p)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

ensure_pkg("torchdiffeq")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

# ── paths ──
WORK   = Path("/kaggle/working/run")
ARTS   = Path("/kaggle/working/artifacts")
REQUIRED = ["config.py","data_generator.py","losses.py","model.py","thermodynamics.py","trainer.py"]
T0     = time.time()
UTC0   = datetime.now(timezone.utc).isoformat()

def is_repo(p): return p.is_dir() and all((p/n).exists() for n in REQUIRED)

def find_repo(root):
    for c in root.rglob("config.py"):
        if is_repo(c.parent): return c.parent
    return None

def find_checkpoint(root):
    for pat in ["**/stage1_best_ep109.pt","**/stage1_best.pt","**/physics_node_best.pt"]:
        hits = sorted(root.rglob(pat.split("/")[-1]))
        if hits: return hits[0]
    return None

# ── workspace setup ──
for d in [WORK, ARTS]: shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)

inp = Path("/kaggle/input")
# try zips first
repo_root = ckpt_path = None
for zp in sorted(inp.rglob("*.zip")):
    tmp = Path("/kaggle/working/_ext"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(zp,"r") as zf: zf.extractall(tmp)
    r = find_repo(tmp)
    if r and not repo_root: repo_root = r
    c = find_checkpoint(tmp)
    if c and not ckpt_path: ckpt_path = c

if not repo_root:
    r = find_repo(inp)
    if r: repo_root = r
if not ckpt_path:
    c = find_checkpoint(inp)
    if c: ckpt_path = c
if not ckpt_path:
    c = find_checkpoint(Path("/kaggle/working"))
    if c: ckpt_path = c

assert repo_root, "Could not find project source with config.py etc. Check your upload."
assert ckpt_path, "Could not find stage1 checkpoint .pt file. Check your upload."
log(f"Source: {repo_root}\nCheckpoint: {ckpt_path}")

shutil.copytree(repo_root, WORK, dirs_exist_ok=True)
shutil.copy2(ckpt_path, WORK / "stage1_best.pt")
os.chdir(WORK); sys.path.insert(0, str(WORK))

for m in list(sys.modules):
    if m in ["config","thermodynamics","data_generator","losses","model","trainer","features",
             "real_data","publication_pipeline","visualizations"]:
        del sys.modules[m]

from config import get_config
from data_generator import build_full_dataset, prepare_train_val_test_split
from model import PhysicsNODE
from thermodynamics import precompute_thermo_tables
from trainer import AusteniteReversionDataset, Trainer, _collate_fn, create_data_loaders, set_seed

# ── helpers ──
def desc(name, df):
    c = int(df["sample_id"].nunique()) if len(df) else 0
    o = {"name":name,"curves":c,"points":len(df)}
    if "provenance" in df.columns and len(df):
        o["by_prov"] = {str(k):int(v) for k,v in df["provenance"].value_counts().items()}
    return o

def real_only(df): return df[df["provenance"].isin(["experimental","user_provided"])].copy()

def make_loader(df, cfg, bs=None):
    ds = AusteniteReversionDataset(df, cfg)
    return DataLoader(ds, batch_size=bs or cfg.model.batch_size, shuffle=False,
                      collate_fn=_collate_fn, num_workers=0, pin_memory=cfg.device.type=="cuda")

def evaluate(trainer, df, name, cfg):
    if len(df)==0: return {"name":name,"empty":True}
    loader = make_loader(df, cfg, bs=min(cfg.model.batch_size,64))
    losses, viol, metrics = trainer._validate(loader)
    r = {"name":name, "loss_total":float(losses["total"]),
         "rmse":float(metrics["rmse"]), "real_rmse":float(metrics["real_rmse"]),
         "mae":float(metrics["mae"]), "real_mae":float(metrics["real_mae"]),
         "endpoint_mae":float(metrics["endpoint_mae"]),
         "real_endpoint_mae":float(metrics["real_endpoint_mae"]),
         "n_obs":int(metrics["n_observed"]), "n_real_obs":int(metrics["n_real_observed"]),
         "curves":int(df["sample_id"].nunique()), "points":len(df)}
    log(f"  [{name}] rmse={r['rmse']:.5f} real_rmse={r['real_rmse']:.5f} "
        f"endpoint={r['endpoint_mae']:.5f} real_endpoint={r['real_endpoint_mae']:.5f}")
    return r

# ── config ──
cfg = get_config()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.model.use_homoscedastic = False
cfg.model.use_gradnorm = False
cfg.model.checkpoint_metric = "val_real_rmse"
cfg.model.batch_size = 24
cfg.model.max_epochs = 120  # for stage1 dataset building consistency
cfg.model.early_stopping_patience = 24
cfg.model.learning_rate = 2.5e-4
cfg.model.scheduler_type = "cosine"
cfg.model.scheduler_eta_min = 5e-6
cfg.model.use_amp = False
cfg.model.adjoint = False
cfg.model.rtol = 5e-3
cfg.model.atol = 1e-4
cfg.model.max_num_steps = 512
cfg.model.swa_start_epoch = 9999
cfg.model.gradient_clip_val = 1.0
cfg.data.synthetic_calibration_samples = 250
cfg.data.synthetic_exploration_samples = 700
cfg.data.real_data_weight = 5.0
cfg.data.provenance_aware_loss = True

set_seed(cfg.model.random_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch,"set_float32_matmul_precision"): torch.set_float32_matmul_precision("high")

log("="*80)
log("STAGE 2 FINE-TUNING FROM RECOVERED STAGE-1 CHECKPOINT")
log("="*80)
log(f"UTC: {UTC0}  |  Device: {cfg.device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ── Phase 1: Build dataset (deterministic, same splits as original) ──
log("\n[1/5] Thermo tables...")
precompute_thermo_tables(cfg, n_comp=16, n_temp=24)

log("[2/5] Building dataset...")
full_df = build_full_dataset(cfg)
train_df, val_df, test_df = prepare_train_val_test_split(full_df, cfg)
train_real = real_only(train_df); val_real = real_only(val_df); test_real = real_only(test_df)
for n,d in [("full",full_df),("train",train_df),("val",val_df),("test",test_df),
            ("train_real",train_real),("val_real",val_real),("test_real",test_real)]:
    log(f"  {json.dumps(desc(n,d))}")

# ── Phase 2: Load stage1 checkpoint & evaluate baseline ──
log("\n[3/5] Loading stage-1 checkpoint...")
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(WORK / "stage1_best.pt", cfg.checkpoint_dir / "physics_node_best.pt")

model_s1 = PhysicsNODE(cfg.model)
trainer_s1 = Trainer(model_s1, cfg, use_gradnorm=False, use_homoscedastic=False)
ckpt = torch.load(cfg.checkpoint_dir / "physics_node_best.pt", map_location=cfg.device, weights_only=False)
model_s1.load_state_dict(ckpt["model"])
log("  Stage-1 checkpoint loaded successfully.")

log("\n  --- Stage-1 baseline evaluation ---")
s1_eval = {
    "val_full":  evaluate(trainer_s1, val_df, "s1_val_full", cfg),
    "test_full": evaluate(trainer_s1, test_df, "s1_test_full", cfg),
    "val_real":  evaluate(trainer_s1, val_real, "s1_val_real", cfg),
    "test_real": evaluate(trainer_s1, test_real, "s1_test_real", cfg),
}

# ── Phase 3: Stage 2 fine-tuning on real-only data ──
can_s2 = train_real["sample_id"].nunique() >= 8 and val_real["sample_id"].nunique() >= 3
if can_s2:
    log("\n[4/5] Stage 2: real-only fine-tuning (60 epochs, NO early stopping)...")
    s2cfg = deepcopy(cfg)
    s2cfg.model.learning_rate = 8e-5
    s2cfg.model.max_epochs = 60
    s2cfg.model.early_stopping_patience = 999  # effectively disabled
    s2cfg.model.batch_size = min(24, cfg.model.batch_size)
    s2cfg.model.scheduler_type = "cosine"
    s2cfg.model.scheduler_eta_min = 1e-6
    s2cfg.model.swa_start_epoch = 9999
    s2cfg.model.checkpoint_metric = "val_real_rmse"

    s2model = PhysicsNODE(s2cfg.model)
    s2model.load_state_dict(ckpt["model"])  # from stage1 best
    s2_train_loader, s2_val_loader = create_data_loaders(train_real, val_real, s2cfg)
    s2trainer = Trainer(s2model, s2cfg, use_gradnorm=False, use_homoscedastic=False)
    s2history = s2trainer.train(s2_train_loader, s2_val_loader, verbose=True)

    # save stage2 artifacts
    s2_best = s2cfg.checkpoint_dir / "physics_node_best.pt"
    s2_last = s2cfg.checkpoint_dir / "physics_node_last.pt"
    if s2_best.exists(): shutil.copy2(s2_best, ARTS / "stage2_best.pt")
    if s2_last.exists(): shutil.copy2(s2_last, ARTS / "stage2_last.pt")
    pd.DataFrame(s2history).to_csv(ARTS / "stage2_history.csv", index_label="epoch")

    # plot stage2 history
    hdf = pd.DataFrame(s2history)
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    ep = np.arange(1,len(hdf)+1)
    axes[0].plot(ep,hdf["train_loss"],label="train"); axes[0].plot(ep,hdf["val_loss"],label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(ep,hdf["val_rmse"],label="val_rmse")
    if "val_real_rmse" in hdf: axes[1].plot(ep,hdf["val_real_rmse"],label="val_real_rmse")
    axes[1].set_title("RMSE"); axes[1].legend()
    axes[2].plot(ep,hdf["val_endpoint_mae"],label="endpoint")
    if "val_real_endpoint_mae" in hdf: axes[2].plot(ep,hdf["val_real_endpoint_mae"],label="real_endpoint")
    axes[2].set_title("Endpoint MAE"); axes[2].legend()
    fig.suptitle("Stage 2: Real-Only Fine-Tuning"); plt.tight_layout()
    fig.savefig(ARTS/"stage2_history.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    # load best and evaluate
    s2trainer.load_checkpoint("best")
    final_trainer = s2trainer; final_cfg = s2cfg; final_stage = "stage2"
else:
    log("\n[4/5] Stage 2 SKIPPED: not enough real curves for fine-tuning.")
    final_trainer = trainer_s1; final_cfg = cfg; final_stage = "stage1"

# ── Phase 4: Final evaluation ──
log(f"\n[5/5] Final evaluation (best from {final_stage})...")
final_eval = {
    "val_full":  evaluate(final_trainer, val_df, "final_val_full", final_cfg),
    "test_full": evaluate(final_trainer, test_df, "final_test_full", final_cfg),
    "val_real":  evaluate(final_trainer, val_real, "final_val_real", final_cfg),
    "test_real": evaluate(final_trainer, test_real, "final_test_real", final_cfg),
}

# ── Phase 5: Publication figures ──
log("\nGenerating figures...")
try:
    final_trainer.model.eval()
    # Parity plot
    fig, ax = plt.subplots(figsize=(5,5))
    for split_name, split_df, color, marker in [("Val Real",val_real,"#0072B2","o"),("Test Real",test_real,"#D55E00","s")]:
        if len(split_df)==0: continue
        loader = make_loader(split_df, final_cfg, bs=64)
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in loader:
                static = batch["static"].to(final_cfg.device)
                f_true = batch["traj"].to(final_cfg.device)
                t_span = batch["t_span"].to(final_cfg.device)
                f_eq = batch["f_eq"].to(final_cfg.device)
                dG = batch["dG_norm"].to(final_cfg.device)
                obs = batch["obs_mask"].to(final_cfg.device)
                lengths = batch["lengths"]
                f_pred = final_trainer.model(static, f_eq, dG, t_span, lengths=lengths)
                ml = min(f_pred.shape[1], f_true.shape[1])
                mask = obs[:,:ml] > 0.5
                all_true.append(f_true[:,:ml][mask].cpu().numpy())
                all_pred.append(f_pred[:,:ml][mask].cpu().numpy())
        t = np.concatenate(all_true); p = np.concatenate(all_pred)
        ax.scatter(t, p, c=color, marker=marker, s=30, alpha=0.7, label=f"{split_name} (n={len(t)})")
    ax.plot([0,1],[0,1],"k--",alpha=0.5); ax.set_xlabel("Observed f_RA"); ax.set_ylabel("Predicted f_RA")
    ax.set_title("Parity Plot — Real Data Only"); ax.legend(); ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
    fig.savefig(ARTS/"parity_real.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    log("  parity_real.png saved")
except Exception as e:
    log(f"  Figure generation error: {e}")

# ── Write summary ──
summary = {
    "utc_start": UTC0, "utc_end": datetime.now(timezone.utc).isoformat(),
    "runtime_min": round((time.time()-T0)/60,2),
    "final_stage": final_stage,
    "device": str(final_cfg.device),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "dataset": {n:desc(n,d) for n,d in [("full",full_df),("train",train_df),("val",val_df),
                ("test",test_df),("train_real",train_real),("val_real",val_real),("test_real",test_real)]},
    "stage1_baseline": s1_eval,
    "final_evaluation": final_eval,
    "best_metric": final_trainer.best_metric_name if hasattr(final_trainer,"best_metric_name") else "unknown",
    "best_metric_value": float(final_trainer.best_checkpoint_metric) if hasattr(final_trainer,"best_checkpoint_metric") else -1,
}
(ARTS/"run_summary.json").write_text(json.dumps(summary,indent=2))
shutil.copy2(final_cfg.checkpoint_dir/"physics_node_best.pt", ARTS/"final_best.pt")

zip_path = shutil.make_archive("/kaggle/working/stage2_artifacts","zip",ARTS)
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

log("\n"+"="*80)
log("RUN COMPLETE")
log("="*80)
log(f"Runtime: {summary['runtime_min']} minutes")
log(f"Final stage: {final_stage}")
log(f"Best {summary['best_metric']}: {summary['best_metric_value']:.6f}")
log("\n--- STAGE-1 BASELINE ---")
log(json.dumps(s1_eval, indent=2))
log("\n--- FINAL (after stage 2) ---")
log(json.dumps(final_eval, indent=2))
log(f"\nArtifact zip: {zip_path}")
