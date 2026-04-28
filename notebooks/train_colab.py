# -*- coding: utf-8 -*-
"""
PhysicsNODE Training Notebook — Google Colab GPU
=================================================

Upload this file to Google Colab (File → Upload notebook).
Select GPU runtime: Runtime → Change runtime type → T4 GPU.

This notebook runs the FULL pipeline:
  1. Install dependencies
  2. Upload/mount project files
  3. Generate 3-tier data (real + calibrated + exploratory)
  4. Train PhysicsNODE (~500 epochs on GPU)
  5. Optuna hyperparameter search (50 trials)
  6. SWAG posterior collection
  7. SHAP explainability
  8. Generate all publication figures
  9. Save everything to Google Drive

Estimated time: ~30-60 min on T4 GPU
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
# !pip install -q torch torchvision torchaudio
# !pip install -q torchdiffeq optuna scikit-learn pandas numpy matplotlib shap
# NOTE: Do NOT install pysr in Colab — it needs Julia and is complex.
# Symbolic regression should be run locally if needed.

print("=" * 60)
print("STEP 0: Checking environment")
print("=" * 60)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be very slow on CPU.")
    print("Go to Runtime → Change runtime type → T4 GPU")

# ============================================================================
# CELL 2: Mount Google Drive & Set Up Project
# ============================================================================
import os
import sys
from pathlib import Path

# Mount Google Drive for checkpoint persistence
try:
    from google.colab import drive
    # drive.mount('/content/drive')  # <-- MUST BE RUN IN NOTEBOOK CELL, NOT SCRIPT
    DRIVE_DIR = Path('/content/drive/MyDrive/PhysicsNODE')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    IN_COLAB = True
    print(f"Google Drive mounted. Checkpoints → {DRIVE_DIR}")
except ImportError:
    DRIVE_DIR = Path('./results')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    IN_COLAB = False
    print("Not in Colab. Saving locally.")

# Option A: Upload zip file
# Upload medium_mn_neural_ode.zip to Colab, then:
# !unzip -q medium_mn_neural_ode.zip -d /content/

# Option B: Git clone (if pushed to GitHub)
# !git clone https://github.com/YOUR_USERNAME/medium_mn_neural_ode.git /content/medium_mn_neural_ode

# Option C: Files are already in /content/medium_mn_neural_ode/
PROJECT_DIR = Path('/content/medium_mn_neural_ode') if IN_COLAB else Path('.')
if not PROJECT_DIR.exists():
    # Try current directory
    PROJECT_DIR = Path('.')

sys.path.insert(0, str(PROJECT_DIR))
os.chdir(str(PROJECT_DIR))
print(f"Project dir: {PROJECT_DIR.resolve()}")
print(f"Files: {[f.name for f in PROJECT_DIR.iterdir() if f.suffix == '.py']}")

# ============================================================================
# CELL 3: Generate Data
# ============================================================================
print("\n" + "=" * 60)
print("STEP 1: Generate 3-Tier Data")
print("=" * 60)

from config import get_config
from data_generator import (
    build_full_dataset, save_synthetic_data,
    create_literature_validation_data, plot_synthetic_curves,
    prepare_train_val_split
)
from thermodynamics import precompute_thermo_tables

config = get_config()
# Force GPU if available
if torch.cuda.is_available():
    config.device = torch.device('cuda')
    print("Using CUDA GPU!")
else:
    config.device = torch.device('cpu')

# Precompute thermodynamic tables
precompute_thermo_tables(config, 20, 30)

# Build full 3-tier dataset
df = build_full_dataset(config)
save_synthetic_data(df, config)
df_train, df_val = prepare_train_val_split(df, config)
df_train.to_csv(config.synthetic_dir / "train.csv", index=False)
df_val.to_csv(config.synthetic_dir / "val.csv", index=False)

# Save literature validation
create_literature_validation_data(config)
plot_synthetic_curves(df, config=config)

# Summary
print(f"\nDataset ready: {len(df)} points ({df['sample_id'].nunique()} curves)")
if 'provenance' in df.columns:
    for prov, count in df['provenance'].value_counts().items():
        print(f"  {prov}: {count} points")

# ============================================================================
# CELL 4: Quick Test Run (50 epochs)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2a: Quick Training Test (50 epochs)")
print("=" * 60)

from model import PhysicsNODE
from trainer import Trainer, set_seed, create_data_loaders

# Short test to verify everything works
config.model.max_epochs = 50
config.model.early_stopping_patience = 50  # Don't stop early during test

set_seed(config.model.random_seed)
model = PhysicsNODE(config.model)
print(model.get_model_summary())

tr_loader, va_loader = create_data_loaders(df_train, df_val, config)
trainer = Trainer(model, config)

print(f"Train: {len(tr_loader)} batches | Val: {len(va_loader)} batches")
print("Starting test run...")
history = trainer.train(tr_loader, va_loader)

print(f"\nTest run complete! Best val loss: {trainer.best_val_loss:.6f}")
print("If this worked, proceed to full training below.\n")

# ============================================================================
# CELL 5: Full Training (500 epochs)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2b: Full Training (500 epochs)")
print("=" * 60)

# Reset for full training
config.model.max_epochs = 500
config.model.early_stopping_patience = 80
config.model.swa_start_epoch = 400

set_seed(config.model.random_seed)
model = PhysicsNODE(config.model)
tr_loader, va_loader = create_data_loaders(df_train, df_val, config)
trainer = Trainer(model, config)

print("Starting full training...")
history = trainer.train(tr_loader, va_loader)

# Save checkpoint to Google Drive
import shutil
best_ckpt = config.checkpoint_dir / "physics_node_best.pt"
if best_ckpt.exists():
    drive_ckpt = DRIVE_DIR / "physics_node_best.pt"
    shutil.copy2(best_ckpt, drive_ckpt)
    print(f"Checkpoint saved to {drive_ckpt}")

last_ckpt = config.checkpoint_dir / "physics_node_last.pt"
if last_ckpt.exists():
    shutil.copy2(last_ckpt, DRIVE_DIR / "physics_node_last.pt")

print(f"Training complete! Best val loss: {trainer.best_val_loss:.6f}")

# ============================================================================
# CELL 6: Optuna Hyperparameter Search (50 trials)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Optuna HPO (50 trials)")
print("=" * 60)

from trainer import run_optuna_hpo
import json

hpo_result = run_optuna_hpo(df_train, df_val, config, n_trials=50)
print(f"\nBest params: {hpo_result['best_params']}")
print(f"Best loss: {hpo_result['best_value']:.6f}")

# Save HPO results
with open(DRIVE_DIR / "hpo_results.json", 'w') as f:
    json.dump({
        'best_params': hpo_result['best_params'],
        'best_value': float(hpo_result['best_value']),
    }, f, indent=2)

# ============================================================================
# CELL 7: Retrain with Best Params
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Retrain with Best Params (500 epochs)")
print("=" * 60)

bp = hpo_result['best_params']
config.model.hidden_dims = [bp['hidden']] * bp['n_layers']
config.model.augmented_dim = bp['aug_dim']
config.model.learning_rate = bp['lr']
config.model.init_dropout_rate = bp['dropout']
config.model.weight_decay = bp['wd']
config.model.max_epochs = 500
config.model.early_stopping_patience = 80

set_seed(config.model.random_seed)
model = PhysicsNODE(config.model)
tr_loader, va_loader = create_data_loaders(df_train, df_val, config)
trainer = Trainer(model, config)
history = trainer.train(tr_loader, va_loader)

# Save to Drive
best_ckpt = config.checkpoint_dir / "physics_node_best.pt"
if best_ckpt.exists():
    shutil.copy2(best_ckpt, DRIVE_DIR / "physics_node_best_optimized.pt")
    print(f"Optimized checkpoint saved.")

print(f"Retrained! Best val loss: {trainer.best_val_loss:.6f}")

# ============================================================================
# CELL 8: SHAP Explainability
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: SHAP Explainability")
print("=" * 60)

from explainability import run_explainability_suite
import numpy as np

# Create sample data for SHAP
from features import compute_diffusivity, compute_hollomon_jaffe
from thermodynamics import get_equilibrium_RA, get_driving_force

sample_data = []
for _, row in df_train.drop_duplicates('sample_id').head(100).iterrows():
    T_K = row['T_celsius'] + 273.15
    D = compute_diffusivity(T_K)
    dG = get_driving_force({'Mn': row['Mn'], 'C': row['C']}, row['T_celsius'], force_fallback=True)
    P = compute_hollomon_jaffe(T_K, max(row['t_seconds'], 1.0))
    sample_data.append([
        (T_K - config.data.T_ref) / config.data.T_scale,
        row['Mn'], row['C'], row.get('Al', 0), row.get('Si', 0),
        np.log10(D + 1e-30), dG / 1000.0, P / 20000.0
    ])
sample_data = np.array(sample_data, dtype=np.float32)

explain_results = run_explainability_suite(model, sample_data, config)
print("\nExplainability complete.")

# ============================================================================
# CELL 9: Generate All Publication Figures
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Generate Publication Figures")
print("=" * 60)

from visualizations import generate_all_figures

sens = explain_results.get('sensitivity') if explain_results else None
shap = explain_results.get('shap') if explain_results else None

generate_all_figures(
    training_history=history,
    sensitivity_results=sens,
    shap_result=shap,
    config=config
)

# Copy figures to Drive
fig_dir = config.figure_dir
drive_fig_dir = DRIVE_DIR / "figures"
drive_fig_dir.mkdir(exist_ok=True)
for fig_file in fig_dir.glob("*.png"):
    shutil.copy2(fig_file, drive_fig_dir / fig_file.name)
for fig_file in fig_dir.glob("*.pdf"):
    shutil.copy2(fig_file, drive_fig_dir / fig_file.name)

print(f"\nFigures saved to {drive_fig_dir}")
print(f"Files: {[f.name for f in drive_fig_dir.iterdir()]}")

# ============================================================================
# CELL 10: Summary & Download
# ============================================================================
print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)

print(f"\nAll results saved to: {DRIVE_DIR}")
print(f"\nFiles in Drive:")
for f in sorted(DRIVE_DIR.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(DRIVE_DIR)} ({size_mb:.1f} MB)")

print(f"""
Next steps:
  1. Download figures from {DRIVE_DIR / 'figures'}
  2. Load checkpoint for inference:
     model.load_state_dict(torch.load('physics_node_best_optimized.pt')['model'])
  3. Run symbolic regression LOCALLY (needs Julia/PySR):
     python main.py --symbolic
  4. Launch Streamlit app LOCALLY:
     python main.py --app
""")
