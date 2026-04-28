"""Create upload bundle for Kaggle stage2 extended retraining."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
OUT = ROOT / "kaggle" / "upload_bundles" / "stage2_extended_200ep.zip"

# Source files needed
src_files = [
    "config.py", "data_generator.py", "losses.py", "model.py",
    "thermodynamics.py", "trainer.py", "features.py", "real_data.py",
    "explainability.py", "main.py", "optimizer_annealing.py",
    "visualizations.py", "streamlit_app.py", "symbolic_regression.py",
    "complete_project.py", "publication_pipeline.py", "__init__.py",
]

# Data files needed
data_files = [
    "data/calphad_tables/C_grid.npy",
    "data/calphad_tables/delta_G_table.npy",
    "data/calphad_tables/f_eq_table.npy",
    "data/calphad_tables/Mn_grid.npy",
    "data/calphad_tables/T_grid.npy",
    "data/literature_validation/literature_validation.csv",
    "data/DATASET_CARD.md",
]

# Checkpoint: use stage2_fixed_best.pt (current best, already 60 epochs of stage2)
checkpoint = "stage2_fixed_best.pt"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    # Source files
    for f in src_files:
        p = SRC / f
        if p.exists():
            zf.write(p, f)
            print(f"  + {f}")
        else:
            print(f"  SKIP {f} (not found)")

    # Data files
    for f in data_files:
        p = SRC / f
        if p.exists():
            zf.write(p, f)
            print(f"  + {f}")
        else:
            print(f"  SKIP {f}")

    # Checkpoint
    ckpt_path = ROOT / "models" / checkpoint
    if ckpt_path.exists():
        zf.write(ckpt_path, checkpoint)
        print(f"  + {checkpoint} ({ckpt_path.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"  ERROR: {checkpoint} not found!")

print(f"\nBundle: {OUT}")
print(f"Size: {OUT.stat().st_size / 1024:.0f} KB")
