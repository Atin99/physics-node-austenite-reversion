import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def ensure_pkg(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


for pkg in ["numpy", "pandas", "scikit-learn", "torchdiffeq", "tqdm"]:
    ensure_pkg(pkg)

zip_candidates = sorted(glob.glob("/kaggle/input/**/*.zip", recursive=True))
bundle_zip = None
for candidate in zip_candidates:
    name = os.path.basename(candidate).lower()
    if "project_4_3_kaggle_input" in name:
        bundle_zip = candidate
        break
if bundle_zip is None:
    raise FileNotFoundError("Upload project_4_3_kaggle_input.zip as a Kaggle dataset first.")

work_root = Path("/kaggle/working/project_4_3_bundle")
if work_root.exists():
    shutil.rmtree(work_root)
work_root.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(bundle_zip, "r") as zf:
    zf.extractall(work_root)

bundle_root = next(work_root.glob("project_4_3_bundle"))
print("Bundle root:", bundle_root)
print("Top level:", [p.name for p in sorted(bundle_root.iterdir())])

