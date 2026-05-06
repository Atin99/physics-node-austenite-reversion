import shutil
import subprocess
import sys
from pathlib import Path


workspace = Path("/kaggle/working/project43_latent_node")
legacy_eval_dir = workspace / "legacy_project4_scripts"

for script_name in ["evaluate_comprehensive.py", "validate_calphad.py", "analysis.py", "ablation_study.py"]:
    src = legacy_eval_dir / script_name
    dst = workspace / script_name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
    if dst.exists():
        subprocess.call([sys.executable, str(dst)], cwd=workspace)

print("Finished evaluation/export pass.")
print("Check /kaggle/working/project43_latent_node and artifact folders for outputs.")
