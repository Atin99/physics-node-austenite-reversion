import shutil
from pathlib import Path


bundle_root = Path("/kaggle/working/project_4_3_bundle/project_4_3_bundle")
latent_src = bundle_root / "src" / "latent_node"
legacy_data = bundle_root / "data" / "legacy_project4"
legacy_cells = bundle_root / "legacy_kaggle_cells"
legacy_eval = bundle_root / "legacy_project4_scripts"

workspace = Path("/kaggle/working/project43_latent_node")
if workspace.exists():
    shutil.rmtree(workspace)
workspace.mkdir(parents=True, exist_ok=True)

shutil.copytree(latent_src, workspace / "src")
shutil.copytree(legacy_data, workspace / "data")
if legacy_cells.exists():
    shutil.copytree(legacy_cells, workspace / "legacy_kaggle_cells")
if legacy_eval.exists():
    shutil.copytree(legacy_eval, workspace / "legacy_project4_scripts")

print("Prepared:", workspace)
print("Contents:", [p.name for p in sorted(workspace.iterdir())])
