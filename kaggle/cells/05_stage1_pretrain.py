import subprocess
import sys
from pathlib import Path


workspace = Path("/kaggle/working/project43_latent_node")
src_dir = workspace / "src"

cmd = [
    sys.executable,
    str(src_dir / "main.py"),
    "--generate-data",
    "--train",
    "--epochs",
    "120",
    "--device",
    "cuda",
]
subprocess.check_call(cmd, cwd=src_dir)

