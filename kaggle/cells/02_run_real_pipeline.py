import json
import subprocess
import sys
from pathlib import Path


bundle_root = Path("/kaggle/working/project_4_3_bundle/project_4_3_bundle")
cmd = [
    sys.executable,
    str(bundle_root / "scripts" / "run_project43_pipeline.py"),
    "--n-synth",
    "5000",
]
subprocess.check_call(cmd, cwd=bundle_root)

summary_path = bundle_root / "outputs" / "pipeline_run_summary.json"
print(json.loads(summary_path.read_text(encoding="utf-8")))

