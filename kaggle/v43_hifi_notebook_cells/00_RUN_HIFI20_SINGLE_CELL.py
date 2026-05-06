# Project 4.3 HiFi20 Kaggle single-cell runner.
# Paste this whole file into one Kaggle notebook cell, attach the HiFi20 zip,
# enable GPU, then Commit/Run.

import glob
import os
from pathlib import Path


HIFI20_ENV = {
    # Use the existing runner machinery, but force a higher quality 20h-safe setup.
    "PROJECT43_PROFILE": "balanced",
    "PROJECT43_BASELINE_N_SYNTH": "3500",
    "PROJECT43_STAGE1_EPOCHS": "90",
    "PROJECT43_STAGE2_EPOCHS": "180",
    "PROJECT43_DATA_SYNTH_CALIB": "360",
    "PROJECT43_DATA_SYNTH_EXPLORE": "1200",
    "PROJECT43_DATA_N_TIME_POINTS": "48",
    "PROJECT43_DATA_REAL_WEIGHT": "5.0",
    "PROJECT43_MODEL_BATCH_SIZE": "32",
    "PROJECT43_MODEL_USE_AMP": "1",
    "PROJECT43_MODEL_ADJOINT": "0",
    "PROJECT43_MODEL_USE_SN": "0",
    "PROJECT43_MODEL_RTOL": "2e-4",
    "PROJECT43_MODEL_ATOL": "1e-6",
    "PROJECT43_MODEL_MAX_STEPS": "3000",
    "PROJECT43_MODEL_AUG_DIM": "3",
    "PROJECT43_MODEL_HIDDEN_DIMS": "128,128,96",
    "PROJECT43_ODE_TIME_TRANSFORM": "log10",
    "PROJECT43_ODE_TIME_LOG_SCALE": "6.0",
    "PHYSICSNODE_BATCH_LOG_EVERY": "3",
    "PHYSICSNODE_BATCH_HEARTBEAT_SEC": "30",
}

os.environ.update(HIFI20_ENV)
print("PROJECT 4.3 HIFI20 WRAPPER")
for key in sorted(HIFI20_ENV):
    print(f"{key}={os.environ[key]}")


def find_runner():
    candidates = []
    for pattern in [
        "/kaggle/input/**/project_4_3_bundle/kaggle/notebook_cells/00_RUN_ALL_SINGLE_CELL.py",
        "/kaggle/input/**/project_4_3_hifi20_bundle/kaggle/notebook_cells/00_RUN_ALL_SINGLE_CELL.py",
        "/kaggle/input/**/00_RUN_ALL_SINGLE_CELL.py",
    ]:
        candidates.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError("Attach project_4_3_hifi20_kaggle_input.zip or project_4_3_kaggle_input.zip as a Kaggle dataset.")
    candidates.sort(key=lambda p: (("hifi20" in str(p).lower()), p.stat().st_mtime), reverse=True)
    return candidates[0]


runner = find_runner()
print(f"Executing base runner: {runner}")
code = runner.read_text(encoding="utf-8")
exec(compile(code, str(runner), "exec"), {"__name__": "__main__"})
