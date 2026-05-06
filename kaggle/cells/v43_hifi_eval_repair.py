# Project 4.3 HiFi eval-only repair cell.
# Paste this whole file into one Kaggle notebook cell.
# Attach project_4_3_hifi_eval_repair_input.zip as the only required input.

import glob
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
WORKSPACE = WORKING / "project43_latent_stage2_real_only"
OUTPUT_ROOT = WORKING / "project_4_3_hifi_eval_repair"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_PATH = OUTPUT_ROOT / "repair_summary.json"
OUTPUT_ZIP = WORKING / "project_4_3_hifi_eval_repair_outputs.zip"

HIFI_EVAL_ENV = {
    "PROJECT43_PROFILE": "balanced",
    "PROJECT43_FAST_MODE": "1",
    "PROJECT43_DATA_SYNTH_CALIB": "360",
    "PROJECT43_DATA_SYNTH_EXPLORE": "1200",
    "PROJECT43_DATA_N_TIME_POINTS": "48",
    "PROJECT43_DATA_REAL_WEIGHT": "5.0",
    "PROJECT43_DATA_REAL_ONLY": "1",
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
}


def log(message=""):
    print(message, flush=True)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def ensure_pkg(import_name, pip_name=None):
    if importlib.util.find_spec(import_name) is not None:
        return
    package_name = pip_name or import_name
    log(f"[setup] Installing {package_name} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


def find_input_zip():
    candidates = [Path(p) for p in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True)]
    candidates = [p for p in candidates if "hifi_eval_repair" in p.name.lower()]
    if not candidates:
        candidates = [Path(p) for p in glob.glob(str(INPUT / "**" / "*.zip"), recursive=True)]
    if not candidates:
        raise FileNotFoundError("Attach project_4_3_hifi_eval_repair_input.zip as a Kaggle dataset input.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_extracted_workspace():
    for path in INPUT.rglob("project43_latent_stage2_real_only"):
        if (path / "src" / "model.py").exists() and (path / "src" / "config.py").exists():
            return path
    return None


def prepare_workspace():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)

    extracted = find_extracted_workspace()
    if extracted is not None:
        shutil.copytree(extracted, WORKSPACE, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        return {"source_kind": "extracted", "source_path": str(extracted)}

    zip_path = find_input_zip()
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [name for name in zf.namelist() if name.startswith("project43_latent_stage2_real_only/")]
        if not members:
            raise FileNotFoundError("Input zip does not contain project43_latent_stage2_real_only/")
        zf.extractall(WORKING, members)
    return {"source_kind": "zip", "source_path": str(zip_path)}


def prepare_checkpoint_aliases():
    candidates = [
        WORKSPACE / "src" / "models" / "checkpoints" / "physics_node_best.pt",
        WORKSPACE / "src" / "models" / "checkpoints" / "physics_node_last.pt",
        WORKSPACE / "project_4_3_final_artifacts" / "latent_stage2_real_only" / "checkpoints" / "physics_node_best.pt",
    ]
    checkpoint = next((path for path in candidates if path.exists()), None)
    if checkpoint is None:
        raise FileNotFoundError("No HiFi trained checkpoint found in repair input.")

    models_dir = WORKSPACE / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        "stage2_fixed_best.pt": models_dir / "stage2_fixed_best.pt",
        "stage2_extended_best.pt": models_dir / "stage2_extended_best.pt",
    }
    for alias_path in aliases.values():
        shutil.copy2(checkpoint, alias_path)
    return {"checkpoint": str(checkpoint), "aliases": {name: str(path) for name, path in aliases.items()}}


def run_eval_script(script_name, env):
    script_path = WORKSPACE / script_name
    if not script_path.exists():
        legacy_path = WORKSPACE / "legacy_project4_scripts" / script_name
        if legacy_path.exists():
            shutil.copy2(legacy_path, script_path)
    if not script_path.exists():
        return {"script": script_name, "status": "skipped", "reason": "script not found"}

    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WORKSPACE),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path = LOG_DIR / f"{script_name}.log"
    log_path.write_text(result.stdout + f"\nRETURN_CODE: {result.returncode}\n", encoding="utf-8", errors="replace")
    if result.returncode == 0:
        print(result.stdout, end="", flush=True)
        return {"script": script_name, "status": "completed", "log": str(log_path)}
    log(f"[skip] {script_name} returned {result.returncode}; full output saved to {log_path}")
    return {"script": script_name, "status": "skipped", "return_code": result.returncode, "log": str(log_path)}


def copy_outputs():
    for folder in ["figures", "analysis_results", "outputs", "models"]:
        source = WORKSPACE / folder
        target = OUTPUT_ROOT / folder
        if source.exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    shutil.make_archive(str(OUTPUT_ZIP.with_suffix("")), "zip", OUTPUT_ROOT)
    return str(OUTPUT_ZIP)


def main():
    os.environ.update(HIFI_EVAL_ENV)
    log("PROJECT 4.3 HIFI EVAL-ONLY REPAIR")
    log("This cell does not train. It only restores checkpoint aliases and reruns evaluation/export.")

    for import_name, pip_name in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("torchdiffeq", "torchdiffeq"),
    ]:
        ensure_pkg(import_name, pip_name)

    source_info = prepare_workspace()
    alias_info = prepare_checkpoint_aliases()
    log(f"[repair] Workspace: {WORKSPACE}")
    log(f"[repair] Checkpoint: {alias_info['checkpoint']}")

    scripts = ["evaluate_comprehensive.py", "analysis.py", "ablation_study.py"]
    if importlib.util.find_spec("pycalphad") is not None:
        scripts.insert(1, "validate_calphad.py")
    else:
        log("[skip] validate_calphad.py: optional pycalphad package is not installed.")

    results = [run_eval_script(script_name, HIFI_EVAL_ENV) for script_name in scripts]
    output_zip = copy_outputs()
    summary = {
        "status": "finished",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_info,
        "alias_info": alias_info,
        "scripts": results,
        "output_zip": output_zip,
    }
    write_json(SUMMARY_PATH, summary)
    copy_outputs()

    log("HIFI EVAL-ONLY REPAIR FINISHED")
    log(json.dumps(summary, indent=2))


try:
    main()
except Exception as exc:
    write_json(
        SUMMARY_PATH,
        {
            "status": "stopped",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
        },
    )
    log(f"[stopped] HiFi eval-only repair stopped: {exc!r}")
    raise
