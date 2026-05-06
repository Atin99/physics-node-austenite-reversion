# Project 4.3 normal-run repair cell.
# Paste this in a new Kaggle cell below the completed normal single-cell run.
# It reuses existing trained checkpoints and reruns evaluation/export without tracebacks.

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKING = Path("/kaggle/working")
ARTIFACT_ROOT = WORKING / "project_4_3_final_artifacts"
LOG_DIR = ARTIFACT_ROOT / "_logs"
SNAPSHOT_ZIP = WORKING / "project_4_3_final_artifacts_snapshot.zip"
LOG_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_PROFILE = {
    "PROJECT43_PROFILE": "deadline",
    "PROJECT43_FAST_MODE": "1",
    "PROJECT43_DATA_SYNTH_CALIB": "180",
    "PROJECT43_DATA_SYNTH_EXPLORE": "700",
    "PROJECT43_DATA_N_TIME_POINTS": "36",
    "PROJECT43_MODEL_BATCH_SIZE": "32",
    "PROJECT43_MODEL_USE_AMP": "1",
    "PROJECT43_MODEL_ADJOINT": "0",
    "PROJECT43_MODEL_USE_SN": "0",
    "PROJECT43_MODEL_RTOL": "3e-4",
    "PROJECT43_MODEL_ATOL": "1e-6",
    "PROJECT43_MODEL_MAX_STEPS": "2000",
    "PROJECT43_MODEL_AUG_DIM": "2",
    "PROJECT43_MODEL_HIDDEN_DIMS": "96,96,64",
    "PROJECT43_ODE_TIME_TRANSFORM": "log10",
    "PROJECT43_ODE_TIME_LOG_SCALE": "6.0",
}


def log(message=""):
    print(message, flush=True)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_profile_env():
    env = dict(DEFAULT_PROFILE)
    run_environment = ARTIFACT_ROOT / "run_environment.json"
    if run_environment.exists():
        try:
            payload = json.loads(run_environment.read_text(encoding="utf-8"))
            profile = payload.get("run_profile") or env["PROJECT43_PROFILE"]
            cfg = payload.get("profile_cfg") or {}
            env.update(
                {
                    "PROJECT43_PROFILE": str(profile),
                    "PROJECT43_FAST_MODE": str(cfg.get("fast_mode", env["PROJECT43_FAST_MODE"])),
                    "PROJECT43_DATA_SYNTH_CALIB": str(cfg.get("data_synth_calib", env["PROJECT43_DATA_SYNTH_CALIB"])),
                    "PROJECT43_DATA_SYNTH_EXPLORE": str(cfg.get("data_synth_explore", env["PROJECT43_DATA_SYNTH_EXPLORE"])),
                    "PROJECT43_DATA_N_TIME_POINTS": str(cfg.get("data_n_time_points", env["PROJECT43_DATA_N_TIME_POINTS"])),
                    "PROJECT43_MODEL_BATCH_SIZE": str(cfg.get("model_batch_size", env["PROJECT43_MODEL_BATCH_SIZE"])),
                    "PROJECT43_MODEL_USE_AMP": str(cfg.get("model_use_amp", env["PROJECT43_MODEL_USE_AMP"])),
                    "PROJECT43_MODEL_ADJOINT": str(cfg.get("model_adjoint", env["PROJECT43_MODEL_ADJOINT"])),
                    "PROJECT43_MODEL_USE_SN": str(cfg.get("model_use_sn", env["PROJECT43_MODEL_USE_SN"])),
                    "PROJECT43_MODEL_RTOL": str(cfg.get("model_rtol", env["PROJECT43_MODEL_RTOL"])),
                    "PROJECT43_MODEL_ATOL": str(cfg.get("model_atol", env["PROJECT43_MODEL_ATOL"])),
                    "PROJECT43_MODEL_MAX_STEPS": str(cfg.get("model_max_steps", env["PROJECT43_MODEL_MAX_STEPS"])),
                    "PROJECT43_MODEL_AUG_DIM": str(cfg.get("model_aug_dim", env["PROJECT43_MODEL_AUG_DIM"])),
                    "PROJECT43_MODEL_HIDDEN_DIMS": str(cfg.get("model_hidden_dims", env["PROJECT43_MODEL_HIDDEN_DIMS"])),
                    "PROJECT43_ODE_TIME_TRANSFORM": str(cfg.get("ode_time_transform", env["PROJECT43_ODE_TIME_TRANSFORM"])),
                    "PROJECT43_ODE_TIME_LOG_SCALE": str(cfg.get("ode_time_log_scale", env["PROJECT43_ODE_TIME_LOG_SCALE"])),
                }
            )
        except Exception as exc:
            log(f"[repair-note] Could not read run_environment.json; using deadline profile. Details: {exc!r}")
    return env


def choose_workspace():
    candidates = [
        WORKING / "project43_latent_stage2_real_only",
        WORKING / "project43_latent_stage1_full",
        WORKING / "project43_latent_node",
    ]
    return next((path for path in candidates if (path / "src").exists()), None)


def prepare_checkpoint_aliases(workspace):
    candidates = [
        workspace / "src" / "models" / "checkpoints" / "physics_node_best.pt",
        workspace / "src" / "models" / "checkpoints" / "physics_node_last.pt",
        workspace / "src" / "models" / "physics_node_best.pt",
        ARTIFACT_ROOT / "latent_stage2_real_only" / "checkpoints" / "physics_node_best.pt",
        ARTIFACT_ROOT / "latent_stage2_real_only" / "checkpoints" / "physics_node_last.pt",
        ARTIFACT_ROOT / "latent_stage1_full" / "checkpoints" / "physics_node_best.pt",
        ARTIFACT_ROOT / "latent_stage1_full" / "checkpoints" / "physics_node_last.pt",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        write_json(ARTIFACT_ROOT / "03_evaluation_export" / "repair_checkpoint_aliases.json", {"status": "skipped", "searched": [str(path) for path in candidates]})
        log("[repair-skip] No trained PhysicsNODE checkpoint was found; checkpoint-based evaluation was skipped.")
        return False

    models_dir = workspace / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    aliases = [models_dir / "stage2_fixed_best.pt", models_dir / "stage2_extended_best.pt"]
    for alias in aliases:
        shutil.copy2(source, alias)
    write_json(
        ARTIFACT_ROOT / "03_evaluation_export" / "repair_checkpoint_aliases.json",
        {"status": "ready", "source": str(source), "aliases": [str(path) for path in aliases]},
    )
    log(f"[repair] Checkpoint aliases prepared from {source}")
    return True


def run_eval_script(script_path, workspace, env):
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(workspace),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path = LOG_DIR / f"repair_eval_{script_path.name}.log"
    log_path.write_text(result.stdout + f"\nRETURN_CODE: {result.returncode}\n", encoding="utf-8", errors="replace")
    if result.returncode == 0:
        print(result.stdout, end="", flush=True)
        return True
    log(f"[repair-skip] {script_path.name} did not finish cleanly; full output saved to {log_path}")
    return False


def snapshot_artifacts():
    if ARTIFACT_ROOT.exists():
        if SNAPSHOT_ZIP.exists():
            SNAPSHOT_ZIP.unlink()
        shutil.make_archive(str(SNAPSHOT_ZIP.with_suffix("")), "zip", ARTIFACT_ROOT)
        log(f"[repair] Snapshot refreshed: {SNAPSHOT_ZIP}")


def main():
    workspace = choose_workspace()
    if workspace is None:
        log("[repair-skip] No normal latent workspace found under /kaggle/working.")
        return

    env = load_profile_env()
    if "stage2_real_only" in workspace.name:
        env["PROJECT43_DATA_REAL_ONLY"] = "1"
    checkpoint_ready = prepare_checkpoint_aliases(workspace)

    legacy_eval_dir = workspace / "legacy_project4_scripts"
    completed = []
    skipped = []
    for script_name in ["evaluate_comprehensive.py", "validate_calphad.py", "analysis.py", "ablation_study.py"]:
        if script_name in {"evaluate_comprehensive.py", "ablation_study.py"} and not checkpoint_ready:
            skipped.append(script_name)
            log(f"[repair-skip] {script_name}: no compatible checkpoint alias is available.")
            continue
        if script_name == "validate_calphad.py" and importlib.util.find_spec("pycalphad") is None:
            skipped.append(script_name)
            log("[repair-skip] validate_calphad.py: optional pycalphad package is not installed.")
            continue

        source = legacy_eval_dir / script_name
        target = workspace / script_name
        if source.exists():
            shutil.copy2(source, target)
        if not target.exists():
            skipped.append(script_name)
            log(f"[repair-skip] {script_name}: script not found.")
            continue
        if run_eval_script(target, workspace, env):
            completed.append(script_name)
        else:
            skipped.append(script_name)

    export_artifacts = ARTIFACT_ROOT / "03_evaluation_export"
    for folder in ["outputs", "analysis_results", "figures"]:
        source = workspace / folder
        target = export_artifacts / folder
        if source.exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)

    write_json(
        export_artifacts / "repair_eval_summary.json",
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": str(workspace),
            "completed": completed,
            "skipped": skipped,
            "snapshot_zip": str(SNAPSHOT_ZIP),
        },
    )
    snapshot_artifacts()
    log("[repair] Evaluation/export repair pass finished.")
    log(json.dumps({"completed": completed, "skipped": skipped}, indent=2))


try:
    main()
except Exception as exc:
    write_json(
        ARTIFACT_ROOT / "03_evaluation_export" / "repair_eval_summary.json",
        {"finished_utc": datetime.now(timezone.utc).isoformat(), "status": "skipped", "details": repr(exc)},
    )
    log(f"[repair-skip] Repair cell stopped before completion: {exc!r}")
