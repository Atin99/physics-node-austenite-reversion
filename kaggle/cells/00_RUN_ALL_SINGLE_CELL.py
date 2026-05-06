# Project 4.3 Kaggle single-cell runner.
# Paste this whole file into one Kaggle notebook cell, attach
# project_4_3_kaggle_input.zip, enable GPU, then Commit/Run.

import glob
import importlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path


RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()
RUN_STARTED_WALL = time.time()
RUNNER_VERSION = "2026-05-06-kaggle-noerror-eval-v5"
CONTINUE_ON_STAGE_ERROR = os.environ.get("PROJECT43_CONTINUE_ON_ERROR", "1") != "0"
RUN_PROFILE = os.environ.get("PROJECT43_PROFILE", "deadline").strip().lower()

PROFILE_DEFAULTS = {
    "full": {
        "baseline_n_synth": 5000,
        "stage1_epochs": 120,
        "stage2_epochs": 150,
        "data_synth_calib": 500,
        "data_synth_explore": 2000,
        "data_n_time_points": 60,
        "model_batch_size": 16,
        "model_use_amp": "0",
        "model_adjoint": "1",
        "model_use_sn": "1",
        "model_rtol": "1e-5",
        "model_atol": "1e-7",
        "model_max_steps": 10000,
        "model_aug_dim": 4,
        "model_hidden_dims": "128,128,96,64",
        "ode_time_transform": "raw",
        "ode_time_log_scale": "6.0",
        "fast_mode": "0",
    },
    "balanced": {
        "baseline_n_synth": 3000,
        "stage1_epochs": 70,
        "stage2_epochs": 90,
        "data_synth_calib": 240,
        "data_synth_explore": 900,
        "data_n_time_points": 42,
        "model_batch_size": 24,
        "model_use_amp": "1",
        "model_adjoint": "0",
        "model_use_sn": "0",
        "model_rtol": "2e-4",
        "model_atol": "1e-6",
        "model_max_steps": 2500,
        "model_aug_dim": 3,
        "model_hidden_dims": "128,96,64",
        "ode_time_transform": "log10",
        "ode_time_log_scale": "6.0",
        "fast_mode": "1",
    },
    "deadline": {
        "baseline_n_synth": 2200,
        "stage1_epochs": 40,
        "stage2_epochs": 70,
        "data_synth_calib": 180,
        "data_synth_explore": 700,
        "data_n_time_points": 36,
        "model_batch_size": 32,
        "model_use_amp": "1",
        "model_adjoint": "0",
        "model_use_sn": "0",
        "model_rtol": "3e-4",
        "model_atol": "1e-6",
        "model_max_steps": 2000,
        "model_aug_dim": 2,
        "model_hidden_dims": "96,96,64",
        "ode_time_transform": "log10",
        "ode_time_log_scale": "6.0",
        "fast_mode": "1",
    },
}

if RUN_PROFILE not in PROFILE_DEFAULTS:
    RUN_PROFILE = "deadline"
PROFILE_CFG = PROFILE_DEFAULTS[RUN_PROFILE]
BASELINE_N_SYNTH = int(os.environ.get("PROJECT43_BASELINE_N_SYNTH", str(PROFILE_CFG["baseline_n_synth"])))
STAGE1_EPOCHS = int(os.environ.get("PROJECT43_STAGE1_EPOCHS", str(PROFILE_CFG["stage1_epochs"])))
STAGE2_EPOCHS = int(os.environ.get("PROJECT43_STAGE2_EPOCHS", str(PROFILE_CFG["stage2_epochs"])))

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
ARTIFACT_ROOT = WORKING / "project_4_3_final_artifacts"
STATE_DIR = ARTIFACT_ROOT / "_state"
LOG_DIR = ARTIFACT_ROOT / "_logs"
SNAPSHOT_ZIP = WORKING / "project_4_3_final_artifacts_snapshot.zip"
EXTRACT_ROOT = WORKING / "project_4_3_bundle"
EXPECTED_BUNDLE_NAME = "project_4_3_bundle"

for directory in [ARTIFACT_ROOT, STATE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

version_file = STATE_DIR / "_runner_version.txt"
if version_file.exists() and version_file.read_text(encoding="utf-8").strip() != RUNNER_VERSION:
    shutil.rmtree(STATE_DIR)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    for old_workspace in [WORKING / "project43_latent_stage1_full", WORKING / "project43_latent_stage2_real_only"]:
        if old_workspace.exists():
            shutil.rmtree(old_workspace)
version_file.write_text(RUNNER_VERSION, encoding="utf-8")


def log(message=""):
    print(message, flush=True)


def profile_env(stage_name=None):
    env = {
        "PROJECT43_PROFILE": RUN_PROFILE,
        "PROJECT43_FAST_MODE": PROFILE_CFG["fast_mode"],
        "PROJECT43_STAGE_NAME": stage_name or "",
        "PROJECT43_DATA_SYNTH_CALIB": os.environ.get("PROJECT43_DATA_SYNTH_CALIB", str(PROFILE_CFG["data_synth_calib"])),
        "PROJECT43_DATA_SYNTH_EXPLORE": os.environ.get("PROJECT43_DATA_SYNTH_EXPLORE", str(PROFILE_CFG["data_synth_explore"])),
        "PROJECT43_DATA_N_TIME_POINTS": os.environ.get("PROJECT43_DATA_N_TIME_POINTS", str(PROFILE_CFG["data_n_time_points"])),
        "PROJECT43_MODEL_BATCH_SIZE": os.environ.get("PROJECT43_MODEL_BATCH_SIZE", str(PROFILE_CFG["model_batch_size"])),
        "PROJECT43_MODEL_USE_AMP": os.environ.get("PROJECT43_MODEL_USE_AMP", PROFILE_CFG["model_use_amp"]),
        "PROJECT43_MODEL_ADJOINT": os.environ.get("PROJECT43_MODEL_ADJOINT", PROFILE_CFG["model_adjoint"]),
        "PROJECT43_MODEL_USE_SN": os.environ.get("PROJECT43_MODEL_USE_SN", PROFILE_CFG["model_use_sn"]),
        "PROJECT43_MODEL_RTOL": os.environ.get("PROJECT43_MODEL_RTOL", PROFILE_CFG["model_rtol"]),
        "PROJECT43_MODEL_ATOL": os.environ.get("PROJECT43_MODEL_ATOL", PROFILE_CFG["model_atol"]),
        "PROJECT43_MODEL_MAX_STEPS": os.environ.get("PROJECT43_MODEL_MAX_STEPS", str(PROFILE_CFG["model_max_steps"])),
        "PROJECT43_MODEL_AUG_DIM": os.environ.get("PROJECT43_MODEL_AUG_DIM", str(PROFILE_CFG["model_aug_dim"])),
        "PROJECT43_MODEL_HIDDEN_DIMS": os.environ.get("PROJECT43_MODEL_HIDDEN_DIMS", PROFILE_CFG["model_hidden_dims"]),
        "PROJECT43_ODE_TIME_TRANSFORM": os.environ.get("PROJECT43_ODE_TIME_TRANSFORM", PROFILE_CFG["ode_time_transform"]),
        "PROJECT43_ODE_TIME_LOG_SCALE": os.environ.get("PROJECT43_ODE_TIME_LOG_SCALE", PROFILE_CFG["ode_time_log_scale"]),
    }
    return env


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def stage_marker(stage):
    return STATE_DIR / f"{stage}.done.json"


def stage_failure(stage):
    return STATE_DIR / f"{stage}.failed.json"


def stage_live(stage):
    return STATE_DIR / f"{stage}.live.json"


def is_done(stage):
    return stage_marker(stage).exists()


def mark_done(stage, payload=None):
    payload = dict(payload or {})
    payload.update(
        {
            "stage": stage,
            "status": "done",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_hours": round((time.time() - RUN_STARTED_WALL) / 3600.0, 4),
        }
    )
    failed = stage_failure(stage)
    if failed.exists():
        failed.unlink()
    live = stage_live(stage)
    if live.exists():
        live.unlink()
    write_json(stage_marker(stage), payload)


def mark_failed(stage, exc):
    payload = {
        "stage": stage,
        "status": "failed",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": round((time.time() - RUN_STARTED_WALL) / 3600.0, 4),
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    live = stage_live(stage)
    if live.exists():
        live.unlink()
    write_json(stage_failure(stage), payload)


def snapshot_artifacts():
    if SNAPSHOT_ZIP.exists():
        SNAPSHOT_ZIP.unlink()
    shutil.make_archive(str(SNAPSHOT_ZIP.with_suffix("")), "zip", ARTIFACT_ROOT)
    log(f"Snapshot: {SNAPSHOT_ZIP}")


def ensure_pkg(import_name, pip_name=None):
    try:
        importlib.import_module(import_name)
    except ImportError:
        pkg = pip_name or import_name
        log(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        importlib.import_module(import_name)


def run_cmd(stage, cmd, cwd=None, env=None, check=True):
    log("")
    log("=" * 80)
    log(f"RUN {stage}")
    log("CMD: " + " ".join(str(x) for x in cmd))
    if cwd:
        log(f"CWD: {cwd}")
    log("=" * 80)

    merged_env = os.environ.copy()
    merged_env["PYTHONUNBUFFERED"] = "1"
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    log_path = LOG_DIR / f"{stage}.log"
    heartbeat_sec = max(10, int(os.environ.get("PROJECT43_CMD_HEARTBEAT_SEC", "30")))
    live_path = stage_live(stage)
    with log_path.open("a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n\n===== {datetime.now(timezone.utc).isoformat()} =====\n")
        lf.write("CMD: " + " ".join(str(x) for x in cmd) + "\n")
        process = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        stage_t0 = time.time()
        last_output_t = stage_t0
        write_json(
            live_path,
            {
                "stage": stage,
                "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "cmd": [str(x) for x in cmd],
                "cwd": str(cwd) if cwd else None,
            },
        )
        line_queue = queue.Queue()

        def _reader():
            try:
                for line in process.stdout:
                    line_queue.put(line)
            finally:
                line_queue.put(None)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        while True:
            try:
                item = line_queue.get(timeout=1.0)
            except queue.Empty:
                item = "__timeout__"

            if item == "__timeout__":
                now = time.time()
                if now - last_output_t >= heartbeat_sec:
                    hb = (
                        f"[runner-heartbeat] {stage} still running | elapsed={(now - stage_t0) / 60.0:.1f}m "
                        f"| no child output for {int(now - last_output_t)}s"
                    )
                    print(hb, flush=True)
                    lf.write(hb + "\n")
                    lf.flush()
                    write_json(
                        live_path,
                        {
                            "stage": stage,
                            "status": "running",
                            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                            "elapsed_minutes": round((now - stage_t0) / 60.0, 3),
                            "seconds_since_child_output": int(now - last_output_t),
                            "log_path": str(log_path),
                        },
                    )
                    last_output_t = now
                continue

            if item is None:
                break

            print(item, end="", flush=True)
            lf.write(item)
            lf.flush()
            last_output_t = time.time()

        rc = process.wait()
        reader.join(timeout=2.0)
        lf.write(f"\nRETURN_CODE: {rc}\n")
        write_json(
            live_path,
            {
                "stage": stage,
                "status": "finished",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_minutes": round((time.time() - stage_t0) / 60.0, 3),
                "return_code": int(rc),
                "log_path": str(log_path),
            },
        )
    if check and rc != 0:
        raise RuntimeError(f"{stage} failed with return code {rc}. See {log_path}")
    return rc


def run_eval_script(script_name, script_path, cwd, env=None):
    stage = f"eval_{script_name}"
    log("")
    log("=" * 80)
    log(f"RUN {stage}")
    log(f"CMD: {sys.executable} {script_path}")
    log(f"CWD: {cwd}")
    log("=" * 80)

    merged_env = os.environ.copy()
    merged_env["PYTHONUNBUFFERED"] = "1"
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path = LOG_DIR / f"{stage}.log"
    log_path.write_text(result.stdout + f"\nRETURN_CODE: {result.returncode}\n", encoding="utf-8", errors="replace")
    if result.returncode == 0:
        print(result.stdout, end="", flush=True)
    else:
        log(f"[skip] {script_name} did not complete cleanly; full output saved to {log_path}")
    return int(result.returncode)


def copy_any(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def elapsed_hours():
    return (time.time() - RUN_STARTED_WALL) / 3600.0


def warn_time(stage):
    log(f"[time] before {stage}: elapsed={elapsed_hours():.2f}h")


def find_bundle_zip():
    candidates = sorted(glob.glob(str(INPUT / "**" / "*.zip"), recursive=True))
    preferred = [Path(p) for p in candidates if "project_4_3_kaggle_input" in Path(p).name.lower()]
    for path in preferred:
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith("kaggle/notebook_cells/00_RUN_ALL_SINGLE_CELL.py"):
                        if RUNNER_VERSION in zf.read(name).decode("utf-8", errors="replace"):
                            return path
        except Exception:
            continue
    if preferred:
        return preferred[-1]
    for path in map(Path, candidates):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
            if any(name.startswith(EXPECTED_BUNDLE_NAME + "/") for name in names):
                return path
        except Exception:
            continue
    raise FileNotFoundError("Attach project_4_3_kaggle_input.zip as a Kaggle dataset input.")


def looks_like_bundle_root(path):
    path = Path(path)
    return (
        path.is_dir()
        and (path / "scripts" / "run_project43_pipeline.py").exists()
        and (path / "src" / "latent_node" / "main.py").exists()
        and (path / "src" / "fusion_baselines" / "train_baselines.py").exists()
    )


def bundle_has_current_runner(path):
    path = Path(path)
    runner_path = path / "kaggle" / "notebook_cells" / "00_RUN_ALL_SINGLE_CELL.py"
    if not runner_path.exists():
        return False
    try:
        return RUNNER_VERSION in runner_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False


def find_extracted_input_bundle():
    direct_candidates = sorted(INPUT.rglob(EXPECTED_BUNDLE_NAME))
    current = [candidate for candidate in direct_candidates if looks_like_bundle_root(candidate) and bundle_has_current_runner(candidate)]
    if current:
        return current[-1]
    for candidate in direct_candidates:
        if looks_like_bundle_root(candidate):
            return candidate

    script_candidates = sorted(INPUT.rglob("run_project43_pipeline.py"))
    current_roots = []
    for script_path in script_candidates:
        root = script_path.parent.parent
        if looks_like_bundle_root(root) and bundle_has_current_runner(root):
            current_roots.append(root)
    if current_roots:
        return current_roots[-1]
    for script_path in script_candidates:
        root = script_path.parent.parent
        if looks_like_bundle_root(root):
            return root

    for child in sorted(INPUT.glob("*")):
        if looks_like_bundle_root(child):
            return child
    return None


def get_bundle_root():
    roots = sorted(EXTRACT_ROOT.glob(EXPECTED_BUNDLE_NAME))
    if roots:
        return roots[0]
    roots = sorted(EXTRACT_ROOT.rglob(EXPECTED_BUNDLE_NAME))
    if roots:
        return roots[0]
    raise FileNotFoundError(f"Could not find extracted {EXPECTED_BUNDLE_NAME} under {EXTRACT_ROOT}")


def setup_extract():
    for import_name, pip_name in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("torchdiffeq", "torchdiffeq"),
        ("tqdm", "tqdm"),
    ]:
        ensure_pkg(import_name, pip_name)

    if get_bundle_root_or_none() is None:
        extracted_bundle = find_extracted_input_bundle()
        if extracted_bundle is not None:
            log(f"Input bundle already extracted: {extracted_bundle}")
            copy_any(extracted_bundle, EXTRACT_ROOT / EXPECTED_BUNDLE_NAME)
            source_info = {"source_kind": "extracted", "source_path": str(extracted_bundle)}
        else:
            bundle_zip = find_bundle_zip()
            log(f"Bundle zip: {bundle_zip}")
            EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bundle_zip, "r") as zf:
                zf.extractall(EXTRACT_ROOT)
            source_info = {"source_kind": "zip", "source_path": str(bundle_zip)}
    else:
        source_info = {"source_kind": "working", "source_path": str(get_bundle_root())}
    bundle_root = get_bundle_root()
    apply_runtime_hotfixes(bundle_root)
    log(f"Bundle root: {bundle_root}")
    write_json(
        ARTIFACT_ROOT / "run_environment.json",
        {
            "started_utc": RUN_STARTED_UTC,
            "runner_version": RUNNER_VERSION,
            "run_profile": RUN_PROFILE,
            "profile_cfg": PROFILE_CFG,
            **source_info,
            "bundle_root": str(bundle_root),
            "python": sys.version,
        },
    )
    return {**source_info, "bundle_root": str(bundle_root)}


def get_bundle_root_or_none():
    try:
        return get_bundle_root()
    except FileNotFoundError:
        return None


def apply_runtime_hotfixes(bundle_root):
    baseline_path = bundle_root / "src" / "fusion_baselines" / "build_real_dataset.py"
    if baseline_path.exists():
        text = baseline_path.read_text(encoding="utf-8")
        old = '    p = cfg.root.parent / "project_4" / "data" / "literature_validation" / "literature_validation.csv"\n    df = pd.read_csv(p)\n'
        new = '''    candidates = [
        cfg.root / "data" / "legacy_project4" / "literature_validation" / "literature_validation.csv",
        cfg.root / "data" / "literature_validation" / "literature_validation.csv",
        cfg.root.parent / "project_4" / "data" / "literature_validation" / "literature_validation.csv",
    ]
    p = next((path for path in candidates if path.exists()), None)
    if p is None:
        searched = "\\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find project_4 literature validation CSV. Searched:\\n{searched}")
    df = pd.read_csv(p)
'''
        if old in text:
            baseline_path.write_text(text.replace(old, new), encoding="utf-8")
            log(f"Applied baseline legacy-data path hotfix: {baseline_path}")


def data_counts_csvs(processed_dir):
    import pandas as pd

    processed_dir = Path(processed_dir)
    names = [
        "experimental_core_exact.csv",
        "experimental_digitized_with_uncertainty.csv",
        "experimental_candidates_needs_review.csv",
        "legacy_unverified_rows.csv",
        "master_rows_all.csv",
        "ml_ready_strict_nonlegacy.csv",
        "ml_ready_real_pool.csv",
        "transfer_real_weak_labels.csv",
        "ml_ready_with_transfer.csv",
        "ml_ready_with_transfer_plus_synth.csv",
    ]
    out = {}
    for name in names:
        path = processed_dir / name
        if path.exists():
            df = pd.read_csv(path)
            out[name] = {"rows": int(len(df)), "cols": int(df.shape[1])}
    return out


def baseline_all():
    bundle_root = get_bundle_root()
    cmd = [
        sys.executable,
        bundle_root / "scripts" / "run_project43_pipeline.py",
        "--n-synth",
        str(BASELINE_N_SYNTH),
        "--train-with-synth",
    ]
    run_cmd("baseline_all_ml_basic_nn", cmd, cwd=bundle_root)

    baseline_artifacts = ARTIFACT_ROOT / "01_baselines_ml_basic_nn"
    copy_any(bundle_root / "outputs", baseline_artifacts / "outputs")
    copy_any(bundle_root / "data" / "processed", baseline_artifacts / "data_processed")
    copy_any(bundle_root / "data" / "synthetic", baseline_artifacts / "data_synthetic")

    counts = data_counts_csvs(bundle_root / "data" / "processed")
    write_json(baseline_artifacts / "data_counts.json", counts)
    return {"data_counts": counts}


def prepare_latent_workspace(stage_name):
    bundle_root = get_bundle_root()
    workspace = WORKING / f"project43_{stage_name}"
    src_dst = workspace / "src"
    workspace_version = workspace / ".runner_version"
    if src_dst.exists() and workspace_version.exists() and workspace_version.read_text(encoding="utf-8").strip() != RUNNER_VERSION:
        shutil.rmtree(workspace)
    elif src_dst.exists() and not workspace_version.exists():
        shutil.rmtree(workspace)
    if not src_dst.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        copy_any(bundle_root / "src" / "latent_node", src_dst)
        copy_any(bundle_root / "data" / "legacy_project4", workspace / "data")
        copy_any(bundle_root / "legacy_kaggle_cells", workspace / "legacy_kaggle_cells")
        copy_any(bundle_root / "legacy_project4_scripts", workspace / "legacy_project4_scripts")
        workspace_version.write_text(RUNNER_VERSION, encoding="utf-8")
    return workspace


def latent_generate(stage_name, real_only=False):
    workspace = prepare_latent_workspace(stage_name)
    src_dir = workspace / "src"
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = [sys.executable, src_dir / "main.py", "--generate-data", "--device", device]
    if real_only:
        cmd.append("--real-only")
    run_cmd(f"{stage_name}_generate_data", cmd, cwd=src_dir, env=profile_env(stage_name))

    stage_artifacts = ARTIFACT_ROOT / stage_name
    copy_any(src_dir / "data" / "synthetic", stage_artifacts / "data_synthetic")
    copy_any(src_dir / "data" / "calphad_tables", stage_artifacts / "calphad_tables")
    copy_any(src_dir / "pipeline.log", stage_artifacts / "pipeline_generate.log")
    return {"workspace": str(workspace), "real_only": real_only}


def latent_train(stage_name, epochs, real_only=False):
    workspace = prepare_latent_workspace(stage_name)
    src_dir = workspace / "src"
    train_csv = src_dir / "data" / "synthetic" / "train.csv"
    val_csv = src_dir / "data" / "synthetic" / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"{stage_name} data not ready. Missing train.csv or val.csv.")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mirror_dir = ARTIFACT_ROOT / stage_name / "checkpoints"
    env = {
        "PHYSICSNODE_CHECKPOINT_MIRROR": mirror_dir,
        "PHYSICSNODE_BATCH_LOG_EVERY": os.environ.get("PHYSICSNODE_BATCH_LOG_EVERY", "1"),
        "PHYSICSNODE_BATCH_HEARTBEAT_SEC": os.environ.get("PHYSICSNODE_BATCH_HEARTBEAT_SEC", "30"),
        **profile_env(stage_name),
    }
    cmd = [
        sys.executable,
        src_dir / "main.py",
        "--train",
        "--epochs",
        str(epochs),
        "--device",
        device,
        "--resume-last",
    ]
    if real_only:
        cmd.append("--real-only")
    run_cmd(f"{stage_name}_train_{epochs}ep", cmd, cwd=src_dir, env=env)

    stage_artifacts = ARTIFACT_ROOT / stage_name
    copy_any(src_dir / "models", stage_artifacts / "models")
    copy_any(src_dir / "logs", stage_artifacts / "logs")
    copy_any(src_dir / "figures", stage_artifacts / "figures")
    copy_any(src_dir / "pipeline.log", stage_artifacts / "pipeline_train.log")
    return {"workspace": str(workspace), "epochs": epochs, "real_only": real_only}


def prepare_eval_checkpoints(chosen):
    chosen = Path(chosen)
    candidates = [
        chosen / "src" / "models" / "checkpoints" / "physics_node_best.pt",
        chosen / "src" / "models" / "checkpoints" / "physics_node_last.pt",
        chosen / "src" / "models" / "physics_node_best.pt",
        ARTIFACT_ROOT / "latent_stage2_real_only" / "checkpoints" / "physics_node_best.pt",
        ARTIFACT_ROOT / "latent_stage2_real_only" / "checkpoints" / "physics_node_last.pt",
        ARTIFACT_ROOT / "latent_stage1_full" / "checkpoints" / "physics_node_best.pt",
        ARTIFACT_ROOT / "latent_stage1_full" / "checkpoints" / "physics_node_last.pt",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        log("[skip] No PhysicsNODE checkpoint found for checkpoint-dependent evaluation scripts.")
        write_json(
            ARTIFACT_ROOT / "03_evaluation_export" / "checkpoint_aliases.json",
            {"status": "skipped", "searched": [str(path) for path in candidates]},
        )
        return False

    models_dir = chosen / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        "stage2_fixed_best.pt": models_dir / "stage2_fixed_best.pt",
        "stage2_extended_best.pt": models_dir / "stage2_extended_best.pt",
    }
    for alias_path in aliases.values():
        shutil.copy2(source, alias_path)
    write_json(
        ARTIFACT_ROOT / "03_evaluation_export" / "checkpoint_aliases.json",
        {"status": "ready", "source": str(source), "aliases": {name: str(path) for name, path in aliases.items()}},
    )
    log(f"Prepared evaluation checkpoint aliases from {source}")
    return True


def evaluate_export():
    workspaces = [
        WORKING / "project43_latent_stage2_real_only",
        WORKING / "project43_latent_stage1_full",
    ]
    chosen = next((w for w in workspaces if (w / "src").exists()), None)
    if chosen is None:
        raise FileNotFoundError("No latent workspace exists for evaluation/export.")

    checkpoint_ready = prepare_eval_checkpoints(chosen)
    legacy_eval_dir = chosen / "legacy_project4_scripts"
    eval_env = profile_env("evaluate_export")
    if "stage2_real_only" in chosen.name:
        eval_env["PROJECT43_DATA_REAL_ONLY"] = "1"
    for script_name in ["evaluate_comprehensive.py", "validate_calphad.py", "analysis.py", "ablation_study.py"]:
        if script_name in {"evaluate_comprehensive.py", "ablation_study.py"} and not checkpoint_ready:
            log(f"[skip] {script_name}: no compatible checkpoint alias is available.")
            continue
        if script_name == "validate_calphad.py" and importlib.util.find_spec("pycalphad") is None:
            log("[skip] validate_calphad.py: optional pycalphad package is not installed.")
            continue
        src = legacy_eval_dir / script_name
        dst = chosen / script_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        if dst.exists():
            run_eval_script(script_name, dst, cwd=chosen, env=eval_env)

    export_artifacts = ARTIFACT_ROOT / "03_evaluation_export"
    copy_any(chosen / "outputs", export_artifacts / "outputs")
    copy_any(chosen / "analysis_results", export_artifacts / "analysis_results")
    copy_any(chosen / "figures", export_artifacts / "figures")
    return {"workspace": str(chosen)}


def final_summary():
    summary = {
        "started_utc": RUN_STARTED_UTC,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": round(elapsed_hours(), 4),
        "artifact_root": str(ARTIFACT_ROOT),
        "snapshot_zip": str(SNAPSHOT_ZIP),
        "done_stages": sorted(p.stem.replace(".done", "") for p in STATE_DIR.glob("*.done.json")),
        "failed_stages": sorted(p.stem.replace(".failed", "") for p in STATE_DIR.glob("*.failed.json")),
        "continue_on_stage_error": CONTINUE_ON_STAGE_ERROR,
    }
    write_json(ARTIFACT_ROOT / "FINAL_RUN_SUMMARY.json", summary)
    snapshot_artifacts()
    log("")
    log("=" * 80)
    log("PROJECT 4.3 SINGLE-CELL RUN FINISHED")
    log(json.dumps(summary, indent=2))
    log("=" * 80)


def run_stage(stage, fn):
    if is_done(stage):
        log(f"[skip] {stage} already done.")
        return
    warn_time(stage)
    try:
        payload = fn() or {}
        mark_done(stage, payload)
        snapshot_artifacts()
    except Exception as exc:
        mark_failed(stage, exc)
        snapshot_artifacts()
        log(f"[failed] {stage}: {exc}")
        if not CONTINUE_ON_STAGE_ERROR:
            raise


def require_done(*stages):
    missing = [stage for stage in stages if not is_done(stage)]
    if missing:
        raise RuntimeError("Required stages are not done: " + ", ".join(missing))


log("=" * 80)
log("PROJECT 4.3 KAGGLE SINGLE-CELL FULL RUN")
log("=" * 80)
log(f"Started UTC: {RUN_STARTED_UTC}")
log(f"Artifact root: {ARTIFACT_ROOT}")
log(f"Continue on stage error: {CONTINUE_ON_STAGE_ERROR}")
log(f"Run profile: {RUN_PROFILE}")
log(
    f"Profile params | baseline_n_synth={BASELINE_N_SYNTH} "
    f"| stage1_epochs={STAGE1_EPOCHS} | stage2_epochs={STAGE2_EPOCHS}"
)
log(
    "Model params | batch=%s amp=%s adjoint=%s sn=%s rtol=%s atol=%s max_steps=%s time=%s"
    % (
        profile_env().get("PROJECT43_MODEL_BATCH_SIZE"),
        profile_env().get("PROJECT43_MODEL_USE_AMP"),
        profile_env().get("PROJECT43_MODEL_ADJOINT"),
        profile_env().get("PROJECT43_MODEL_USE_SN"),
        profile_env().get("PROJECT43_MODEL_RTOL"),
        profile_env().get("PROJECT43_MODEL_ATOL"),
        profile_env().get("PROJECT43_MODEL_MAX_STEPS"),
        profile_env().get("PROJECT43_ODE_TIME_TRANSFORM"),
    )
)

run_stage("00_setup_extract", setup_extract)
run_stage("01_baselines_ml_basic_nn", lambda: (require_done("00_setup_extract"), baseline_all())[1])
run_stage("02_latent_stage1_generate_full", lambda: (require_done("00_setup_extract"), latent_generate("latent_stage1_full", real_only=False))[1])
run_stage(
    f"03_latent_stage1_train_{STAGE1_EPOCHS}ep",
    lambda: (require_done("02_latent_stage1_generate_full"), latent_train("latent_stage1_full", epochs=STAGE1_EPOCHS, real_only=False))[1],
)
run_stage("04_latent_stage2_generate_real_only", lambda: (require_done("00_setup_extract"), latent_generate("latent_stage2_real_only", real_only=True))[1])
run_stage(
    f"05_latent_stage2_train_{STAGE2_EPOCHS}ep_real_only",
    lambda: (require_done("04_latent_stage2_generate_real_only"), latent_train("latent_stage2_real_only", epochs=STAGE2_EPOCHS, real_only=True))[1],
)
run_stage(
    "06_evaluate_export",
    lambda: (
        require_done(f"03_latent_stage1_train_{STAGE1_EPOCHS}ep", f"05_latent_stage2_train_{STAGE2_EPOCHS}ep_real_only"),
        evaluate_export(),
    )[1],
)
final_summary()
