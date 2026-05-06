from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "kaggle" / "input_package" / "project_4_3_bundle"
BUILD_DIR = ROOT / "kaggle" / "input_build"
UPLOAD_DIR = ROOT / "kaggle" / "upload"


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
        ),
    )


def copy_tree_slim(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "models",
            "checkpoints",
            "figures",
            "outputs",
            "logs",
            "synthetic",
        ),
    )


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_package() -> Path:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    copy_tree_slim(ROOT / "src", PACKAGE_DIR / "src")
    copy_tree(ROOT / "scripts", PACKAGE_DIR / "scripts")
    copy_tree(ROOT / "data" / "raw" / "csv", PACKAGE_DIR / "data" / "raw" / "csv")
    copy_tree(ROOT / "data" / "processed", PACKAGE_DIR / "data" / "processed")
    legacy_data = ROOT / "data" / "legacy_project4"
    if legacy_data.exists():
        lit = legacy_data / "literature_validation"
        calphad = legacy_data / "calphad_tables"
        if lit.exists():
            copy_tree(lit, PACKAGE_DIR / "data" / "legacy_project4" / "literature_validation")
        if calphad.exists():
            copy_tree(calphad, PACKAGE_DIR / "data" / "legacy_project4" / "calphad_tables")
    copy_tree(ROOT / "docs" / "prompts", PACKAGE_DIR / "docs" / "prompts")
    copy_tree(ROOT / "docs" / "reports", PACKAGE_DIR / "docs" / "reports")
    copy_tree(ROOT / "notebooks", PACKAGE_DIR / "notebooks")
    copy_tree(ROOT / "tests", PACKAGE_DIR / "tests")
    copy_tree(ROOT / "kaggle" / "notebook_cells", PACKAGE_DIR / "kaggle" / "notebook_cells")

    # Pull in the battle-tested legacy Kaggle cells as references.
    legacy_cells_src = ROOT / "references" / "project_4_original" / "kaggle" / "cells"
    if legacy_cells_src.exists():
        copy_tree(legacy_cells_src, PACKAGE_DIR / "legacy_kaggle_cells")
    legacy_eval_dir = ROOT / "references" / "project_4_original"
    (PACKAGE_DIR / "legacy_project4_scripts").mkdir(parents=True, exist_ok=True)
    for file_name in ["analysis.py", "evaluate_comprehensive.py", "validate_calphad.py", "ablation_study.py"]:
        src = legacy_eval_dir / file_name
        if src.exists():
            copy_file(src, PACKAGE_DIR / "legacy_project4_scripts" / file_name)

    for file_name in ["README.md", "KAGGLE_README.md", "requirements-kaggle.txt"]:
        copy_file(ROOT / file_name, PACKAGE_DIR / file_name)

    manifest = {
        "package_root": str(PACKAGE_DIR),
        "includes": [
            "src",
            "scripts",
            "data/raw/csv",
            "data/processed",
            "data/legacy_project4/literature_validation",
            "data/legacy_project4/calphad_tables",
            "docs/prompts",
            "docs/reports",
            "notebooks",
            "tests",
            "kaggle/notebook_cells",
            "legacy_kaggle_cells",
            "legacy_project4_scripts",
        ],
    }
    with (PACKAGE_DIR / "bundle_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return PACKAGE_DIR


def build_zip(package_dir: Path) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BUILD_DIR / "project_4_3_kaggle_input.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_dir():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            arcname = path.relative_to(package_dir.parent).as_posix()
            zf.write(path, arcname=arcname)

    upload_copy = UPLOAD_DIR / zip_path.name
    shutil.copy2(zip_path, upload_copy)
    return zip_path


def main() -> None:
    package_dir = build_package()
    zip_path = build_zip(package_dir)
    print(json.dumps({"package_dir": str(package_dir), "zip_path": str(zip_path)}, indent=2))


if __name__ == "__main__":
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    main()
