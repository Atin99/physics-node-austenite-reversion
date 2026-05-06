from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILD = ROOT / "scripts" / "build_kaggle_bundle.py"
HIFI_DIR = ROOT / "kaggle_20h_hifi"
PACKAGE_PARENT = HIFI_DIR / "input_package"
PACKAGE_DIR = PACKAGE_PARENT / "project_4_3_bundle"
BUILD_DIR = HIFI_DIR / "input_build"
UPLOAD_DIR = HIFI_DIR / "upload"
ZIP_NAME = "project_4_3_hifi20_kaggle_input.zip"


def load_base_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("base_kaggle_builder", BASE_BUILD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_BUILD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zip_package(package_dir: Path) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BUILD_DIR / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            arcname = path.relative_to(package_dir.parent).as_posix()
            zf.write(path, arcname=arcname)

    shutil.copy2(zip_path, UPLOAD_DIR / ZIP_NAME)
    return zip_path


def main() -> None:
    base = load_base_builder()
    base.PACKAGE_DIR = PACKAGE_DIR
    base.BUILD_DIR = BUILD_DIR
    base.UPLOAD_DIR = UPLOAD_DIR
    package_dir = base.build_package()

    hifi_cell = HIFI_DIR / "notebook_cells" / "00_RUN_HIFI20_SINGLE_CELL.py"
    dst = package_dir / "kaggle_20h_hifi" / "notebook_cells" / hifi_cell.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hifi_cell, dst)
    readme = HIFI_DIR / "README_HIFI20.md"
    shutil.copy2(readme, package_dir / "README_HIFI20.md")

    zip_path = zip_package(package_dir)
    print(json.dumps({"package_dir": str(package_dir), "zip_path": str(zip_path)}, indent=2))


if __name__ == "__main__":
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    main()
