from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Iterable


def import_pandas(project_root: Path):
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        pydeps_candidates = [
            project_root / "_pydeps",
            project_root / "references" / "project_4_2_original" / "_pydeps",
        ]
        for pydeps in pydeps_candidates:
            if pydeps.exists():
                sys.path.insert(0, str(pydeps))
        import pandas as pd  # type: ignore
    return pd


def load_csv_robust(path: Path, pd):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected = len(header)
        rows = []
        for row in reader:
            if len(row) > expected:
                row = row[: expected - 1] + [",".join(row[expected - 1 :])]
            elif len(row) < expected:
                row = row + [""] * (expected - len(row))
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def to_float(value) -> float:
    if value is None:
        return math.nan
    s = str(value).strip()
    if not s or s.lower() in {"na", "nan", "nd", "-", "--", "none"}:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_columns(df, columns: Iterable[str], default=""):
    for col in columns:
        if col not in df.columns:
            df[col] = default
    return df
