from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

from .config import Project42Config, ensure_dirs
from .io_helpers import ensure_columns, import_pandas, load_csv_robust, normalize_text, to_float


STD_COLUMNS = [
    "Study",
    "Year",
    "DOI_or_ID",
    "Source_URL",
    "Source_Access",
    "Source_File",
    "Inclusion_Tier",
    "Reliability_Tier",
    "Mn_wt",
    "C_wt",
    "Al_wt",
    "Si_wt",
    "Cu_wt",
    "Ni_wt",
    "Cr_wt",
    "Mo_wt",
    "Nb_wt",
    "V_wt",
    "T_anneal_C",
    "Time_sec",
    "RA_fraction",
    "RA_pct",
    "Measurement_Method",
    "Data_Quality",
    "Initial_Condition",
    "Source_Ref",
    "Extraction_Note",
    "ML_Ready",
    "Row_Key",
]


def _load_open_access(cfg: Project42Config, pd):
    df = load_csv_robust(cfg.raw_dir / "csv" / "open_access_verified_additions.csv", pd)
    df["Source_File"] = "open_access_verified_additions.csv"
    df = ensure_columns(df, ["Mo_wt", "Nb_wt", "V_wt"], default="")
    return df


def _load_transfer(cfg: Project42Config, pd):
    df = load_csv_robust(cfg.raw_dir / "csv" / "transfer_real_ra_stability.csv", pd)
    df["Source_File"] = "transfer_real_ra_stability.csv"
    df = ensure_columns(df, ["Cu_wt", "Ni_wt", "Cr_wt", "V_wt", "Initial_Condition"], default="")
    # Keep process path in Initial_Condition when available.
    if "T_path_C" in df.columns:
        df["Initial_Condition"] = df["Initial_Condition"].where(
            df["Initial_Condition"].astype(str).str.strip() != "",
            df["T_path_C"],
        )
    # Hold_T_C is used as annealing temperature for model-ready rows.
    if "Hold_T_C" in df.columns:
        df["T_anneal_C"] = df["Hold_T_C"]
    elif "T_path_C" in df.columns:
        df["T_anneal_C"] = df["T_path_C"]
    return df


def _load_thesis(cfg: Project42Config, pd):
    df = load_csv_robust(cfg.raw_dir / "csv" / "thesis_pdf_mined_real_ra.csv", pd)
    df["Source_File"] = "thesis_pdf_mined_real_ra.csv"
    if "Source_Type" in df.columns:
        df["Source_Access"] = df["Source_Type"]
    df = ensure_columns(df, ["Cu_wt", "Source_Access"], default="")
    return df


def _load_project4_legacy(cfg: Project42Config, pd):
    candidates = [
        cfg.root / "data" / "legacy_project4" / "literature_validation" / "literature_validation.csv",
        cfg.root / "data" / "literature_validation" / "literature_validation.csv",
        cfg.root.parent / "project_4" / "data" / "literature_validation" / "literature_validation.csv",
    ]
    p = next((path for path in candidates if path.exists()), None)
    if p is None:
        searched = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find project_4 literature validation CSV. Searched:\n{searched}")
    df = pd.read_csv(p)
    rename_map = {
        "study_id": "Study",
        "doi": "DOI_or_ID",
        "T_celsius": "T_anneal_C",
        "t_seconds": "Time_sec",
        "f_RA": "RA_fraction",
        "f_RA_pct": "RA_pct",
        "method": "Measurement_Method",
        "data_quality": "Data_Quality",
        "initial_condition": "Initial_Condition",
        "source_ref": "Source_Ref",
        "provenance": "Source_Access",
    }
    df = df.rename(columns=rename_map)
    df["Source_File"] = "project4_literature_validation.csv"
    df["Inclusion_Tier"] = "legacy_project4"
    df["Source_URL"] = ""
    df["Extraction_Note"] = df.get("journal", "").fillna("").astype(str)
    for col in ["Mo", "Nb", "Mn", "C", "Al", "Si"]:
        if col in df.columns:
            df[f"{col}_wt"] = df[col]
    df = ensure_columns(df, ["Cu_wt", "Ni_wt", "Cr_wt", "V_wt", "Mo_wt", "Nb_wt"], default="")
    return df


def _classify_reliability(row: Dict[str, object]) -> str:
    src = normalize_text(row.get("Source_File")).lower()
    tier = normalize_text(row.get("Inclusion_Tier")).lower()
    dq = normalize_text(row.get("Data_Quality")).lower()
    src_ref = normalize_text(row.get("Source_Ref")).lower()
    year = to_float(row.get("Year"))

    if src == "project4_literature_validation.csv":
        if "search summary" in src_ref or (not math.isnan(year) and year > 2026):
            return "legacy_unverified"
        if "digitized" in dq:
            return "experimental_digitized"
        if dq in {"table", "text_reported"}:
            return "legacy_core_exact"
        return "legacy_unverified"

    if tier.startswith("candidate") or "partial" in tier:
        return "candidate_review"
    if "approx" in dq or "discrepant" in dq or "abstract" in dq:
        return "candidate_review"
    if tier.startswith("transfer_") or tier.startswith("secondary_"):
        return "transfer_or_secondary"
    if "digitized" in dq:
        return "experimental_digitized"
    if tier.startswith("thesis_"):
        return "experimental_core_exact"
    if tier == "core_exact":
        return "experimental_core_exact"
    if dq in {"table", "text_reported", "table_pdf_mined", "table_thesis"}:
        return "experimental_core_exact"
    return "unknown_review"


def _ml_ready(row: Dict[str, object]) -> bool:
    mn = to_float(row.get("Mn_wt"))
    c = to_float(row.get("C_wt"))
    t = to_float(row.get("T_anneal_C"))
    tm = to_float(row.get("Time_sec"))
    ra = to_float(row.get("RA_fraction"))
    if any(math.isnan(v) for v in [mn, c, t, tm, ra]):
        return False
    if not (0.0 <= ra <= 1.0):
        return False
    if not (0.0 <= mn <= 30.0 and 0.0 <= c <= 2.0):
        return False
    if not (0.0 <= t <= 1400.0 and tm >= 0.0):
        return False
    return True


def _row_key(row: Dict[str, object]) -> str:
    def r(v, d):
        f = to_float(v)
        return "nan" if math.isnan(f) else f"{round(f, d)}"

    study = normalize_text(row.get("Study")).lower()
    method = normalize_text(row.get("Measurement_Method")).lower()
    return "|".join(
        [
            study,
            r(row.get("Mn_wt"), 4),
            r(row.get("C_wt"), 4),
            r(row.get("Al_wt"), 4),
            r(row.get("Si_wt"), 4),
            r(row.get("T_anneal_C"), 3),
            r(row.get("Time_sec"), 3),
            r(row.get("RA_fraction"), 5),
            method,
        ]
    )


def _finalize(df, pd):
    keep = [
        "Study",
        "Year",
        "DOI_or_ID",
        "Source_URL",
        "Source_Access",
        "Source_File",
        "Inclusion_Tier",
        "Mn_wt",
        "C_wt",
        "Al_wt",
        "Si_wt",
        "Cu_wt",
        "Ni_wt",
        "Cr_wt",
        "Mo_wt",
        "Nb_wt",
        "V_wt",
        "T_anneal_C",
        "Time_sec",
        "RA_fraction",
        "RA_pct",
        "Measurement_Method",
        "Data_Quality",
        "Initial_Condition",
        "Source_Ref",
        "Extraction_Note",
    ]
    df = ensure_columns(df, keep, default="")
    df = df[keep].copy()

    numeric_cols = [
        "Year",
        "Mn_wt",
        "C_wt",
        "Al_wt",
        "Si_wt",
        "Cu_wt",
        "Ni_wt",
        "Cr_wt",
        "Mo_wt",
        "Nb_wt",
        "V_wt",
        "T_anneal_C",
        "Time_sec",
        "RA_fraction",
        "RA_pct",
    ]
    for col in numeric_cols:
        df[col] = df[col].map(to_float)

    missing_frac = df["RA_fraction"].isna() & df["RA_pct"].notna()
    df.loc[missing_frac, "RA_fraction"] = df.loc[missing_frac, "RA_pct"] / 100.0
    missing_pct = df["RA_pct"].isna() & df["RA_fraction"].notna()
    df.loc[missing_pct, "RA_pct"] = df.loc[missing_pct, "RA_fraction"] * 100.0

    df["Reliability_Tier"] = df.apply(_classify_reliability, axis=1)
    df["ML_Ready"] = df.apply(_ml_ready, axis=1)
    df["Row_Key"] = df.apply(_row_key, axis=1)
    df = df.drop_duplicates(subset=["Row_Key"], keep="first")
    df = ensure_columns(df, STD_COLUMNS, default="")
    return df[STD_COLUMNS].copy()


def build_real_dataset(cfg: Project42Config):
    pd = import_pandas(cfg.root)
    frames = [
        _load_open_access(cfg, pd),
        _load_transfer(cfg, pd),
        _load_thesis(cfg, pd),
        _load_project4_legacy(cfg, pd),
    ]
    df = _finalize(pd.concat(frames, ignore_index=True), pd)
    return df


def export_splits(df, cfg: Project42Config) -> Dict[str, int]:
    pd = import_pandas(cfg.root)
    out = cfg.processed_dir
    core = df[df["Reliability_Tier"].isin({"experimental_core_exact", "legacy_core_exact"}) & df["ML_Ready"]].copy()
    digitized = df[(df["Reliability_Tier"] == "experimental_digitized") & df["ML_Ready"]].copy()
    transfer = df[(df["Reliability_Tier"] == "transfer_or_secondary") & df["ML_Ready"]].copy()
    candidates = df[(df["Reliability_Tier"].isin({"candidate_review", "unknown_review"})) | (~df["ML_Ready"])].copy()
    legacy_unverified = df[df["Reliability_Tier"] == "legacy_unverified"].copy()

    ml_ready_real = pd.concat([core, digitized], ignore_index=True).drop_duplicates(subset=["Row_Key"])
    ml_ready_with_transfer = (
        pd.concat([core, digitized, transfer], ignore_index=True).drop_duplicates(subset=["Row_Key"])
    )
    strict_nonlegacy = ml_ready_real[ml_ready_real["Source_File"] != "project4_literature_validation.csv"].copy()

    df.to_csv(out / "master_rows_all.csv", index=False)
    core.to_csv(out / "experimental_core_exact.csv", index=False)
    digitized.to_csv(out / "experimental_digitized_with_uncertainty.csv", index=False)
    transfer.to_csv(out / "transfer_real_weak_labels.csv", index=False)
    candidates.to_csv(out / "experimental_candidates_needs_review.csv", index=False)
    legacy_unverified.to_csv(out / "legacy_unverified_rows.csv", index=False)
    ml_ready_real.to_csv(out / "ml_ready_real_pool.csv", index=False)
    ml_ready_with_transfer.to_csv(out / "ml_ready_with_transfer.csv", index=False)
    strict_nonlegacy.to_csv(out / "ml_ready_strict_nonlegacy.csv", index=False)

    summary = {
        "master_rows_all": int(len(df)),
        "experimental_core_exact": int(len(core)),
        "experimental_digitized": int(len(digitized)),
        "transfer_real_weak_labels": int(len(transfer)),
        "candidate_or_not_ready": int(len(candidates)),
        "legacy_unverified": int(len(legacy_unverified)),
        "ml_ready_real_pool": int(len(ml_ready_real)),
        "ml_ready_with_transfer": int(len(ml_ready_with_transfer)),
        "ml_ready_strict_nonlegacy": int(len(strict_nonlegacy)),
        "unique_studies_master": int(df["Study"].nunique()),
        "unique_studies_ml_ready_real": int(ml_ready_real["Study"].nunique()),
    }
    with (out / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    cfg = Project42Config()
    ensure_dirs(cfg)
    global pd
    pd = import_pandas(cfg.root)
    df = build_real_dataset(cfg)
    summary = export_splits(df, cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
