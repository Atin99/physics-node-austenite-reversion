from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np

from .config import Project42Config, ensure_dirs
from .io_helpers import import_pandas


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ac1_proxy(mn, c, al, si, ni=0.0, cr=0.0):
    # Empirical proxy used only for synthetic-surrogate labeling.
    ac1 = 723.0 - 10.7 * mn + 29.1 * si + 16.9 * ni + 16.9 * cr + 20.0 * al - 35.0 * c
    return np.clip(ac1, 450.0, 900.0)


def generate_surrogate_rows(real_df, n_rows: int, seed: int, pd):
    if len(real_df) < 5:
        raise ValueError("Need at least 5 real rows to bootstrap synthetic-surrogate sampling.")

    rng = np.random.default_rng(seed)
    numeric_cols = ["Mn_wt", "C_wt", "Al_wt", "Si_wt", "Cu_wt", "Ni_wt", "Cr_wt", "Mo_wt", "Nb_wt", "V_wt"]

    base = real_df.sample(n=n_rows, replace=True, random_state=seed).reset_index(drop=True).copy()
    for col in numeric_cols:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = base[col].fillna(0.0).astype(float)

    # Composition perturbation around observed data manifold.
    for col, sigma in [("Mn_wt", 0.22), ("C_wt", 0.015), ("Al_wt", 0.09), ("Si_wt", 0.06)]:
        base[col] = np.clip(base[col] + rng.normal(0.0, sigma, size=n_rows), 0.0, None)

    t_min = max(1.0, float(np.nanpercentile(real_df["Time_sec"], 2)))
    t_max = float(np.nanpercentile(real_df["Time_sec"], 98))
    if t_max <= t_min:
        t_max = t_min * 50.0
    log_t = rng.uniform(np.log(t_min), np.log(t_max), size=n_rows)
    base["Time_sec"] = np.exp(log_t)

    t_med = float(np.nanmedian(real_df["T_anneal_C"]))
    t_std = float(np.nanstd(real_df["T_anneal_C"]))
    if t_std < 20:
        t_std = 35.0
    base["T_anneal_C"] = np.clip(rng.normal(loc=t_med, scale=t_std, size=n_rows), 450.0, 900.0)

    mn = base["Mn_wt"].to_numpy()
    c = base["C_wt"].to_numpy()
    al = base["Al_wt"].to_numpy()
    si = base["Si_wt"].to_numpy()
    ni = base["Ni_wt"].to_numpy()
    cr = base["Cr_wt"].to_numpy()
    t = base["T_anneal_C"].to_numpy()
    time_s = base["Time_sec"].to_numpy()

    ac1 = _ac1_proxy(mn, c, al, si, ni=ni, cr=cr)
    theta = (t - ac1) / 35.0
    f_eq = np.clip(_sigmoid(theta) * (0.30 + 0.35 * np.clip(c, 0, 0.6) + 0.03 * np.clip(mn, 0, 12)), 0.02, 0.95)

    tau = np.exp(6.2 + 12500.0 / (t + 273.15) + 0.06 * mn - 2.2 * c - 0.15 * al)
    n_exp = np.clip(1.25 + 1.0 * _sigmoid((t - ac1) / 70.0) + rng.normal(0.0, 0.08, size=n_rows), 1.05, 3.20)
    ra = f_eq * (1.0 - np.exp(-np.power(np.maximum(time_s, 1.0) / np.maximum(tau, 1.0), n_exp)))
    ra = np.clip(ra + rng.normal(0.0, 0.015, size=n_rows), 0.0, 0.98)

    synth = base.copy()
    synth["Study"] = "synthetic_calphad_surrogate_pool"
    synth["Synthetic_ID"] = [f"syn_{i:05d}" for i in range(1, n_rows + 1)]
    synth["Year"] = date.today().year
    synth["DOI_or_ID"] = "synthetic_calphad_surrogate"
    synth["Source_URL"] = ""
    synth["Source_Access"] = "generated_local_surrogate"
    synth["Source_File"] = "generated_physics_surrogate.csv"
    synth["Inclusion_Tier"] = "synthetic_calphad_surrogate"
    synth["Reliability_Tier"] = "synthetic_calphad_surrogate"
    synth["RA_fraction"] = ra
    synth["RA_pct"] = ra * 100.0
    synth["Measurement_Method"] = "synthetic_surrogate"
    synth["Data_Quality"] = "generated_physics_surrogate"
    synth["Initial_Condition"] = "synthetic_surrogate"
    synth["Source_Ref"] = "two_stage_jmak_like_proxy"
    synth["Extraction_Note"] = (
        "Synthetic surrogate generated from empirical Ac1 proxy and two-stage time evolution. "
        "Not experimental."
    )
    synth["ML_Ready"] = True
    synth["Ac1_proxy_C"] = ac1
    synth["f_eq_proxy"] = f_eq
    synth["n_proxy"] = n_exp
    synth["tau_proxy_sec"] = tau
    return synth


def main():
    parser = argparse.ArgumentParser(description="Generate labeled physics-surrogate synthetic dataset.")
    parser.add_argument("--n-rows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Project42Config()
    ensure_dirs(cfg)
    pd = import_pandas(cfg.root)

    real_path = cfg.processed_dir / "ml_ready_real_pool.csv"
    if not real_path.exists():
        raise FileNotFoundError(f"Missing {real_path}. Run build_real_dataset.py first.")
    real_df = pd.read_csv(real_path)
    synth_df = generate_surrogate_rows(real_df, n_rows=args.n_rows, seed=args.seed, pd=pd)

    out_csv = cfg.synthetic_dir / "synthetic_calphad_surrogate.csv"
    synth_df.to_csv(out_csv, index=False)

    summary = {
        "synthetic_rows": int(len(synth_df)),
        "real_bootstrap_rows": int(len(real_df)),
        "out_csv": str(out_csv),
    }
    with (cfg.synthetic_dir / "synthetic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
