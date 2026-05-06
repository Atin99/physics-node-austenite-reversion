from __future__ import annotations

import argparse
import json
from pathlib import Path

from src42.build_real_dataset import build_real_dataset, export_splits
from src42.config import Project42Config, ensure_dirs
from src42.generate_physics_surrogate import generate_surrogate_rows
from src42.io_helpers import import_pandas
from src42.train_baselines import run_training


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end project_4.2 data + baseline pipeline.")
    parser.add_argument("--n-synth", type=int, default=5000, help="Synthetic-surrogate rows to generate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-with-synth",
        action="store_true",
        help="Also run LOSO baselines on the real+transfer+synthetic combined pool.",
    )
    args = parser.parse_args()

    cfg = Project42Config()
    ensure_dirs(cfg)
    pd = import_pandas(cfg.root)

    master_df = build_real_dataset(cfg)
    split_summary = export_splits(master_df, cfg)

    real_pool = pd.read_csv(cfg.processed_dir / "ml_ready_real_pool.csv")
    synth_df = generate_surrogate_rows(real_pool, n_rows=args.n_synth, seed=args.seed, pd=pd)
    synth_path = cfg.synthetic_dir / "synthetic_calphad_surrogate.csv"
    synth_df.to_csv(synth_path, index=False)

    transfer_pool = pd.read_csv(cfg.processed_dir / "ml_ready_with_transfer.csv")
    shared_cols = [c for c in transfer_pool.columns if c in synth_df.columns]
    combined = pd.concat([transfer_pool[shared_cols], synth_df[shared_cols]], ignore_index=True)
    combined_path = cfg.processed_dir / "ml_ready_with_transfer_plus_synth.csv"
    combined.to_csv(combined_path, index=False)

    real_summary = run_training(cfg.processed_dir / "ml_ready_real_pool.csv", out_prefix="scenario_real_only", seed=args.seed)
    transfer_summary = run_training(
        cfg.processed_dir / "ml_ready_with_transfer.csv",
        out_prefix="scenario_real_plus_transfer",
        seed=args.seed,
    )

    synth_summary = None
    if args.train_with_synth:
        synth_summary = run_training(
            combined_path,
            out_prefix="scenario_real_transfer_synth",
            seed=args.seed,
        )

    summary = {
        "split_summary": split_summary,
        "synthetic_rows_generated": int(len(synth_df)),
        "synthetic_csv": str(synth_path),
        "combined_csv": str(combined_path),
        "baseline_runs": {
            "real_only": real_summary,
            "real_plus_transfer": transfer_summary,
            "real_transfer_synth": synth_summary,
        },
    }
    out = cfg.outputs_dir / "pipeline_run_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

