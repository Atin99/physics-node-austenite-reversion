"""One-off script to regenerate and save synthetic data locally."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from data_generator import (
    build_full_dataset, build_real_dataset,
    build_calibrated_synthetic, generate_exploratory_synthetic,
    save_synthetic_data, prepare_train_val_test_split,
    plot_synthetic_curves
)

cfg = get_config()
print("Building full 3-tier dataset...")
df = build_full_dataset(cfg)

# Save the full combined dataset
save_synthetic_data(df, cfg)

# Also save individual splits
train_df, val_df, test_df = prepare_train_val_test_split(df, cfg)
train_df.to_csv(cfg.synthetic_dir / "train.csv", index=False)
val_df.to_csv(cfg.synthetic_dir / "val.csv", index=False)
test_df.to_csv(cfg.synthetic_dir / "test.csv", index=False)

# Full dataset too
df.to_csv(cfg.synthetic_dir / "full_dataset.csv", index=False)

# Plot overview
try:
    plot_synthetic_curves(df, n_show=10, config=cfg)
    print("Saved data overview figure.")
except Exception as e:
    print(f"Plot failed (non-critical): {e}")

# Summary
print(f"\n{'='*60}")
print(f"Total rows: {len(df)}")
print(f"Unique curves: {df['sample_id'].nunique()}")
print(f"\nProvenance breakdown:")
print(df['provenance'].value_counts().to_string())
print(f"\nTrain: {len(train_df)} rows, {train_df['sample_id'].nunique()} curves")
print(f"Val:   {len(val_df)} rows, {val_df['sample_id'].nunique()} curves")
print(f"Test:  {len(test_df)} rows, {test_df['sample_id'].nunique()} curves")
print(f"\nFiles saved to: {cfg.synthetic_dir}")
for f in sorted(cfg.synthetic_dir.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
print(f"{'='*60}")
