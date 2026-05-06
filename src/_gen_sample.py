"""Create a small preview sample of the synthetic dataset for git."""
import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

for data_dir in ['data/synthetic', 'src/data/synthetic']:
    full_path = os.path.join(os.path.dirname(__file__), data_dir, 'synthetic_kinetics.csv')
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        # Take 10 curves per provenance type
        samples = []
        for prov in df['provenance'].unique():
            sub = df[df['provenance'] == prov]
            ids = sub['sample_id'].unique()[:10]
            samples.append(sub[sub['sample_id'].isin(ids)])
        sample = pd.concat(samples, ignore_index=True)
        out_path = os.path.join(os.path.dirname(full_path), 'synthetic_sample_preview.csv')
        sample.to_csv(out_path, index=False)
        print(f"Created {out_path}: {len(sample)} rows, {sample['sample_id'].nunique()} curves")
        print(f"Provenance: {dict(sample['provenance'].value_counts())}")
        break
