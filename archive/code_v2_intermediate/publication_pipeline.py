import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_config
from explainability import run_explainability_suite
from model import PhysicsNODE
from symbolic_regression import extract_symbolic_equation
from trainer import AusteniteReversionDataset
from visualizations import generate_all_figures

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, config):
    model = PhysicsNODE(config.model).to(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def select_representative_ids(df: pd.DataFrame, n_curves: int = 3):
    endpoints = df.sort_values('t_seconds').groupby('sample_id').tail(1)
    chosen = []
    for provenance in ['experimental', 'synthetic_calibrated', 'synthetic_exploratory']:
        subset = endpoints[endpoints['provenance'] == provenance]
        if len(subset) == 0:
            continue
        chosen.append(int(subset.nlargest(1, 'f_RA').iloc[0]['sample_id']))
        if len(chosen) >= n_curves:
            return chosen[:n_curves]
    if len(chosen) < n_curves:
        extras = endpoints[~endpoints['sample_id'].isin(chosen)].nlargest(n_curves - len(chosen), 'f_RA')
        chosen.extend(int(v) for v in extras['sample_id'].tolist())
    return chosen[:n_curves]


def build_prediction_panels(model, dataset, sample_ids, config, n_samples):
    predictions = {}
    by_id = {sample['sample_id']: sample for sample in dataset.samples}
    for sample_id in sample_ids:
        sample = by_id[sample_id]
        static = sample['static'].unsqueeze(0).to(config.device)
        f_eq = sample['f_eq'].view(1, 1).to(config.device)
        dG = sample['dG_norm'].view(1, 1).to(config.device)
        t_span = sample['t_span'].to(config.device)
        with torch.no_grad():
            mean, lower, upper = model.predict_with_uncertainty(static, f_eq, dG, t_span, n_samples=n_samples)
        label = f"{sample['provenance']} | {sample['T_celsius']:.0f}C | id={sample_id}"
        predictions[label] = {
            't_hours': (sample['t_span'].cpu().numpy() / 3600.0),
            'f_RA_mean': mean[0].cpu().numpy(),
            'f_RA_lower': lower[0].cpu().numpy(),
            'f_RA_upper': upper[0].cpu().numpy(),
            'f_RA_true': sample['traj'].cpu().numpy(),
        }
    return predictions


def build_parity_arrays(model, dataset, config, max_curves):
    f_true_all = []
    f_pred_all = []
    for sample in dataset.samples[:max_curves]:
        static = sample['static'].unsqueeze(0).to(config.device)
        f_eq = sample['f_eq'].view(1, 1).to(config.device)
        dG = sample['dG_norm'].view(1, 1).to(config.device)
        t_span = sample['t_span'].to(config.device)
        with torch.no_grad():
            pred = model(static, f_eq, dG, t_span)[0].cpu().numpy()
        f_true_all.append(sample['traj'].cpu().numpy())
        f_pred_all.append(pred)
    return np.concatenate(f_true_all), np.concatenate(f_pred_all)


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures and analysis from a trained PhysicsNODE checkpoint.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint. Defaults to physics_node_best.pt')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split used for parity and representative curves')
    parser.add_argument('--max-parity-curves', type=int, default=200, help='Maximum number of curves for parity generation')
    parser.add_argument('--uq-samples', type=int, default=16, help='Monte Carlo samples used for uncertainty throughout the publication pipeline')
    parser.add_argument('--symbolic-samples', type=int, default=500, help='Synthetic samples passed to symbolic regression')
    parser.add_argument('--skip-explain', action='store_true', help='Skip explainability analyses')
    parser.add_argument('--skip-symbolic', action='store_true', help='Skip symbolic regression')
    parser.add_argument('--device', type=str, default=None, help='Override device, e.g. cpu or cuda')
    args = parser.parse_args()

    config = get_config()
    if args.device:
        config.device = torch.device(args.device)
    config.model.n_mc_samples = args.uq_samples

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (config.checkpoint_dir / 'physics_node_best.pt')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split_path = config.synthetic_dir / f'{args.split}.csv'
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")
    df = pd.read_csv(split_path)
    dataset = AusteniteReversionDataset(df, config)

    model, checkpoint = load_model(checkpoint_path, config)
    history = checkpoint.get('history')

    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"Using split: {split_path} ({len(dataset)} curves)")

    representative_ids = select_representative_ids(df, n_curves=3)
    predictions = build_prediction_panels(model, dataset, representative_ids, config, args.uq_samples)
    f_true, f_pred = build_parity_arrays(model, dataset, config, max_curves=args.max_parity_curves)

    explain_results = {}
    if not args.skip_explain:
        sample_data = np.stack([sample['static'].cpu().numpy() for sample in dataset.samples[: min(len(dataset), 64)]])
        explain_results = run_explainability_suite(model, sample_data=sample_data, config=config)

    symbolic_result = None
    if not args.skip_symbolic:
        symbolic_result = extract_symbolic_equation(model, config=config, n_samples=args.symbolic_samples)

    generate_all_figures(
        training_history=history,
        sensitivity_results=explain_results.get('sensitivity'),
        shap_result=explain_results.get('shap'),
        predictions=predictions,
        f_true=f_true,
        f_pred=f_pred,
        config=config,
    )

    summary = {
        'checkpoint': str(checkpoint_path),
        'split': args.split,
        'representative_ids': representative_ids,
        'consistency': explain_results.get('consistency'),
        'symbolic_best_equation': None if symbolic_result is None else symbolic_result.get('best_equation'),
        'symbolic_best_loss': None if symbolic_result is None else symbolic_result.get('best_loss'),
    }
    summary_path = config.figure_dir / 'publication_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info(f"Saved publication summary to {summary_path}")


if __name__ == '__main__':
    main()
