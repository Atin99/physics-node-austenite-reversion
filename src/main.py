import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S',
                    handlers=[logging.StreamHandler(), logging.FileHandler('pipeline.log', mode='w', encoding='utf-8')])
logger = logging.getLogger(__name__)


def step_generate_data(config):
    logger.info("STEP 1: Generate Data (3-Tier: Real -> Calibrated -> Exploratory)")
    from data_generator import (
        build_full_dataset,
        create_literature_validation_data,
        plot_synthetic_curves,
        prepare_train_val_test_split,
        save_synthetic_data,
    )
    from thermodynamics import precompute_thermo_tables

    precompute_thermo_tables(config, 20, 30)
    df = build_full_dataset(config)
    save_synthetic_data(df, config)
    df_train, df_val, df_test = prepare_train_val_test_split(df, config)
    df_train.to_csv(config.synthetic_dir / "train.csv", index=False)
    df_val.to_csv(config.synthetic_dir / "val.csv", index=False)
    df_test.to_csv(config.synthetic_dir / "test.csv", index=False)
    create_literature_validation_data(config)
    plot_synthetic_curves(df, config=config)

    # Report provenance breakdown
    if 'provenance' in df.columns:
        for prov, count in df['provenance'].value_counts().items():
            logger.info(f"  {prov}: {count} data points")

    logger.info(
        f"Generated {len(df)} total data points ({df['sample_id'].nunique()} curves) | "
        f"train={df_train['sample_id'].nunique()} val={df_val['sample_id'].nunique()} test={df_test['sample_id'].nunique()}"
    )
    return df_train, df_val


def step_train(config, df_train=None, df_val=None):
    logger.info("STEP 2: Train PhysicsNODE")
    import pandas as pd
    from model import PhysicsNODE
    from trainer import Trainer, set_seed, create_data_loaders

    if df_train is None:
        p = config.synthetic_dir / "train.csv"
        if not p.exists():
            return None, None
        df_train = pd.read_csv(p)
        df_val = pd.read_csv(config.synthetic_dir / "val.csv")

    set_seed(config.model.random_seed)
    model = PhysicsNODE(config.model)
    logger.info(model.get_model_summary())
    tr, va = create_data_loaders(df_train, df_val, config)
    trainer = Trainer(model, config)
    history = trainer.train(tr, va)
    return model, history


def step_optimize(config, model=None):
    logger.info("STEP 3: Schedule Optimization")
    from model import PhysicsNODE
    from optimizer_annealing import optimize_single_objective, recommend_schedule

    if model is None:
        model = PhysicsNODE(config.model).to(config.device)
        p = config.checkpoint_dir / "physics_node_best.pt"
        if p.exists():
            model.load_state_dict(torch.load(p, map_location=config.device, weights_only=False)['model'])

    alloys = [
        {'Mn': 7.0, 'C': 0.1, 'Al': 1.5, 'Si': 0.5},
        {'Mn': 10.0, 'C': 0.1, 'Al': 0.0, 'Si': 0.0},
        {'Mn': 5.0, 'C': 0.2, 'Al': 2.0, 'Si': 0.5},
    ]
    results = {}
    for comp in alloys:
        name = f"Fe-{comp['Mn']:.0f}Mn-{comp['C']:.1f}C"
        r = optimize_single_objective(model, comp, 0.30, config, 50)
        results[name] = r
        logger.info(recommend_schedule(model, comp, 0.30, config))
    return results


def step_explain(config, model=None):
    logger.info("STEP 4: Explainability")
    from model import PhysicsNODE
    from explainability import run_explainability_suite

    if model is None:
        model = PhysicsNODE(config.model).to(config.device)
        p = config.checkpoint_dir / "physics_node_best.pt"
        if p.exists():
            model.load_state_dict(torch.load(p, map_location=config.device, weights_only=False)['model'])

    return run_explainability_suite(model, config=config)


def step_symbolic(config, model=None):
    logger.info("STEP 5: Symbolic Regression")
    from model import PhysicsNODE
    from symbolic_regression import extract_symbolic_equation

    if model is None:
        model = PhysicsNODE(config.model).to(config.device)
        p = config.checkpoint_dir / "physics_node_best.pt"
        if p.exists():
            model.load_state_dict(torch.load(p, map_location=config.device, weights_only=False)['model'])

    return extract_symbolic_equation(model, config)


def step_figures(config, history=None, explain_results=None, opt_results=None):
    logger.info("STEP 6: Figures")
    from visualizations import generate_all_figures
    sens = explain_results.get('sensitivity') if explain_results else None
    shap = explain_results.get('shap') if explain_results else None
    first_opt = list(opt_results.values())[0] if opt_results else None
    generate_all_figures(history, sens, shap, first_opt, config=config)


def step_app():
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "streamlit_app.py")])


def main():
    parser = argparse.ArgumentParser(description="PhysicsNODE Pipeline")
    parser.add_argument('--generate-data', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--symbolic', action='store_true')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--app', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--real-only', action='store_true', help='Train only on real experimental data')
    parser.add_argument('--n-samples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.all:
        args.generate_data = args.train = args.optimize = args.explain = args.symbolic = args.figures = True

    config = get_config()
    if args.n_samples:
        config.data.n_synthetic_samples = args.n_samples
    if args.epochs:
        config.model.max_epochs = args.epochs
    if args.device:
        config.device = torch.device(args.device)

    logger.info(f"Device: {config.device} | Adjoint: {config.model.adjoint} | AMP: {config.model.use_amp}")
    if args.real_only:
        config.data.real_only = True
        logger.info("MODE: Real-data-only (no synthetic augmentation)")
    t0 = time.time()
    model, history, explain_results, opt_results = None, None, None, None
    df_train, df_val = None, None

    if args.generate_data:
        df_train, df_val = step_generate_data(config)
    if args.train:
        model, history = step_train(config, df_train, df_val)
    if args.optimize:
        opt_results = step_optimize(config, model)
    if args.explain:
        explain_results = step_explain(config, model)
    if args.symbolic:
        step_symbolic(config, model)
    if args.figures:
        step_figures(config, history, explain_results, opt_results)
    if args.app:
        step_app()
        return

    logger.info(f"Pipeline done in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
