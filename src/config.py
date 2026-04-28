from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import torch

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
LITERATURE_DIR = DATA_DIR / "literature_validation"
CALPHAD_DIR = DATA_DIR / "calphad_tables"
USER_DATA_DIR = DATA_DIR / "user_experimental"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "figures"
LOG_DIR = PROJECT_ROOT / "logs"

for d in [SYNTHETIC_DIR, LITERATURE_DIR, CALPHAD_DIR, USER_DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


@dataclass
class PhysicalConstants:
    R: float = 8.314
    D0_Mn: float = 1.785e-5
    Q_Mn: float = 264_000.0
    D0_Mn_ferrite: float = 7.56e-5
    Q_Mn_ferrite: float = 224_500.0
    D0_C: float = 2.343e-5
    Q_C: float = 148_000.0
    angel_intercept: float = 413.0
    angel_C_N: float = -462.0
    angel_Si: float = -9.2
    angel_Mn: float = -8.1
    angel_Cr: float = -13.7
    angel_Ni: float = -9.5
    angel_Mo: float = -18.5
    C_HJ: float = 20.0
    epsilon_nucleation: float = 0.005
    jmak_n_min: float = 1.5
    jmak_n_max: float = 3.0
    jmak_k0: float = 1e-8
    jmak_Q_eff: float = 200_000.0


@dataclass
class CompositionBounds:
    Mn_min: float = 4.0
    Mn_max: float = 12.0
    C_min: float = 0.05
    C_max: float = 0.30
    Al_min: float = 0.0
    Al_max: float = 3.0
    Si_min: float = 0.0
    Si_max: float = 2.0
    N_min: float = 0.0
    N_max: float = 0.02
    T_ICA_min: float = 575.0
    T_ICA_max: float = 750.0
    t_min: float = 0.0
    t_max: float = 10_000.0


@dataclass
class ModelConfig:
    latent_dim: int = 32
    composition_embed_dim: int = 32
    n_attention_heads: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 96, 64])
    activation: str = "silu"
    input_dim: int = 8  # static features: T_n, Mn, C, Al, Si, D_log, dG_n, PHJ_n (5 comp + 3 physics)
    output_dim: int = 1
    use_spectral_norm: bool = True
    augmented_dim: int = 4

    init_dropout_rate: float = 0.1
    weight_regularizer: float = 1e-6
    dropout_regularizer: float = 1e-5

    solver: str = "dopri5"
    adjoint: bool = True
    rtol: float = 1e-5
    atol: float = 1e-7
    n_eval_points: int = 50
    max_num_steps: int = 10000

    f_eq_tolerance: float = 1.02
    f_initial: float = 0.001

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 800
    early_stopping_patience: int = 80
    gradient_clip_val: float = 1.0
    use_amp: bool = False
    accumulate_grad_batches: int = 1
    use_homoscedastic: bool = False
    use_gradnorm: bool = False
    checkpoint_metric: str = "val_real_rmse"

    scheduler_type: str = "cosine_warm_restarts"
    scheduler_T_0: int = 100
    scheduler_T_mult: int = 2
    scheduler_eta_min: float = 1e-7

    swa_start_epoch: int = 400
    swa_lr: float = 1e-4
    swag_rank: int = 20
    swag_collect_freq: int = 5

    homoscedastic_init: Dict[str, float] = field(default_factory=lambda: {
        "data": 0.0, "physics": 1.0, "monotone": 0.0, "bound": -1.0,
    })

    n_ensemble_members: int = 5
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 137, 256, 512, 1024])
    n_mc_samples: int = 100
    confidence_level: float = 0.95

    optuna_n_trials: int = 100
    optuna_timeout: int = 14400
    random_seed: int = 42


@dataclass
class DataConfig:
    n_synthetic_samples: int = 8000
    n_time_points: int = 60
    noise_sigma_min: float = 0.015
    noise_sigma_max: float = 0.04
    outlier_fraction: float = 0.03
    outlier_magnitude: float = 0.12
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    T_ref: float = 673.0
    T_scale: float = 100.0
    use_curriculum: bool = True
    curriculum_stages: int = 3

    # Provenance-aware data settings
    use_real_data: bool = True
    real_data_weight: float = 3.0  # upweight real vs synthetic in loss
    synthetic_calibration_samples: int = 500
    synthetic_exploration_samples: int = 2000
    provenance_aware_loss: bool = True
    real_only: bool = False  # train exclusively on real data
    real_curve_group_min_points: int = 2


@dataclass
class OptimizationConfig:
    target_RA_fraction: float = 0.30
    target_RA_tolerance: float = 0.03
    T_margin_from_Ac1: float = 10.0
    T_margin_from_Ac3: float = 10.0
    t_min_anneal: float = 60.0
    t_max_anneal: float = 7200.0
    n_optuna_trials: int = 200
    n_pareto_points: int = 80
    use_gp_surrogate: bool = True


@dataclass
class VisualizationConfig:
    dpi: int = 300
    figsize_single: Tuple[float, float] = (4.5, 3.5)
    figsize_double: Tuple[float, float] = (7.5, 5.5)
    font_family: str = "serif"
    font_size: int = 10
    tick_size: int = 8
    legend_size: int = 8
    line_width: float = 1.5
    marker_size: float = 4.0
    save_formats: List[str] = field(default_factory=lambda: ["pdf", "png"])
    colors: List[str] = field(default_factory=lambda: [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#F0E442", "#56B4E9", "#E69F00", "#000000",
    ])


@dataclass
class Config:
    physics: PhysicalConstants = field(default_factory=PhysicalConstants)
    composition: CompositionBounds = field(default_factory=CompositionBounds)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    device: torch.device = field(default_factory=detect_device)
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    synthetic_dir: Path = SYNTHETIC_DIR
    literature_dir: Path = LITERATURE_DIR
    calphad_dir: Path = CALPHAD_DIR
    user_data_dir: Path = USER_DATA_DIR
    model_dir: Path = MODEL_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    figure_dir: Path = FIGURE_DIR
    log_dir: Path = LOG_DIR
    use_wandb: bool = False
    wandb_project: str = "medium-mn-node"


def get_config() -> Config:
    return Config()


if __name__ == "__main__":
    cfg = get_config()
    print(f"Device: {cfg.device}")
    print(f"Latent dim: {cfg.model.latent_dim}")
    print(f"Hidden: {cfg.model.hidden_dims}")
    print(f"Adjoint: {cfg.model.adjoint}")
    print(f"AMP: {cfg.model.use_amp}")
