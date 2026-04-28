import logging
import math
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import Config, get_config
from features import compute_diffusivity, compute_hollomon_jaffe
from losses import PhysicsConstrainedLoss
from model import EnsembleNODE, PhysicsNODE, SWAG

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class AusteniteReversionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: Optional[Config] = None):
        if config is None:
            config = get_config()
        self.samples = []
        for sid in df['sample_id'].unique():
            sub = df[df['sample_id'] == sid].sort_values('t_seconds')
            Mn, C, Al, Si = sub['Mn'].iloc[0], sub['C'].iloc[0], sub['Al'].iloc[0], sub['Si'].iloc[0]
            T_c = sub['T_celsius'].iloc[0]
            T_K = T_c + 273.15
            f_eq, dG = sub['f_eq'].iloc[0], sub['delta_G'].iloc[0]
            D_Mn = compute_diffusivity(T_K)
            P_HJ = compute_hollomon_jaffe(T_K, max(sub['t_seconds'].median(), 1.0))
            obs_mask = sub['is_observed'].to_numpy(dtype=np.float32) if 'is_observed' in sub.columns else np.ones(len(sub), dtype=np.float32)

            static = np.array(
                [
                    (T_K - config.data.T_ref) / config.data.T_scale,
                    Mn,
                    C,
                    Al,
                    Si,
                    np.log10(D_Mn + 1e-30),
                    dG / 1000.0,
                    P_HJ / 20000.0,
                ],
                dtype=np.float32,
            )
            self.samples.append(
                {
                    'sample_id': int(sid),
                    'static': torch.tensor(static, dtype=torch.float32),
                    'traj': torch.tensor(sub['f_RA'].to_numpy(dtype=np.float32), dtype=torch.float32),
                    't_span': torch.tensor(sub['t_seconds'].to_numpy(dtype=np.float32), dtype=torch.float32),
                    'obs_mask': torch.tensor(obs_mask, dtype=torch.float32),
                    'f_eq': torch.tensor(f_eq, dtype=torch.float32),
                    'dG_norm': torch.tensor(dG / 1000.0, dtype=torch.float32),
                    'k_jmak': torch.tensor(sub['k_jmak'].iloc[0], dtype=torch.float32),
                    'n_jmak': torch.tensor(sub['n_jmak'].iloc[0], dtype=torch.float32),
                    'T_celsius': float(T_c),
                    'provenance': sub['provenance'].iloc[0] if 'provenance' in sub.columns else 'unknown',
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_fn(batch):
    batch_size = len(batch)
    lengths = torch.tensor([sample['traj'].shape[0] for sample in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    traj = torch.zeros(batch_size, max_len, dtype=torch.float32)
    t_span = torch.zeros(batch_size, max_len, dtype=torch.float32)
    point_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    obs_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for idx, sample in enumerate(batch):
        cur_len = sample['traj'].shape[0]
        traj[idx, :cur_len] = sample['traj']
        t_span[idx, :cur_len] = sample['t_span']
        point_mask[idx, :cur_len] = 1.0
        obs_mask[idx, :cur_len] = sample['obs_mask']
        if cur_len > 0 and cur_len < max_len:
            traj[idx, cur_len:] = sample['traj'][-1]
            t_span[idx, cur_len:] = sample['t_span'][-1]

    return {
        'static': torch.stack([sample['static'] for sample in batch]),
        'traj': traj,
        't_span': t_span,
        'point_mask': point_mask,
        'obs_mask': obs_mask,
        'lengths': lengths,
        'f_eq': torch.stack([sample['f_eq'] for sample in batch]).unsqueeze(1),
        'dG_norm': torch.stack([sample['dG_norm'] for sample in batch]).unsqueeze(1),
        'k_jmak': torch.stack([sample['k_jmak'] for sample in batch]),
        'n_jmak': torch.stack([sample['n_jmak'] for sample in batch]),
        'provenance': [sample['provenance'] for sample in batch],
    }


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model: PhysicsNODE,
        config: Optional[Config] = None,
        use_gradnorm: Optional[bool] = None,
        use_homoscedastic: Optional[bool] = None,
    ):
        if config is None:
            config = get_config()
        self.model = model
        self.config = config
        self.mc = config.model
        self.device = config.device
        self.model.to(self.device)

        if use_gradnorm is None:
            use_gradnorm = bool(getattr(self.mc, 'use_gradnorm', False))
        if use_homoscedastic is None:
            use_homoscedastic = bool(getattr(self.mc, 'use_homoscedastic', False))
        self.use_homoscedastic = use_homoscedastic

        self.criterion = PhysicsConstrainedLoss(
            config=self.mc,
            use_gradnorm=use_gradnorm,
            use_homoscedastic=use_homoscedastic,
        )
        if use_homoscedastic:
            self.criterion.to(self.device)
            all_params = list(model.parameters()) + list(self.criterion.parameters())
        else:
            all_params = list(model.parameters())

        self.optimizer = torch.optim.AdamW(all_params, lr=self.mc.learning_rate, weight_decay=self.mc.weight_decay)

        if self.mc.scheduler_type == "cosine_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.mc.scheduler_T_0,
                T_mult=self.mc.scheduler_T_mult,
                eta_min=self.mc.scheduler_eta_min,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.mc.max_epochs,
                eta_min=self.mc.scheduler_eta_min,
            )

        self.scaler = GradScaler('cuda', enabled=self.mc.use_amp and torch.cuda.is_available())
        self.swag = None
        if self.mc.swa_start_epoch < self.mc.max_epochs:
            self.swag = SWAG(model, rank=self.mc.swag_rank, device=self.device)

        history_keys = [
            'train_loss',
            'val_loss',
            'train_data',
            'train_physics',
            'train_mono',
            'train_bound',
            'val_data',
            'val_physics',
            'val_mono',
            'val_bound',
            'val_mae',
            'val_rmse',
            'val_real_mae',
            'val_real_rmse',
            'val_endpoint_mae',
            'val_real_endpoint_mae',
            'lr',
            'violations',
            'nfe',
            'skipped_batches',
        ]
        self.history: Dict[str, List[float]] = {key: [] for key in history_keys}
        self.best_val_loss = float('inf')
        self.best_checkpoint_metric = float('inf')
        self.best_metric_name = getattr(self.mc, 'checkpoint_metric', 'val_real_rmse')
        self.best_val_metrics: Dict[str, float] = {}
        self.patience_counter = 0

    def _get_shared_layer(self):
        for module in self.model.ode_func.net:
            if hasattr(module, 'weight'):
                return module
        return None

    def _selection_metric(self, metrics: Dict[str, float]) -> float:
        preferred = getattr(self.mc, 'checkpoint_metric', 'val_real_rmse')
        if preferred == 'val_real_rmse' and metrics.get('n_real_observed', 0) > 0:
            value = metrics.get('real_rmse', float('inf'))
        elif preferred == 'val_rmse':
            value = metrics.get('rmse', float('inf'))
        elif preferred == 'val_real_mae' and metrics.get('n_real_observed', 0) > 0:
            value = metrics.get('real_mae', float('inf'))
        else:
            value = metrics.get('rmse', float('inf'))
        if not math.isfinite(value):
            value = self.best_val_loss
        return value

    def _train_epoch(self, loader, epoch):
        self.model.train()
        totals = {'total': 0.0, 'data': 0.0, 'physics': 0.0, 'monotone': 0.0, 'bound': 0.0}
        nfe_total, n, skipped = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        accum_steps = 0
        shared_layer = self._get_shared_layer()

        for step, batch in enumerate(loader):
            static = batch['static'].to(self.device)
            f_true = batch['traj'].to(self.device)
            t_span = batch['t_span'].to(self.device)
            point_mask = batch['point_mask'].to(self.device)
            obs_mask = batch['obs_mask'].to(self.device)
            lengths = batch['lengths']
            f_eq = batch['f_eq'].to(self.device)
            dG = batch['dG_norm'].to(self.device)
            k_j = batch['k_jmak'].to(self.device)
            n_j = batch['n_jmak'].to(self.device)

            with autocast('cuda', enabled=self.mc.use_amp and torch.cuda.is_available()):
                try:
                    f_pred = self.model(static, f_eq, dG, t_span, lengths=lengths)
                    ml = min(f_pred.shape[1], f_true.shape[1])
                    t_loss = t_span[:, :ml]
                    loss_d = self.criterion(
                        f_pred[:, :ml],
                        f_true[:, :ml],
                        f_eq,
                        t_loss,
                        k_j,
                        n_j,
                        self.model,
                        shared_layer,
                        epoch,
                        point_mask=point_mask[:, :ml],
                        obs_mask=obs_mask[:, :ml],
                    )

                    provenance = batch.get('provenance', ['unknown'] * static.shape[0])
                    real_mask = torch.tensor(
                        [p in ('experimental', 'user_provided') for p in provenance],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if real_mask.sum() > 0 and getattr(self.config.data, 'provenance_aware_loss', False):
                        weight = 1.0 + (self.config.data.real_data_weight - 1.0) * real_mask.mean()
                        loss_d['total'] = loss_d['total'] * weight
                except Exception as exc:
                    skipped += 1
                    logger.warning(f"Skipped batch {step}: {exc}")
                    continue

            self.scaler.scale(loss_d['total']).backward()
            accum_steps += 1
            if accum_steps >= self.mc.accumulate_grad_batches:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.mc.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accum_steps = 0

            for key in totals:
                if key in loss_d:
                    totals[key] += float(loss_d[key].detach().item())
            nfe_total += float(self.model.ode_func.nfe)
            self.model.ode_func.nfe = 0
            n += 1

        if accum_steps > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.mc.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return {key: value / max(n, 1) for key, value in totals.items()}, nfe_total / max(n, 1), skipped

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        totals = {'total': 0.0, 'data': 0.0, 'physics': 0.0, 'monotone': 0.0, 'bound': 0.0}
        n, violations, valid_pairs = 0, 0.0, 0.0
        abs_sum = sq_sum = obs_count = 0.0
        real_abs_sum = real_sq_sum = real_obs_count = 0.0
        endpoint_abs = endpoint_count = 0.0
        real_endpoint_abs = real_endpoint_count = 0.0

        for batch in loader:
            static = batch['static'].to(self.device)
            f_true = batch['traj'].to(self.device)
            t_span = batch['t_span'].to(self.device)
            point_mask = batch['point_mask'].to(self.device)
            obs_mask = batch['obs_mask'].to(self.device)
            lengths = batch['lengths']
            f_eq = batch['f_eq'].to(self.device)
            dG = batch['dG_norm'].to(self.device)
            k_j = batch['k_jmak'].to(self.device)
            n_j = batch['n_jmak'].to(self.device)

            try:
                f_pred = self.model(static, f_eq, dG, t_span, lengths=lengths)
                ml = min(f_pred.shape[1], f_true.shape[1])
                t_loss = t_span[:, :ml]
                point_mask_cut = point_mask[:, :ml]
                obs_mask_cut = obs_mask[:, :ml]
                pred_cut = f_pred[:, :ml]
                true_cut = f_true[:, :ml]
                loss_d = self.criterion(
                    pred_cut,
                    true_cut,
                    f_eq,
                    t_loss,
                    k_j,
                    n_j,
                    point_mask=point_mask_cut,
                    obs_mask=obs_mask_cut,
                )
            except Exception as exc:
                logger.warning(f"Validation batch error: {exc}")
                continue

            for key in totals:
                if key in loss_d:
                    totals[key] += float(loss_d[key].item())

            valid_pair_mask = (point_mask_cut[:, 1:] > 0.5) & (point_mask_cut[:, :-1] > 0.5)
            if valid_pair_mask.numel() > 0:
                violations += float(((pred_cut[:, 1:] - pred_cut[:, :-1] < -0.001) & valid_pair_mask).sum().item())
                valid_pairs += float(valid_pair_mask.sum().item())

            diff = pred_cut - true_cut
            obs_mask_bool = obs_mask_cut > 0.5
            if obs_mask_bool.any():
                obs_diff = diff[obs_mask_bool]
                abs_sum += float(obs_diff.abs().sum().item())
                sq_sum += float(obs_diff.pow(2).sum().item())
                obs_count += float(obs_diff.numel())

            provenance = batch.get('provenance', ['unknown'] * pred_cut.shape[0])
            real_rows = torch.tensor(
                [p in ('experimental', 'user_provided') for p in provenance],
                dtype=torch.bool,
                device=pred_cut.device,
            )
            if real_rows.any():
                real_obs_mask = obs_mask_bool & real_rows.unsqueeze(1)
                if real_obs_mask.any():
                    real_diff = diff[real_obs_mask]
                    real_abs_sum += float(real_diff.abs().sum().item())
                    real_sq_sum += float(real_diff.pow(2).sum().item())
                    real_obs_count += float(real_diff.numel())

            for row_idx in range(pred_cut.shape[0]):
                observed_idx = torch.nonzero(obs_mask_bool[row_idx], as_tuple=False).flatten()
                if observed_idx.numel() == 0:
                    continue
                last_idx = int(observed_idx[-1].item())
                endpoint_abs += float((pred_cut[row_idx, last_idx] - true_cut[row_idx, last_idx]).abs().item())
                endpoint_count += 1.0
                if bool(real_rows[row_idx].item()):
                    real_endpoint_abs += float((pred_cut[row_idx, last_idx] - true_cut[row_idx, last_idx]).abs().item())
                    real_endpoint_count += 1.0

            n += 1

        def safe_mae(total_abs, count):
            return total_abs / count if count > 0 else float('inf')

        def safe_rmse(total_sq, count):
            return math.sqrt(total_sq / count) if count > 0 else float('inf')

        metrics = {
            'mae': safe_mae(abs_sum, obs_count),
            'rmse': safe_rmse(sq_sum, obs_count),
            'real_mae': safe_mae(real_abs_sum, real_obs_count),
            'real_rmse': safe_rmse(real_sq_sum, real_obs_count),
            'endpoint_mae': safe_mae(endpoint_abs, endpoint_count),
            'real_endpoint_mae': safe_mae(real_endpoint_abs, real_endpoint_count),
            'n_observed': obs_count,
            'n_real_observed': real_obs_count,
        }
        return (
            {key: value / max(n, 1) for key, value in totals.items()},
            violations / max(valid_pairs, 1.0),
            metrics,
        )

    def train(self, train_loader, val_loader, verbose=True):
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Training on {self.device} | Epochs: {self.mc.max_epochs} | Batch: {self.mc.batch_size} | LR: {self.mc.learning_rate}"
        )
        logger.info(f"AMP: {self.mc.use_amp} | Adjoint: {self.mc.adjoint} | SWAG: {self.swag is not None}")
        logger.info(f"Checkpoint metric: {self.best_metric_name}")
        logger.info(f"{'=' * 60}")
        t0 = time.time()

        for epoch in range(1, self.mc.max_epochs + 1):
            train_loss, nfe, skipped = self._train_epoch(train_loader, epoch)
            val_loss, viol, val_metrics = self._validate(val_loader)
            self.scheduler.step()

            self.history['train_loss'].append(train_loss['total'])
            self.history['val_loss'].append(val_loss['total'])
            self.history['train_data'].append(train_loss['data'])
            self.history['train_physics'].append(train_loss['physics'])
            self.history['train_mono'].append(train_loss['monotone'])
            self.history['train_bound'].append(train_loss['bound'])
            self.history['val_data'].append(val_loss['data'])
            self.history['val_physics'].append(val_loss['physics'])
            self.history['val_mono'].append(val_loss['monotone'])
            self.history['val_bound'].append(val_loss['bound'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_real_mae'].append(val_metrics['real_mae'])
            self.history['val_real_rmse'].append(val_metrics['real_rmse'])
            self.history['val_endpoint_mae'].append(val_metrics['endpoint_mae'])
            self.history['val_real_endpoint_mae'].append(val_metrics['real_endpoint_mae'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['violations'].append(viol)
            self.history['nfe'].append(nfe)
            self.history['skipped_batches'].append(float(skipped))

            if self.swag and epoch >= self.mc.swa_start_epoch and epoch % self.mc.swag_collect_freq == 0:
                self.swag.collect()

            checkpoint_metric = self._selection_metric(val_metrics)
            if checkpoint_metric < self.best_checkpoint_metric:
                self.best_checkpoint_metric = checkpoint_metric
                self.best_val_loss = float(val_loss['total'])
                self.best_val_metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in val_metrics.items()}
                self.patience_counter = 0
                self.save_checkpoint('best')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.mc.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if verbose:
                logger.info(
                    "E%4d | T:%.5f V:%.5f | RMSE:%.5f RealRMSE:%.5f | End:%.5f RealEnd:%.5f | NFE:%.0f | Viol:%.2f%% | Skip:%d | LR:%.1e"
                    % (
                        epoch,
                        train_loss['total'],
                        val_loss['total'],
                        val_metrics['rmse'],
                        val_metrics['real_rmse'],
                        val_metrics['endpoint_mae'],
                        val_metrics['real_endpoint_mae'],
                        nfe,
                        viol * 100.0,
                        skipped,
                        self.optimizer.param_groups[0]['lr'],
                    )
                )

        logger.info(
            f"Done in {(time.time() - t0) / 60:.1f}m | Best {self.best_metric_name}: {self.best_checkpoint_metric:.6f}"
        )
        self.save_checkpoint('last')
        return self.history

    def save_checkpoint(self, tag='best'):
        path = self.config.checkpoint_dir / f"physics_node_{tag}.pt"
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val': self.best_val_loss,
            'best_metric_name': self.best_metric_name,
            'best_metric_value': self.best_checkpoint_metric,
            'best_val_metrics': self.best_val_metrics,
            'history': self.history,
            'config': {
                'hidden_dims': self.mc.hidden_dims,
                'latent_dim': self.mc.latent_dim,
                'augmented_dim': self.mc.augmented_dim,
                'use_homoscedastic': self.use_homoscedastic,
            },
        }
        if self.use_homoscedastic:
            state['criterion'] = self.criterion.state_dict()
        if self.swag and self.swag.n_collected > 0:
            state['swag_mean'] = self.swag.mean
            state['swag_sq_mean'] = self.swag.sq_mean
        torch.save(state, path)
        return path

    def load_checkpoint(self, tag='best'):
        path = self.config.checkpoint_dir / f"physics_node_{tag}.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.best_val_loss = ckpt.get('best_val', float('inf'))
        self.best_metric_name = ckpt.get('best_metric_name', self.best_metric_name)
        self.best_checkpoint_metric = ckpt.get('best_metric_value', self.best_checkpoint_metric)
        self.best_val_metrics = ckpt.get('best_val_metrics', {})
        if 'criterion' in ckpt and self.use_homoscedastic:
            self.criterion.load_state_dict(ckpt['criterion'])
        if 'history' in ckpt:
            self.history = ckpt['history']


def create_data_loaders(train_df, val_df, config=None):
    if config is None:
        config = get_config()
    train_ds = AusteniteReversionDataset(train_df, config)
    val_ds = AusteniteReversionDataset(val_df, config)

    provenance_counts = Counter(sample['provenance'] for sample in train_ds.samples)
    sample_weights = []
    for sample in train_ds.samples:
        prov = sample['provenance']
        observed_idx = torch.nonzero(sample['obs_mask'] > 0.5, as_tuple=False).flatten()
        if observed_idx.numel() > 0:
            last_idx = int(observed_idx[-1].item())
        else:
            last_idx = sample['traj'].shape[0] - 1
        endpoint = float(sample['traj'][last_idx].item())
        duration = float(sample['t_span'][last_idx].item()) if sample['t_span'].numel() else 0.0
        provenance_weight = len(train_ds) / max(provenance_counts[prov], 1)
        endpoint_weight = 1.0 + 4.0 * min(endpoint, 0.5)
        duration_weight = 1.0 + 0.15 * min(np.log10(max(duration, 1.0)), 4.0)
        sample_weights.append(provenance_weight * endpoint_weight * duration_weight)

    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.model.batch_size,
        sampler=sampler,
        collate_fn=_collate_fn,
        drop_last=False,
        num_workers=0,
        pin_memory=config.device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.model.batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
        pin_memory=config.device.type == 'cuda',
    )
    return train_loader, val_loader


def optuna_objective(trial, train_df, val_df, config):
    lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
    n_layers = trial.suggest_int('n_layers', 2, 6)
    hidden = trial.suggest_categorical('hidden', [64, 96, 128, 192])
    aug_dim = trial.suggest_int('aug_dim', 2, 8)
    dropout = trial.suggest_float('dropout', 0.01, 0.25)
    wd = trial.suggest_float('wd', 1e-6, 1e-3, log=True)

    config.model.hidden_dims = [hidden] * n_layers
    config.model.augmented_dim = aug_dim
    config.model.learning_rate = lr
    config.model.init_dropout_rate = dropout
    config.model.weight_decay = wd
    config.model.max_epochs = 150

    set_seed(config.model.random_seed)
    model = PhysicsNODE(config.model)
    tr_loader, va_loader = create_data_loaders(train_df, val_df, config)
    trainer = Trainer(model, config)
    trainer.train(tr_loader, va_loader, verbose=False)
    return trainer.best_checkpoint_metric


def run_optuna_hpo(train_df, val_df, config=None, n_trials=50):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if config is None:
        config = get_config()
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.model.random_seed))
    study.optimize(
        lambda trial: optuna_objective(trial, train_df, val_df, config),
        n_trials=n_trials,
        timeout=config.model.optuna_timeout,
        show_progress_bar=True,
    )
    logger.info(f"Best: {study.best_trial.value:.6f} | {study.best_trial.params}")
    return {'best_params': study.best_trial.params, 'best_value': study.best_trial.value, 'study': study}


def train_ensemble(train_df, val_df, config=None):
    if config is None:
        config = get_config()
    ensemble = EnsembleNODE(config.model.n_ensemble_members, config.model)
    for idx, (model, seed) in enumerate(zip(ensemble.models, config.model.ensemble_seeds)):
        logger.info(f"Ensemble {idx + 1}/{config.model.n_ensemble_members} seed={seed}")
        set_seed(seed)
        tr_loader, va_loader = create_data_loaders(train_df, val_df, config)
        trainer = Trainer(model, config)
        trainer.train(tr_loader, va_loader)
        torch.save(model.state_dict(), config.checkpoint_dir / f"ensemble_{idx}.pt")
    return ensemble
