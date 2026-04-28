from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import json
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast

from config import get_config, Config
from model import PhysicsNODE, SWAG, EnsembleNODE
from losses import PhysicsConstrainedLoss
from features import compute_diffusivity, compute_hollomon_jaffe

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

            static = np.array([(T_K - config.data.T_ref) / config.data.T_scale, Mn, C, Al, Si,
                               np.log10(D_Mn + 1e-30), dG / 1000.0, P_HJ / 20000.0], dtype=np.float32)
            self.samples.append({
                'sample_id': int(sid),
                'static': torch.tensor(static),
                'traj': torch.tensor(sub['f_RA'].values, dtype=torch.float32),
                't_span': torch.tensor(sub['t_seconds'].values, dtype=torch.float32),
                'f_eq': torch.tensor(f_eq, dtype=torch.float32),
                'dG_norm': torch.tensor(dG / 1000.0, dtype=torch.float32),
                'k_jmak': torch.tensor(sub['k_jmak'].iloc[0], dtype=torch.float32),
                'n_jmak': torch.tensor(sub['n_jmak'].iloc[0], dtype=torch.float32),
                'T_celsius': float(T_c),
                'provenance': sub['provenance'].iloc[0] if 'provenance' in sub.columns else 'unknown',
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_fn(batch):
    min_len = min(b['traj'].shape[0] for b in batch)
    return {
        'static': torch.stack([b['static'] for b in batch]),
        'traj': torch.stack([b['traj'][:min_len] for b in batch]),
        't_span': torch.stack([b['t_span'][:min_len] for b in batch]),
        'f_eq': torch.stack([b['f_eq'] for b in batch]).unsqueeze(1),
        'dG_norm': torch.stack([b['dG_norm'] for b in batch]).unsqueeze(1),
        'k_jmak': torch.stack([b['k_jmak'] for b in batch]),
        'n_jmak': torch.stack([b['n_jmak'] for b in batch]),
        'provenance': [b['provenance'] for b in batch],
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
    def __init__(self, model: PhysicsNODE, config: Optional[Config] = None, use_gradnorm=False, use_homoscedastic=True):
        if config is None:
            config = get_config()
        self.model = model
        self.config = config
        self.mc = config.model
        self.device = config.device
        self.model.to(self.device)

        self.criterion = PhysicsConstrainedLoss(config=self.mc, use_gradnorm=use_gradnorm, use_homoscedastic=use_homoscedastic)
        if use_homoscedastic:
            self.criterion.to(self.device)
            all_params = list(model.parameters()) + list(self.criterion.parameters())
        else:
            all_params = list(model.parameters())

        self.optimizer = torch.optim.AdamW(all_params, lr=self.mc.learning_rate, weight_decay=self.mc.weight_decay)

        if self.mc.scheduler_type == "cosine_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.mc.scheduler_T_0, T_mult=self.mc.scheduler_T_mult, eta_min=self.mc.scheduler_eta_min)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.mc.max_epochs, eta_min=self.mc.scheduler_eta_min)

        self.scaler = GradScaler('cuda', enabled=self.mc.use_amp and torch.cuda.is_available())
        self.swag = None
        if self.mc.swa_start_epoch < self.mc.max_epochs:
            self.swag = SWAG(model, rank=self.mc.swag_rank, device=self.device)

        self.history: Dict[str, List[float]] = {k: [] for k in ['train_loss', 'val_loss', 'train_data', 'train_physics', 'train_mono', 'train_bound', 'val_data', 'val_physics', 'val_mono', 'val_bound', 'lr', 'violations', 'nfe']}
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _train_epoch(self, loader, epoch):
        self.model.train()
        totals = {'total': 0, 'data': 0, 'physics': 0, 'monotone': 0, 'bound': 0}
        nfe_total, n = 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        accum_steps = 0

        for step, batch in enumerate(loader):
            static = batch['static'].to(self.device)
            f_true = batch['traj'].to(self.device)
            t_span = batch['t_span'].to(self.device)
            f_eq = batch['f_eq'].to(self.device)
            dG = batch['dG_norm'].to(self.device)
            k_j, n_j = batch['k_jmak'].to(self.device), batch['n_jmak'].to(self.device)

            with autocast('cuda', enabled=self.mc.use_amp and torch.cuda.is_available()):
                try:
                    f_pred = self.model(static, f_eq, dG, t_span)
                    ml = min(f_pred.shape[1], f_true.shape[1])
                    shared = None
                    for m in self.model.ode_func.net:
                        if hasattr(m, 'weight'):
                            shared = m
                            break
                    t_loss = t_span[:, :ml]
                    loss_d = self.criterion(f_pred[:, :ml], f_true[:, :ml], f_eq, t_loss, k_j, n_j, self.model, shared, epoch)

                    # Provenance-aware weighting: upweight real data
                    provenance = batch.get('provenance', ['unknown'] * static.shape[0])
                    real_mask = torch.tensor([p in ('experimental', 'user_provided') for p in provenance],
                                             dtype=torch.float32, device=self.device)
                    if real_mask.sum() > 0 and hasattr(self.config.data, 'provenance_aware_loss') and self.config.data.provenance_aware_loss:
                        weight = 1.0 + (self.config.data.real_data_weight - 1.0) * real_mask.mean()
                        loss_d['total'] = loss_d['total'] * weight
                except Exception as e:
                    logger.warning(f"Skipped batch {step}: {e}")
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

            for k in totals:
                if k in loss_d:
                    totals[k] += loss_d[k].detach().item()
            nfe_total += self.model.ode_func.nfe
            self.model.ode_func.nfe = 0
            n += 1

        if accum_steps > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.mc.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return {k: v / max(n, 1) for k, v in totals.items()}, nfe_total / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        totals = {'total': 0, 'data': 0, 'physics': 0, 'monotone': 0, 'bound': 0}
        violations, n_pts, n = 0, 0, 0

        for batch in loader:
            static = batch['static'].to(self.device)
            f_true = batch['traj'].to(self.device)
            t_span = batch['t_span'].to(self.device)
            f_eq = batch['f_eq'].to(self.device)
            dG = batch['dG_norm'].to(self.device)
            k_j, n_j = batch['k_jmak'].to(self.device), batch['n_jmak'].to(self.device)

            try:
                f_pred = self.model(static, f_eq, dG, t_span)
                ml = min(f_pred.shape[1], f_true.shape[1])
                t_loss = t_span[:, :ml]
                loss_d = self.criterion(f_pred[:, :ml], f_true[:, :ml], f_eq, t_loss, k_j, n_j)
                for k in totals:
                    if k in loss_d:
                        totals[k] += loss_d[k].item()
                violations += (f_pred[:, 1:] - f_pred[:, :-1] < -0.001).sum().item()
                n_pts += f_pred.numel()
                n += 1
            except Exception as e:
                logger.warning(f"Validation batch error: {e}")
                continue

        return {k: v / max(n, 1) for k, v in totals.items()}, violations / max(n_pts, 1)

    def train(self, train_loader, val_loader, verbose=True):
        logger.info(f"{'='*60}")
        logger.info(f"Training on {self.device} | Epochs: {self.mc.max_epochs} | Batch: {self.mc.batch_size} | LR: {self.mc.learning_rate}")
        logger.info(f"AMP: {self.mc.use_amp} | Adjoint: {self.mc.adjoint} | SWAG: {self.swag is not None}")
        logger.info(f"{'='*60}")
        t0 = time.time()

        for epoch in range(1, self.mc.max_epochs + 1):
            train_loss, nfe = self._train_epoch(train_loader, epoch)
            val_loss, viol = self._validate(val_loader)
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
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['violations'].append(viol)
            self.history['nfe'].append(nfe)

            if self.swag and epoch >= self.mc.swa_start_epoch and epoch % self.mc.swag_collect_freq == 0:
                self.swag.collect()

            if val_loss['total'] < self.best_val_loss:
                self.best_val_loss = val_loss['total']
                self.patience_counter = 0
                self.save_checkpoint('best')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.mc.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if verbose and (epoch % 25 == 0 or epoch == 1):
                logger.info(f"E{epoch:>4d} | T:{train_loss['total']:.5f} V:{val_loss['total']:.5f} | NFE:{nfe:.0f} | Viol:{viol:.2%} | LR:{self.optimizer.param_groups[0]['lr']:.1e}")

        logger.info(f"Done in {(time.time()-t0)/60:.1f}m | Best: {self.best_val_loss:.6f}")
        self.save_checkpoint('last')
        return self.history

    def save_checkpoint(self, tag='best'):
        path = self.config.checkpoint_dir / f"physics_node_{tag}.pt"
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'best_val': self.best_val_loss, 'history': self.history, 'config': {'hidden_dims': self.mc.hidden_dims, 'latent_dim': self.mc.latent_dim, 'augmented_dim': self.mc.augmented_dim}}
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
        self.best_val_loss = ckpt['best_val']
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
        endpoint = float(sample['traj'][-1].item())
        duration = float(sample['t_span'][-1].item()) if sample['t_span'].numel() else 0.0
        provenance_weight = len(train_ds) / max(provenance_counts[prov], 1)
        endpoint_weight = 1.0 + 4.0 * min(endpoint, 0.5)
        duration_weight = 1.0 + 0.15 * min(np.log10(max(duration, 1.0)), 4.0)
        sample_weights.append(provenance_weight * endpoint_weight * duration_weight)
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config.model.batch_size, sampler=sampler, collate_fn=_collate_fn, drop_last=True, num_workers=0, pin_memory=config.device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=config.model.batch_size, shuffle=False, collate_fn=_collate_fn, num_workers=0, pin_memory=config.device.type == 'cuda')
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
    return trainer.best_val_loss


def run_optuna_hpo(train_df, val_df, config=None, n_trials=50):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if config is None:
        config = get_config()
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.model.random_seed))
    study.optimize(lambda t: optuna_objective(t, train_df, val_df, config), n_trials=n_trials, timeout=config.model.optuna_timeout, show_progress_bar=True)
    logger.info(f"Best: {study.best_trial.value:.6f} | {study.best_trial.params}")
    return {'best_params': study.best_trial.params, 'best_value': study.best_trial.value, 'study': study}


def train_ensemble(train_df, val_df, config=None):
    if config is None:
        config = get_config()
    ensemble = EnsembleNODE(config.model.n_ensemble_members, config.model)
    for i, (model, seed) in enumerate(zip(ensemble.models, config.model.ensemble_seeds)):
        logger.info(f"Ensemble {i+1}/{config.model.n_ensemble_members} seed={seed}")
        set_seed(seed)
        tr, va = create_data_loaders(train_df, val_df, config)
        trainer = Trainer(model, config)
        trainer.train(tr, va)
        torch.save(model.state_dict(), config.checkpoint_dir / f"ensemble_{i}.pt")
    return ensemble
