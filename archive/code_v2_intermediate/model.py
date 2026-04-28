import math
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    from torchdiffeq import odeint
    odeint_adjoint = odeint

from config import get_config, ModelConfig


def physics_gate(f, f_eq, dG, epsilon_nuc=0.005):
    driving = torch.abs(dG)
    saturation = F.relu(f_eq - f)
    nucleation = f + epsilon_nuc
    return driving * saturation * nucleation


class SiLUGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.silu(b)


# (Concrete Dropout removed - violates ODE deterministic continuous vector field assumption for dopri5)


class CompositionEncoder(nn.Module):
    def __init__(self, in_dim=5, embed_dim=32, n_heads=4, use_sn=True):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)
        self.element_embed = nn.Embedding(in_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.05)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        if use_sn:
            self.proj = spectral_norm(self.proj)

    def forward(self, comp_vec):
        B, D = comp_vec.shape
        idx = torch.arange(D, device=comp_vec.device).unsqueeze(0).expand(B, -1)
        val_embed = self.embed(comp_vec.unsqueeze(-1))
        elem_embed = self.element_embed(idx)
        tokens = val_embed + elem_embed
        attn_out, _ = self.attn(tokens, tokens, tokens)
        pooled = self.norm(attn_out.mean(dim=1))
        return self.proj(pooled)


class FiLMConditioner(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        return x * (1 + self.gamma(cond)) + self.beta(cond)


class AugmentedODEFunc(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        aug = config.augmented_dim
        state_dim = 1 + aug
        cond_dim = config.composition_embed_dim + 5
        total_in = state_dim + cond_dim

        dims = config.hidden_dims
        layers = []
        in_d = total_in
        for hd in dims:
            lin = nn.Linear(in_d, hd)
            if config.use_spectral_norm:
                lin = spectral_norm(lin)
            layers.extend([lin, nn.SiLU()])
            in_d = hd
        self.net = nn.Sequential(*layers)
        self.film_layers = nn.ModuleList([FiLMConditioner(cond_dim, hd) for hd in dims])
        self.rate_head = nn.Linear(in_d, 1)
        self.aug_head = nn.Linear(in_d, aug) if aug > 0 else None
        self.epsilon_nuc = get_config().physics.epsilon_nucleation
        self.nfe = 0
        self._conditioning = None
        self._f_eq = None
        self._dG = None
        self._reset_output_heads()

    def _reset_output_heads(self):
        nn.init.normal_(self.rate_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.rate_head.bias, -0.4)
        if self.aug_head is not None:
            nn.init.normal_(self.aug_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.aug_head.bias)

    def set_conditioning(self, cond, f_eq=None, dG=None):
        self._conditioning = cond
        if f_eq is not None:
            self._f_eq = f_eq
        if dG is not None:
            self._dG = dG

    def _encode_state(self, state, cond):
        x = torch.cat([state, cond], dim=-1)
        idx = 0
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.SiLU) and idx < len(self.film_layers):
                x = self.film_layers[idx](x, cond)
                idx += 1
        return x

    def forward(self, t, state):
        self.nfe += 1
        cond = self._conditioning
        hidden = self._encode_state(state, cond)

        f_eq = torch.clamp(self._f_eq, min=1e-6)
        f_now = torch.clamp(state[:, :1], min=0.0)
        rate = F.softplus(self.rate_head(hidden)) * 0.01
        drive = torch.abs(self._dG) + 0.1
        saturation = torch.clamp(f_eq - f_now, min=0.0) + 1e-6
        nucleation = f_now + self.epsilon_nuc
        df_dt = rate * drive * saturation * nucleation

        if self.aug_head is None:
            return df_dt

        aug_dt = 0.1 * torch.tanh(self.aug_head(hidden))
        return torch.cat([df_dt, aug_dt], dim=-1)


class PhysicsNODE(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = get_config().model
        self.config = config

        self.comp_encoder = CompositionEncoder(
            in_dim=5, embed_dim=config.composition_embed_dim,
            n_heads=config.n_attention_heads, use_sn=config.use_spectral_norm,
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(5, config.composition_embed_dim),
            nn.SiLU(),
            nn.Linear(config.composition_embed_dim, 5),
        )
        self.ode_func = AugmentedODEFunc(config)

        phys = get_config().physics
        self.epsilon_nuc = phys.epsilon_nucleation
        self.f_eq_tolerance = config.f_eq_tolerance
        self.f_initial = config.f_initial
        self.aug_dim = config.augmented_dim
        self.use_adjoint = config.adjoint

    def _build_conditioning(self, static_features, f_eq, delta_G_norm):
        comp_input = static_features[:, :5]
        comp_embed = self.comp_encoder(comp_input)
        phys = torch.cat([static_features[:, 5:8], f_eq, delta_G_norm], dim=-1)
        phys_out = self.physics_proj(phys)
        return torch.cat([comp_embed, phys_out], dim=-1)

    def _solve_trajectory(self, cond, f_eq, delta_G_norm, t_eval):
        device = cond.device
        y0 = torch.zeros(cond.shape[0], 1 + self.aug_dim, device=device)
        y0[:, 0] = self.f_initial
        solver = odeint_adjoint if self.use_adjoint and self.training else odeint
        self.ode_func.set_conditioning(cond, f_eq, delta_G_norm)
        trajectory = solver(
            self.ode_func, y0, t_eval,
            method=self.config.solver,
            rtol=self.config.rtol, atol=self.config.atol,
            options={'max_num_steps': self.config.max_num_steps},
        )
        f_raw = trajectory[:, :, 0].transpose(0, 1)
        f_nonneg = torch.clamp(f_raw, min=0.0)
        upper = (f_eq.squeeze(-1) * self.f_eq_tolerance).unsqueeze(1).expand_as(f_nonneg)
        return torch.min(f_nonneg, upper)

    def forward(self, static_features, f_eq, delta_G_norm, t_span):
        B = static_features.shape[0]
        cond = self._build_conditioning(static_features, f_eq, delta_G_norm)
        self.ode_func.nfe = 0
        if t_span.dim() == 1:
            return self._solve_trajectory(cond, f_eq, delta_G_norm, t_span)
        if t_span.dim() != 2 or t_span.shape[0] != B:
            raise ValueError(f"Expected t_span with shape [T] or [B, T], got {tuple(t_span.shape)}")

        trajectories = []
        for i in range(B):
            traj_i = self._solve_trajectory(cond[i:i + 1], f_eq[i:i + 1], delta_G_norm[i:i + 1], t_span[i])
            trajectories.append(traj_i.squeeze(0))
        return torch.stack(trajectories, dim=0)

    def predict_with_uncertainty(self, static_features, f_eq, delta_G_norm, t_span, n_samples=100):
        was_training = self.training
        self.train()
        preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(static_features, f_eq, delta_G_norm, t_span)
                preds.append(pred)
        if not was_training:
            self.eval()
        stacked = torch.stack(preds)
        alpha = 1.0 - self.config.confidence_level
        lo = alpha / 2
        hi = 1.0 - lo
        mean = stacked.mean(0)
        lower = stacked.quantile(lo, dim=0)
        upper = stacked.quantile(hi, dim=0)
        return mean, lower, upper

    def get_model_summary(self):
        n = sum(p.numel() for p in self.parameters())
        nt = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"PhysicsNODE | Params: {n:,} ({nt:,} trainable) | Latent: {self.config.latent_dim} | Aug: {self.aug_dim} | Adjoint: {self.use_adjoint}"


class SWAG(nn.Module):
    def __init__(self, base_model, rank=20, device=None):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.dev = device or torch.device('cpu')
        self.n_collected = 0
        self.mean = {}
        self.sq_mean = {}
        self.deviations = {}
        for name, param in base_model.named_parameters():
            self.mean[name] = torch.zeros_like(param.data, device=self.dev)
            self.sq_mean[name] = torch.zeros_like(param.data, device=self.dev)
            self.deviations[name] = []

    def collect(self):
        for name, param in self.base_model.named_parameters():
            self.mean[name] = (self.n_collected * self.mean[name] + param.data.to(self.dev)) / (self.n_collected + 1)
            self.sq_mean[name] = (self.n_collected * self.sq_mean[name] + param.data.to(self.dev) ** 2) / (self.n_collected + 1)
            dev = (param.data.to(self.dev) - self.mean[name]).unsqueeze(0)
            if len(self.deviations[name]) >= self.rank:
                self.deviations[name].pop(0)
            self.deviations[name].append(dev)
        self.n_collected += 1

    def sample(self, scale=1.0):
        for name, param in self.base_model.named_parameters():
            var = torch.clamp(self.sq_mean[name] - self.mean[name] ** 2, min=1e-30)
            diag_sample = self.mean[name] + scale * torch.sqrt(var) * torch.randn_like(var)
            if self.deviations[name]:
                D = torch.cat(self.deviations[name], dim=0)
                z = torch.randn(D.shape[0], device=self.dev)
                lr_sample = (z @ D.view(D.shape[0], -1)).view(param.shape) / math.sqrt(2 * (self.rank - 1))
            else:
                lr_sample = 0
            param.data = diag_sample + scale * lr_sample

    def predict_with_uncertainty(self, forward_fn, n_samples=30, **kwargs):
        preds = []
        for _ in range(n_samples):
            self.sample()
            with torch.no_grad():
                preds.append(forward_fn(**kwargs))
        stacked = torch.stack(preds)
        return stacked.mean(0), stacked.std(0), stacked.quantile(0.025, 0), stacked.quantile(0.975, 0)


class EnsembleNODE(nn.Module):
    def __init__(self, n_members=5, config=None):
        super().__init__()
        if config is None:
            config = get_config().model
        self.models = nn.ModuleList([PhysicsNODE(config) for _ in range(n_members)])

    def forward(self, *args, **kwargs):
        preds = [m(*args, **kwargs) for m in self.models]
        stacked = torch.stack(preds)
        return stacked.mean(0), stacked.std(0)
