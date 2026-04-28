from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config, ModelConfig


class HomoscedasticUncertainty(nn.Module):
    def __init__(self, n_tasks=4, init_log_vars=None):
        super().__init__()
        if init_log_vars is None:
            init_log_vars = [0.0, 1.0, 0.0, -1.0]
        self.log_vars = nn.ParameterList([nn.Parameter(torch.tensor(v)) for v in init_log_vars])

    def get_weights(self):
        return [torch.exp(-lv) for lv in self.log_vars]

    def regularization(self):
        return sum(lv for lv in self.log_vars)


class GradNormBalancer:
    def __init__(self, n_tasks, alpha=1.5, lr=0.025):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.lr = lr
        self.weights = torch.ones(n_tasks)
        self.initial_losses = None

    def update(self, losses, shared_layer, epoch):
        if shared_layer is None or not shared_layer.weight.requires_grad:
            return self.weights
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.detach().item() for l in losses])
        grads = []
        for i, loss_value in enumerate(losses):
            if loss_value.requires_grad:
                grad = torch.autograd.grad(
                    self.weights[i] * loss_value,
                    shared_layer.weight,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                grads.append(grad.norm() if grad is not None else torch.tensor(0.0))
            else:
                grads.append(torch.tensor(0.0))
        grads_t = torch.stack(grads)
        mean_grad = grads_t.mean()
        current_losses = torch.tensor([loss_value.detach().item() for loss_value in losses])
        ratios = current_losses / (self.initial_losses + 1e-8)
        target_grads = mean_grad * (ratios / ratios.mean()) ** self.alpha
        self.weights = self.weights * torch.exp(-self.lr * (grads_t - target_grads).sign())
        self.weights = self.weights * self.n_tasks / self.weights.sum()
        return self.weights


class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, use_gradnorm=False, use_homoscedastic=False):
        super().__init__()
        if config is None:
            config = get_config().model
        self.config = config
        self.use_homoscedastic = use_homoscedastic

        if use_homoscedastic:
            init = list(config.homoscedastic_init.values()) if hasattr(config, 'homoscedastic_init') else [0, 1, 0, -1]
            self.uncertainty = HomoscedasticUncertainty(4, init)
        self.gradnorm = GradNormBalancer(4) if use_gradnorm else None

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return values.mean()
        mask = mask.to(values.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    def _data_loss(self, f_pred, f_true, obs_mask=None):
        delta = 0.05
        abs_err = (f_pred - f_true).abs()
        quadratic = torch.clamp(abs_err, max=delta)
        linear = abs_err - quadratic
        huber = 0.5 * quadratic.pow(2) + delta * linear
        mse = (f_pred - f_true).pow(2)
        huber_loss = self._masked_mean(huber, obs_mask)
        mse_loss = self._masked_mean(mse, obs_mask)
        return 0.7 * huber_loss + 0.3 * mse_loss

    def _physics_loss(self, f_pred, t_span, k_jmak, n_jmak, f_eq, point_mask=None):
        if t_span.dim() == 1:
            t_eval = t_span.unsqueeze(0).expand(f_pred.shape[0], -1)
        else:
            t_eval = t_span
        t_eval = t_eval.detach()
        valid_mask = point_mask.bool() if point_mask is not None else torch.ones_like(f_pred, dtype=torch.bool)
        f_eq_expanded = f_eq.squeeze(-1).unsqueeze(1)
        f_jmak = f_eq_expanded * (1.0 - torch.exp(-k_jmak.unsqueeze(1) * t_eval.pow(n_jmak.unsqueeze(1))))
        f_jmak = torch.minimum(f_jmak.clamp_min(0.0), f_eq_expanded)
        last_valid_time = torch.where(valid_mask, t_eval, torch.zeros_like(t_eval)).amax(dim=1, keepdim=True).clamp_min(1e-6)
        early_mask = valid_mask & (t_eval <= 0.5 * last_valid_time)
        if early_mask.any():
            target_mask = early_mask
        else:
            target_mask = valid_mask
        return self._masked_mean((f_pred - f_jmak).pow(2), target_mask)

    def _monotonicity_loss(self, f_pred, point_mask=None):
        if f_pred.shape[1] < 2:
            return torch.zeros((), device=f_pred.device)
        df = f_pred[:, 1:] - f_pred[:, :-1]
        violations = F.relu(-df).pow(2)
        if point_mask is None:
            return violations.mean()
        pair_mask = point_mask[:, 1:] * point_mask[:, :-1]
        return self._masked_mean(violations, pair_mask)

    def _bound_loss(self, f_pred, f_eq, point_mask=None):
        excess = F.relu(f_pred - f_eq.squeeze(-1).unsqueeze(1)).pow(2)
        return self._masked_mean(excess, point_mask)

    def forward(
        self,
        f_pred,
        f_true,
        f_eq,
        t_span,
        k_jmak=None,
        n_jmak=None,
        model=None,
        shared_layer=None,
        epoch=0,
        point_mask=None,
        obs_mask=None,
    ) -> Dict[str, torch.Tensor]:
        if obs_mask is None:
            obs_mask = point_mask
        L_data = self._data_loss(f_pred, f_true, obs_mask=obs_mask)
        L_mono = self._monotonicity_loss(f_pred, point_mask=point_mask)
        L_bound = self._bound_loss(f_pred, f_eq, point_mask=point_mask)

        if k_jmak is not None and n_jmak is not None:
            L_phys = self._physics_loss(f_pred, t_span, k_jmak, n_jmak, f_eq, point_mask=point_mask)
        else:
            L_phys = torch.tensor(0.0, device=f_pred.device)

        losses = [L_data, L_phys, L_mono, L_bound]

        if self.use_homoscedastic:
            weights = self.uncertainty.get_weights()
            total = sum(weight * loss_value for weight, loss_value in zip(weights, losses)) + self.uncertainty.regularization()
        elif self.gradnorm is not None:
            weights = self.gradnorm.update(losses, shared_layer, epoch) if shared_layer else torch.ones(4, device=f_pred.device)
            total = sum(weight * loss_value for weight, loss_value in zip(weights, losses))
        else:
            total = L_data + 0.1 * L_phys + 0.5 * L_mono + 0.5 * L_bound

        return {
            'total': total,
            'data': L_data,
            'physics': L_phys,
            'monotone': L_mono,
            'bound': L_bound,
        }
