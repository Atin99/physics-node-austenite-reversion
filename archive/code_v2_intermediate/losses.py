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
        for i, l in enumerate(losses):
            if l.requires_grad:
                g = torch.autograd.grad(self.weights[i] * l, shared_layer.weight, retain_graph=True, allow_unused=True)[0]
                grads.append(g.norm() if g is not None else torch.tensor(0.0))
            else:
                grads.append(torch.tensor(0.0))
        grads_t = torch.stack(grads)
        mean_grad = grads_t.mean()
        current_losses = torch.tensor([l.detach().item() for l in losses])
        ratios = current_losses / (self.initial_losses + 1e-8)
        target_grads = mean_grad * (ratios / ratios.mean()) ** self.alpha
        grad_loss = (grads_t - target_grads).abs().sum()
        self.weights = self.weights * torch.exp(-self.lr * (grads_t - target_grads).sign())
        self.weights = self.weights * self.n_tasks / self.weights.sum()
        return self.weights


class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, use_gradnorm=False, use_homoscedastic=True):
        super().__init__()
        if config is None:
            config = get_config().model
        self.config = config
        self.use_homoscedastic = use_homoscedastic

        if use_homoscedastic:
            init = list(config.homoscedastic_init.values()) if hasattr(config, 'homoscedastic_init') else [0, 1, 0, -1]
            self.uncertainty = HomoscedasticUncertainty(4, init)
        self.gradnorm = GradNormBalancer(4) if use_gradnorm else None

    def _data_loss(self, f_pred, f_true):
        huber = F.huber_loss(f_pred, f_true, delta=0.05)
        mse = F.mse_loss(f_pred, f_true)
        return 0.7 * huber + 0.3 * mse

    def _physics_loss(self, f_pred, t_span, k_jmak, n_jmak, f_eq):
        if t_span.dim() == 1:
            t_eval = t_span.unsqueeze(0).expand(f_pred.shape[0], -1)
        else:
            t_eval = t_span
        t_eval = t_eval.detach()
        f_eq_expanded = f_eq.squeeze(-1).unsqueeze(1)
        f_jmak = f_eq_expanded * (1 - torch.exp(-k_jmak.unsqueeze(1) * t_eval.pow(n_jmak.unsqueeze(1))))
        f_jmak = torch.minimum(f_jmak.clamp_min(0.0), f_eq_expanded)
        mask = t_eval < (t_eval[:, -1:].clamp_min(1e-6) * 0.5)
        if mask.sum() > 0:
            return F.mse_loss(f_pred[mask], f_jmak[mask])
        return F.mse_loss(f_pred, f_jmak)

    def _monotonicity_loss(self, f_pred):
        df = f_pred[:, 1:] - f_pred[:, :-1]
        violations = F.relu(-df)
        return violations.pow(2).mean()

    def _bound_loss(self, f_pred, f_eq):
        excess = F.relu(f_pred - f_eq.squeeze(-1).unsqueeze(1))
        return excess.pow(2).mean()

    def forward(self, f_pred, f_true, f_eq, t_span, k_jmak=None, n_jmak=None, model=None, shared_layer=None, epoch=0):
        L_data = self._data_loss(f_pred, f_true)
        L_mono = self._monotonicity_loss(f_pred)
        L_bound = self._bound_loss(f_pred, f_eq)

        if k_jmak is not None and n_jmak is not None:
            L_phys = self._physics_loss(f_pred, t_span, k_jmak, n_jmak, f_eq)
        else:
            L_phys = torch.tensor(0.0, device=f_pred.device)

        losses = [L_data, L_phys, L_mono, L_bound]

        if self.use_homoscedastic:
            weights = self.uncertainty.get_weights()
            total = sum(w * l for w, l in zip(weights, losses)) + self.uncertainty.regularization()
        elif self.gradnorm is not None:
            w = self.gradnorm.update(losses, shared_layer, epoch) if shared_layer else torch.ones(4)
            total = sum(wi * li for wi, li in zip(w, losses))
        else:
            total = L_data + 0.1 * L_phys + 0.5 * L_mono + 0.5 * L_bound

        return {
            'total': total, 'data': L_data, 'physics': L_phys,
            'monotone': L_mono, 'bound': L_bound
        }
