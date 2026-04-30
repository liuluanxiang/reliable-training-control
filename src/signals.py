from dataclasses import dataclass
import math
import torch


@dataclass
class SignalConfig:
    alpha: float = 0.5
    beta: float = 0.9
    delta: float = 0.05
    prior_sigma: float = 0.05
    lambda_g: float = 1e-4
    lambda_u: float = 1e-3


class RunningZ:
    def __init__(self, eps=1e-8):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = eps

    def update(self, x):
        x = float(x)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if self.n < 2:
            return 0.0
        std = max((self.m2 / (self.n - 1)) ** 0.5, self.eps)
        return (x - self.mean) / std


class ReliabilitySignals:
    def __init__(self, model, n_train: int, cfg: SignalConfig):
        self.cfg = cfg
        self.n_train = int(n_train)
        self.theta0 = {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.prev_state = None
        self.z_pb = RunningZ()
        self.z_pd = RunningZ()
        self.z_val = RunningZ()
        self.rbar = None

    @torch.no_grad()
    def begin_epoch(self, model):
        self.prev_state = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def gradient_norm(self, model):
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += torch.sum(p.grad.detach() ** 2).item()
        return total ** 0.5

    @torch.no_grad()
    def update_norm(self, model):
        if self.prev_state is None:
            return 0.0
        total = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            diff = p.detach() - self.prev_state[name].to(p.device)
            total += torch.sum(diff ** 2).item()
        return total ** 0.5

    @torch.no_grad()
    def complexity_ct(self, model):
        total = 0.0
        sigma2 = self.cfg.prior_sigma ** 2
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            diff = p.detach() - self.theta0[name].to(p.device)
            total += torch.sum(diff ** 2).item()
        return total / (2.0 * sigma2)

    def compute(self, model, train_nll, val_nll, grad_norm, update_norm, variant="reliability"):
        c_t = self.complexity_ct(model)
        n = max(1, self.n_train)

        pb = val_nll + math.sqrt(max(0.0, (c_t + math.log(2.0 * math.sqrt(n) / self.cfg.delta)) / (2.0 * n)))
        v_t = max(0.0, val_nll - train_nll)
        pd = v_t + self.cfg.lambda_g * grad_norm + self.cfg.lambda_u * update_norm

        pb_z = self.z_pb.update(pb)
        pd_z = self.z_pd.update(pd)
        val_z = self.z_val.update(val_nll)

        if variant == "reliability_pb_only":
            r = pb_z
        elif variant == "reliability_pd_only":
            r = pd_z
        elif variant == "reliability_val_loss_only":
            r = val_z
        else:
            r = self.cfg.alpha * pb_z + (1.0 - self.cfg.alpha) * pd_z

        if variant == "reliability_no_smoothing":
            self.rbar = r
        elif self.rbar is None:
            self.rbar = r
        else:
            self.rbar = self.cfg.beta * self.rbar + (1.0 - self.cfg.beta) * r

        return {
            "complexity_ct": c_t,
            "pb_signal": pb,
            "risk_violation_vt": v_t,
            "pd_signal": pd,
            "pb_z": pb_z,
            "pd_z": pd_z,
            "val_loss_z": val_z,
            "reliability_r": r,
            "reliability_rbar": self.rbar,
        }
