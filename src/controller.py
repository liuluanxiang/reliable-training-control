from dataclasses import dataclass


@dataclass
class ControllerConfig:
    gamma: float = 0.5
    min_lr: float = 1e-5
    p_lr: int = 8
    p_stop: int = 18
    epsilon: float = 1e-4


class ReliabilityController:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.best_rbar = float("inf")
        self.best_val_loss = float("inf")
        self.bad_rel = 0
        self.bad_val = 0
        self.lr_reductions = 0
        self.first_lr_drop_epoch = -1
        self.stop_reason = ""

    def step(self, optimizer, rbar, val_loss, epoch):
        lr_reduced = False
        early_stop = False

        if rbar < self.best_rbar - self.cfg.epsilon:
            self.best_rbar = rbar
            self.bad_rel = 0
        else:
            self.bad_rel += 1

        if val_loss < self.best_val_loss - self.cfg.epsilon:
            self.best_val_loss = val_loss
            self.bad_val = 0
        else:
            self.bad_val += 1

        if self.bad_rel >= self.cfg.p_lr:
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * self.cfg.gamma, self.cfg.min_lr)
            self.bad_rel = 0
            self.lr_reductions += 1
            if self.first_lr_drop_epoch < 0:
                self.first_lr_drop_epoch = epoch
            lr_reduced = True

        if self.bad_val >= self.cfg.p_stop or self.bad_rel >= self.cfg.p_stop:
            early_stop = True
            self.stop_reason = "patience_exceeded"

        return {
            "lr_reduced": lr_reduced,
            "early_stop": early_stop,
            "stop_reason": self.stop_reason,
            "bad_rel": self.bad_rel,
            "bad_val": self.bad_val,
            "first_lr_drop_epoch": self.first_lr_drop_epoch,
            "lr_reductions": self.lr_reductions,
        }
