"""基于可靠性信号的学习率调整与提前停止控制器。"""

from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """控制器超参数。

    ``p_lr`` 和 ``p_stop`` 分别控制降学习率与停止训练的耐心窗口；二者必须
    使用独立计数器，否则一次降学习率会错误地延后可靠性提前停止。
    """

    gamma: float = 0.5
    min_lr: float = 1e-5
    p_lr: int = 8
    p_stop: int = 18
    epsilon: float = 1e-4


class ReliabilityController:
    """跟踪验证损失与平滑可靠性指标，并给出训练控制动作。

    该类只负责决策，不执行 epoch 训练或验证。优化器由调用方传入，因此
    学习率修改会直接作用于当前训练过程。
    """

    def __init__(self, cfg: ControllerConfig):
        """初始化最优值、独立耐心计数器和动作审计字段。"""

        self.cfg = cfg
        # 两个最优值都按“越小越好”解释，并使用 epsilon 抑制数值噪声。
        self.best_rbar = float("inf")
        self.best_val_loss = float("inf")
        # 降学习率和提前停止各自保留可靠性停滞计数，避免动作间相互干扰。
        self.bad_rel_lr = 0
        self.bad_rel_stop = 0
        self.bad_val = 0
        self.lr_reductions = 0
        self.first_lr_drop_epoch = -1
        self.stop_reason = ""

    def step(self, optimizer, rbar, val_loss, epoch):
        """消费一个 epoch 的观测，返回学习率与停止决策及审计状态。"""

        lr_reduced = False
        early_stop = False

        # 可靠性改善时同时重置两个窗口；否则分别累积相同的停滞事实。
        if rbar < self.best_rbar - self.cfg.epsilon:
            self.best_rbar = rbar
            self.bad_rel_lr = 0
            self.bad_rel_stop = 0
        else:
            self.bad_rel_lr += 1
            self.bad_rel_stop += 1

        if val_loss < self.best_val_loss - self.cfg.epsilon:
            self.best_val_loss = val_loss
            self.bad_val = 0
        else:
            self.bad_val += 1

        if self.bad_rel_lr >= self.cfg.p_lr:
            for group in optimizer.param_groups:
                old_lr = group["lr"]
                group["lr"] = max(old_lr * self.cfg.gamma, self.cfg.min_lr)
                lr_reduced = lr_reduced or group["lr"] < old_lr
            self.bad_rel_lr = 0
            if lr_reduced:
                self.lr_reductions += 1
                if self.first_lr_drop_epoch < 0:
                    self.first_lr_drop_epoch = epoch

        # 验证性能或可靠性任一长期停滞，都足以结束自适应预算实验。
        if self.bad_val >= self.cfg.p_stop or self.bad_rel_stop >= self.cfg.p_stop:
            early_stop = True
            reasons = []
            if self.bad_val >= self.cfg.p_stop:
                reasons.append("validation_stagnation")
            if self.bad_rel_stop >= self.cfg.p_stop:
                reasons.append("reliability_stagnation")
            self.stop_reason = "+".join(reasons)

        return {
            "lr_reduced": lr_reduced,
            "early_stop": early_stop,
            "stop_reason": self.stop_reason,
            "bad_rel": self.bad_rel_stop,
            "bad_rel_lr": self.bad_rel_lr,
            "bad_rel_stop": self.bad_rel_stop,
            "bad_val": self.bad_val,
            "first_lr_drop_epoch": self.first_lr_drop_epoch,
            "lr_reductions": self.lr_reductions,
        }
