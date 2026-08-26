"""论文核心可靠性信号 PB、PD 及其在线聚合实现。"""

from dataclasses import dataclass
import math
import time
import torch


@dataclass
class SignalConfig:
    """可靠性信号超参数，与实验配置中的 ``controller`` 字段对应。"""

    alpha: float = 0.5
    beta: float = 0.9
    delta: float = 0.05
    prior_sigma: float = 0.05
    lambda_g: float = 1e-4
    lambda_u: float = 1e-3


class RunningZ:
    """用 Welford 算法维护仅依赖历史与当前观测的在线 z-score。

    在线统计避免使用完整训练轨迹的未来信息。首个样本没有可估标准差，按
    协议返回 0；后续使用样本方差 ``m2 / (n - 1)``。
    """

    def __init__(self, eps=1e-8):
        """建立空的在线均值/二阶中心矩状态。"""

        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = eps

    def update(self, x):
        """纳入新观测并返回基于更新后统计量的标准化值。"""

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
    """计算训练过程中的复杂度、动态风险与平滑可靠性指标。

    ``theta0`` 是模型初始化参数，用于计算相对初始化的复杂度；
    ``prev_state`` 是当前 epoch 开始时的参数，用于计算单 epoch 更新范数。
    两者语义不同，不能复用同一份快照。
    """

    def __init__(self, model, n_train: int, cfg: SignalConfig):
        """冻结初始化快照，并建立 PB/PD/验证损失的独立在线统计量。"""

        self.cfg = cfg
        self.n_train = int(n_train)
        # 固定先验中心 theta_0：整个 run 内不再更新。
        self.theta0 = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.prev_state = None
        self.z_pb = RunningZ()
        self.z_pd = RunningZ()
        self.z_val = RunningZ()
        self.rbar = None
        self.gradient_norm_seconds = 0.0
        self.update_norm_seconds = 0.0

    @staticmethod
    def _sync_model(model):
        """在 CUDA 计时边界同步设备，避免把异步内核时间漏出统计。"""

        parameter = next(model.parameters(), None)
        if parameter is not None and parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)

    @torch.no_grad()
    def begin_epoch(self, model):
        """记录 epoch 起点参数，并清零本 epoch 的控制器计时。"""

        self.gradient_norm_seconds = 0.0
        self.update_norm_seconds = 0.0
        self.prev_state = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def gradient_norm(self, model):
        """返回当前反向传播梯度的全参数 L2 范数。"""

        self._sync_model(model)
        started = time.perf_counter()
        terms = []
        for p in model.parameters():
            if p.grad is not None:
                terms.append(torch.sum(p.grad.detach() ** 2))
        value = torch.stack(terms).sum().sqrt().item() if terms else 0.0
        self._sync_model(model)
        self.gradient_norm_seconds += time.perf_counter() - started
        return value

    @torch.no_grad()
    def update_norm(self, model):
        """返回 ``||theta_t - theta_epoch_start||_2``。"""

        self._sync_model(model)
        started = time.perf_counter()
        if self.prev_state is None:
            return 0.0
        terms = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            diff = p.detach() - self.prev_state[name]
            terms.append(torch.sum(diff ** 2))
        value = torch.stack(terms).sum().sqrt().item() if terms else 0.0
        self._sync_model(model)
        self.update_norm_seconds += time.perf_counter() - started
        return value

    @torch.no_grad()
    def complexity_ct(self, model):
        """计算 ``C_t = ||theta_t-theta_0||^2 / (2*sigma^2)``。"""

        self._sync_model(model)
        terms = []
        sigma2 = self.cfg.prior_sigma ** 2
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            diff = p.detach() - self.theta0[name]
            terms.append(torch.sum(diff ** 2))
        total = torch.stack(terms).sum().item() if terms else 0.0
        self._sync_model(model)
        return total / (2.0 * sigma2)

    def compute(self, model, train_nll, val_nll, grad_norm, update_norm, variant="reliability"):
        """计算指定消融版本的 PB、PD、瞬时风险 ``r`` 与 EMA ``rbar``。

        PB 将验证 NLL 与相对初始化的复杂度惩罚组合；PD 描述验证/训练风险
        差、梯度范数和参数更新范数。各 ``variant`` 只替换论文消融所要求的
        项，其余在线统计和控制接口保持一致。
        """

        pb_started = time.perf_counter()
        c_t = self.complexity_ct(model)
        n = max(1, self.n_train)

        # PAC-Bayes 风格上界项：delta 控制置信水平，n 为训练子集样本数。
        pb = val_nll + math.sqrt(max(0.0, (c_t + math.log(2.0 * math.sqrt(n) / self.cfg.delta)) / (2.0 * n)))
        pb_seconds = time.perf_counter() - pb_started
        pd_started = time.perf_counter()
        # 只惩罚验证风险高于训练风险的部分，负泛化差截断为 0。
        v_t = max(0.0, val_nll - train_nll)
        pd_terms = {
            "reliability_pd_full": v_t + self.cfg.lambda_g * grad_norm + self.cfg.lambda_u * update_norm,
            "reliability_pd_v_only": v_t,
            "reliability_pd_v_h": v_t + self.cfg.lambda_g * grad_norm,
            "reliability_pd_v_u": v_t + self.cfg.lambda_u * update_norm,
            "reliability_pd_h_u": self.cfg.lambda_g * grad_norm + self.cfg.lambda_u * update_norm,
        }
        pd = pd_terms.get(variant, pd_terms["reliability_pd_full"])
        pd_seconds = self.gradient_norm_seconds + self.update_norm_seconds + (time.perf_counter() - pd_started)

        # PB/PD 数量级不同，先分别在线标准化，再按 alpha 融合。
        standardization_started = time.perf_counter()
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

        # beta 控制跨 epoch 的指数平滑；无平滑消融直接使用瞬时 r。
        if variant == "reliability_no_smoothing":
            self.rbar = r
        elif self.rbar is None:
            self.rbar = r
        else:
            self.rbar = self.cfg.beta * self.rbar + (1.0 - self.cfg.beta) * r
        standardization_seconds = time.perf_counter() - standardization_started

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
            "pb_computation_seconds": pb_seconds,
            "pd_computation_seconds": pd_seconds,
            "standardization_smoothing_seconds": standardization_seconds,
        }
