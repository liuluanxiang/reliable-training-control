import numpy as np
import torch


def _trapezoid(y, x=None):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.total / max(1, self.count)


def accuracy(logits, targets):
    return float((logits.argmax(dim=1) == targets).float().mean().item())


@torch.no_grad()
def topk_accuracy(logits, targets, k=5):
    k = min(k, logits.size(1))
    idx = logits.topk(k, dim=1).indices
    return float(idx.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item())


@torch.no_grad()
def expected_calibration_error(logits, targets, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for i in range(n_bins):
        mask = conf.gt(bins[i]) & conf.le(bins[i + 1])
        prop = mask.float().mean()
        if prop.item() > 0:
            ece += torch.abs(correct[mask].mean() - conf[mask].mean()) * prop
    return float(ece.item())


@torch.no_grad()
def maximum_calibration_error(logits, targets, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    mce = 0.0
    for i in range(n_bins):
        mask = conf.gt(bins[i]) & conf.le(bins[i + 1])
        if mask.float().sum().item() > 0:
            gap = torch.abs(correct[mask].mean() - conf[mask].mean()).item()
            mce = max(mce, gap)
    return float(mce)


@torch.no_grad()
def brier_score(logits, targets):
    probs = torch.softmax(logits, dim=1)
    onehot = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
    return float(((probs - onehot) ** 2).sum(dim=1).mean().item())


@torch.no_grad()
def confidence_stats(logits):
    probs = torch.softmax(logits, dim=1)
    conf, _ = probs.max(dim=1)
    sorted_probs, _ = probs.sort(dim=1, descending=True)
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "confidence_mean": float(conf.mean().item()),
        "confidence_std": float(conf.std(unbiased=True).item()) if len(conf) > 1 else 0.0,
        "margin_mean": float(margin.mean().item()),
        "entropy_mean": float(entropy.mean().item()),
    }


@torch.no_grad()
def reliability_bins(logits, targets, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    rows = []
    for i in range(n_bins):
        lo, hi = float(bins[i].item()), float(bins[i + 1].item())
        mask = conf.gt(bins[i]) & conf.le(bins[i + 1])
        count = int(mask.float().sum().item())
        if count > 0:
            rows.append({
                "bin_low": lo,
                "bin_high": hi,
                "confidence": float(conf[mask].mean().item()),
                "accuracy": float(correct[mask].mean().item()),
                "count": count,
            })
        else:
            rows.append({
                "bin_low": lo,
                "bin_high": hi,
                "confidence": (lo + hi) / 2.0,
                "accuracy": np.nan,
                "count": 0,
            })
    return rows


@torch.no_grad()
def risk_coverage_curve(logits, targets):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(targets).float()
    order = torch.argsort(conf, descending=True)
    correct = correct[order].cpu().numpy()
    n = len(correct)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - np.cumsum(correct) / np.arange(1, n + 1)
    return coverage, risk


@torch.no_grad()
def aurc_eaurc(logits, targets):
    coverage, risk = risk_coverage_curve(logits, targets)
    aurc = float(_trapezoid(risk, coverage))
    error = 1.0 - float((logits.argmax(dim=1) == targets).float().mean().item())
    # approximate optimal AURC; clamp to avoid log(0)
    optimal = error + (1.0 - error) * np.log(max(1e-12, 1.0 - error))
    eaurc = float(aurc - optimal)
    return aurc, eaurc


def area_under_curve(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values[0]) if len(values) else np.nan
    return float(_trapezoid(values) / (len(values) - 1))


def slope_last(values, window=10):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    w = min(window, len(values) - 1)
    return float(np.mean(np.diff(values[-(w + 1):])))


def spearman_np(x, y):
    import pandas as pd
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    rx = pd.Series(x[mask]).rank(method="average").values
    ry = pd.Series(y[mask]).rank(method="average").values
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson_np(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def mean_std_text(values, digits=4):
    import pandas as pd
    s = pd.Series(values).dropna().astype(float)
    if len(s) == 0:
        return ""
    if len(s) == 1:
        return f"{s.iloc[0]:.{digits}f}"
    return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}"
