import numpy as np
import warnings

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - graceful fallback for minimal environments.
    scipy_stats = None


def _clean(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def mean_ci(values, ci=0.95):
    vals = _clean(values)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    if len(vals) == 1:
        return mean, mean, mean
    alpha = 1.0 - ci
    if scipy_stats is not None:
        half = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals)))
    else:
        half = float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return mean, mean - half, mean + half


def paired_ttest(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or scipy_stats is None:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(scipy_stats.ttest_rel(a[mask], b[mask], nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def wilcoxon_test(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or scipy_stats is None:
        return np.nan
    try:
        return float(scipy_stats.wilcoxon(a[mask], b[mask]).pvalue)
    except Exception:
        return np.nan


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    diff = a[mask] - b[mask]
    sd = np.std(diff, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.mean(diff) / sd)


def format_p_value(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def p_stars(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def spearman_with_p(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan, np.nan
    if scipy_stats is None:
        return np.nan, np.nan
    try:
        res = scipy_stats.spearmanr(a[mask], b[mask])
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan
