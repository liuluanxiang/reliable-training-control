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


def mean_std(values):
    vals = _clean(values)
    if len(vals) == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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


def significance_marker(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


HIGHER_BETTER = {"test_acc", "test_top5_acc", "aulc_val_acc"}
LOWER_BETTER = {
    "test_nll", "test_ece", "test_mce", "test_brier", "test_aurc", "test_eaurc",
    "final_gen_gap_loss", "late_degradation",
}
BASELINE_METHODS = ["step", "cosine", "plateau"]


def metric_higher_is_better(metric):
    if metric in HIGHER_BETTER:
        return True
    if metric in LOWER_BETTER:
        return False
    return True


def compare_ours_vs_best_baseline(summary_df, metric, group_cols):
    import pandas as pd

    columns = list(group_cols) + [
        "metric", "ours", "best_baseline", "best_baseline_name", "mean_difference",
        "cohen_d", "paired_ttest_p", "wilcoxon_p", "significance", "n_pairs",
    ]
    if summary_df is None or len(summary_df) == 0 or metric not in summary_df.columns:
        return pd.DataFrame(columns=columns)
    if "method" not in summary_df.columns or "seed" not in summary_df.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    higher = metric_higher_is_better(metric)
    for keys, group in summary_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ours = group[group["method"] == "reliability"]
        baselines = group[group["method"].isin(BASELINE_METHODS)]
        if ours.empty or baselines.empty:
            continue

        baseline_means = baselines.groupby("method")[metric].mean(numeric_only=True).dropna()
        if baseline_means.empty:
            continue
        best_name = baseline_means.idxmax() if higher else baseline_means.idxmin()
        best = baselines[baselines["method"] == best_name]

        ours_by_seed = ours.set_index("seed")[metric]
        best_by_seed = best.set_index("seed")[metric]
        seeds = ours_by_seed.index.intersection(best_by_seed.index)
        if len(seeds) >= 2:
            ours_vals = ours_by_seed.loc[seeds].to_numpy(dtype=float)
            best_vals = best_by_seed.loc[seeds].to_numpy(dtype=float)
            t_p = paired_ttest(ours_vals, best_vals)
            w_p = wilcoxon_test(ours_vals, best_vals)
            d = cohens_d(ours_vals, best_vals)
            diff = float(np.nanmean(ours_vals - best_vals))
            marker = significance_marker(t_p)
        else:
            ours_vals = np.asarray([], dtype=float)
            best_vals = np.asarray([], dtype=float)
            t_p = np.nan
            w_p = np.nan
            d = np.nan
            diff = np.nan
            marker = "n/a"

        ours_mean = float(pd.to_numeric(ours[metric], errors="coerce").mean())
        best_mean = float(pd.to_numeric(best[metric], errors="coerce").mean())
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update({
            "metric": metric,
            "ours": ours_mean,
            "best_baseline": best_mean,
            "best_baseline_name": best_name,
            "mean_difference": diff,
            "cohen_d": d,
            "paired_ttest_p": t_p,
            "wilcoxon_p": w_p,
            "significance": marker,
            "n_pairs": int(len(seeds)),
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


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
