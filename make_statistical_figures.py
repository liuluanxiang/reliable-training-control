import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import (  # noqa: E402
    PALETTE, add_panel_label, save_figure, set_nature_style,
)
from src.results_io import filter_quick  # noqa: E402
from src.stats_utils import compare_ours_vs_best_baseline, mean_ci  # noqa: E402


METHOD_LABELS = {
    "reliability": "Ours / Full",
    "reliability_full": "Ours / Full",
    "reliability_pb_only": "PB-only",
    "reliability_pd_only": "PD-only",
    "reliability_no_smoothing": "No smoothing",
    "reliability_no_ema": "No smoothing",
    "reliability_val_loss_only": "Val-loss only",
    "reliability_monitor_only": "Monitor-only",
}

ABLATION_METHODS = [
    "reliability",
    "reliability_full",
    "reliability_pb_only",
    "reliability_pd_only",
    "reliability_no_smoothing",
    "reliability_no_ema",
    "reliability_val_loss_only",
    "reliability_monitor_only",
]


def collect_summaries(results_dir):
    results_dir = Path(results_dir)
    master = results_dir / "master_summary.csv"
    if master.exists():
        try:
            return pd.read_csv(master)
        except Exception:
            pass
    rows = []
    for path in results_dir.glob("*/*/seed_*/summary.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row.setdefault("experiment", path.parts[-4])
            row.setdefault("method", path.parts[-3])
            row.setdefault("seed", int(path.parts[-2].replace("seed_", "")))
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def parse_sensitivity_experiment(name):
    match = re.search(r"alpha(\d{3})_beta(\d{3})", str(name))
    if not match:
        return np.nan, np.nan
    return int(match.group(1)) / 100.0, int(match.group(2)) / 100.0


def add_alpha_beta(summary):
    if summary.empty:
        return summary
    out = summary.copy()
    if "controller_alpha" not in out.columns:
        out["controller_alpha"] = np.nan
    if "controller_beta" not in out.columns:
        out["controller_beta"] = np.nan
    for idx, row in out.iterrows():
        if pd.isna(row.get("controller_alpha")) or pd.isna(row.get("controller_beta")):
            alpha, beta = parse_sensitivity_experiment(row.get("experiment", ""))
            if np.isfinite(alpha):
                out.at[idx, "controller_alpha"] = alpha
            if np.isfinite(beta):
                out.at[idx, "controller_beta"] = beta
    return out


def unavailable(ax, message="data unavailable"):
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=PALETTE["midgray"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def method_label(method):
    return METHOD_LABELS.get(str(method), str(method))


def method_color(method):
    return PALETTE.get(method, PALETTE["gray"])


def dot_plot(ax, df, metric, title, ylabel):
    if df.empty or metric not in df.columns or "method" not in df.columns:
        unavailable(ax, f"{metric} unavailable")
        ax.set_title(title, pad=8)
        return

    methods = [m for m in ABLATION_METHODS if m in set(df["method"])]
    if not methods:
        unavailable(ax, "ablation results unavailable")
        ax.set_title(title, pad=8)
        return

    for i, method in enumerate(methods):
        values = pd.to_numeric(df[df["method"] == method][metric], errors="coerce").dropna().to_numpy(float)
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        color = method_color(method)
        size = 34 if method == "reliability" else 24
        ax.scatter(np.full(len(values), i) + jitter, values, s=size, color=color, alpha=0.82, zorder=3)
        mean, lo, hi = mean_ci(values)
        ax.errorbar(
            i, mean, yerr=[[mean - lo], [hi - mean]], fmt="_", markersize=16,
            color=PALETTE["axis"], capsize=2.5, linewidth=1.2, zorder=4,
        )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_label(m) for m in methods], rotation=28, ha="right")
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        ax.set_ylim(ymin, ymax + 0.08 * (ymax - ymin))


def lr_epoch_panel(ax, df):
    if df.empty or "method" not in df.columns:
        unavailable(ax, "ablation results unavailable")
        ax.set_title("Controller effort remains bounded", pad=8)
        return
    methods = [m for m in ABLATION_METHODS if m in set(df["method"])]
    if not methods:
        unavailable(ax, "ablation results unavailable")
        ax.set_title("Controller effort remains bounded", pad=8)
        return

    ax2 = ax.twinx()
    for axis, metric, marker, label, offset in [
        (ax, "lr_reductions", "o", "LR reductions", -0.08),
        (ax2, "epochs_run", "^", "Epochs run", 0.08),
    ]:
        if metric not in df.columns:
            continue
        for i, method in enumerate(methods):
            values = pd.to_numeric(df[df["method"] == method][metric], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            jitter = np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else np.array([0.0])
            color = method_color(method)
            axis.scatter(np.full(len(values), i + offset) + jitter, values, s=22, color=color, marker=marker, alpha=0.78)
            mean, lo, hi = mean_ci(values)
            axis.errorbar(i + offset, mean, yerr=[[mean - lo], [hi - mean]], fmt="_", color=PALETTE["axis"], capsize=2.2)
        axis.scatter([], [], marker=marker, color=PALETTE["axis"], label=label)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_label(m) for m in methods], rotation=28, ha="right")
    ax.set_ylabel("LR reductions")
    ax2.set_ylabel("Epochs run")
    ax.set_title("Controller effort remains bounded", pad=8)
    handles, labels = [], []
    for axis in [ax, ax2]:
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=6.5)


def figure_ablation(summary, out_dir):
    exp_type = summary.get("experiment_type", pd.Series("", index=summary.index)).astype(str) if not summary.empty else pd.Series(dtype=str)
    df = summary[exp_type == "ablation"].copy() if not summary.empty else pd.DataFrame()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), constrained_layout=True)
    axes = axes.ravel()

    dot_plot(axes[0], df, "test_acc", "Full controller preserves accuracy", "Test accuracy")
    dot_plot(axes[1], df, "test_ece", "Calibration requires both signals", "ECE")
    dot_plot(axes[2], df, "test_brier", "Probabilistic error is reduced", "Brier score")
    lr_epoch_panel(axes[3], df)

    for ax, label in zip(axes, "abcd"):
        add_panel_label(ax, label)
    fig.suptitle("Figure 4.11 | Ablation study of reliability-aware control", y=1.02, fontsize=10)
    save_figure(fig, Path(out_dir) / "fig_4_11_ablation_study")


def heatmap(ax, df, metric, title, cmap, fmt="{:.3f}"):
    if df.empty or metric not in df.columns:
        unavailable(ax, f"{metric} unavailable")
        ax.set_title(title, pad=8)
        return
    alphas = sorted(pd.to_numeric(df["controller_alpha"], errors="coerce").dropna().unique())
    betas = sorted(pd.to_numeric(df["controller_beta"], errors="coerce").dropna().unique())
    if not alphas or not betas:
        unavailable(ax, "sensitivity results unavailable")
        ax.set_title(title, pad=8)
        return

    matrix = np.full((len(betas), len(alphas)), np.nan)
    for yi, beta in enumerate(betas):
        for xi, alpha in enumerate(alphas):
            vals = pd.to_numeric(
                df[(df["controller_alpha"] == alpha) & (df["controller_beta"] == beta)][metric],
                errors="coerce",
            ).dropna()
            if len(vals):
                matrix[yi, xi] = float(vals.mean())

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f"{b:.2f}" for b in betas])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title, pad=8)
    for yi in range(len(betas)):
        for xi in range(len(alphas)):
            if np.isfinite(matrix[yi, xi]):
                ax.text(xi, yi, fmt.format(matrix[yi, xi]), ha="center", va="center", fontsize=6.5)
    if 0.50 in alphas and 0.90 in betas:
        xi = alphas.index(0.50)
        yi = betas.index(0.90)
        ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor=PALETTE["accent"], linewidth=1.8))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.045)


def figure_sensitivity(summary, out_dir):
    df = add_alpha_beta(summary)
    exp_type = df.get("experiment_type", pd.Series("", index=df.index)).astype(str) if not df.empty else pd.Series(dtype=str)
    df = df[exp_type == "sensitivity"].copy() if not df.empty else pd.DataFrame()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.5), constrained_layout=True)
    axes = axes.ravel()

    heatmap(axes[0], df, "test_acc", "Accuracy is stable near the default", "viridis")
    heatmap(axes[1], df, "test_ece", "Calibration remains low", "magma_r")
    heatmap(axes[2], df, "test_brier", "Brier score sensitivity", "magma_r")
    heatmap(axes[3], df, "final_gen_gap_loss", "Generalization gap sensitivity", "magma_r")

    for ax, label in zip(axes, "abcd"):
        add_panel_label(ax, label)
    fig.suptitle("Figure 4.12 | Hyperparameter sensitivity analysis", y=1.02, fontsize=10)
    save_figure(fig, Path(out_dir) / "fig_4_12_sensitivity_analysis")


def figure_statistical_tests(summary, out_dir):
    if summary.empty:
        return

    exp_type = summary.get("experiment_type", pd.Series("", index=summary.index)).astype(str)
    df = summary[~exp_type.isin(["quick", "ablation", "sensitivity"])].copy()
    metrics = [
        ("test_acc", "Test accuracy", "a"),
        ("test_nll", "Test NLL", "b"),
        ("test_ece", "ECE", "c"),
        ("test_brier", "Brier score", "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.9), constrained_layout=True)
    axes = axes.ravel()

    for ax, (metric, title, label) in zip(axes, metrics):
        comp = compare_ours_vs_best_baseline(df, metric, ["dataset", "model", "experiment"])
        if comp.empty:
            unavailable(ax, f"{metric} unavailable")
            ax.set_title(title, pad=8)
            add_panel_label(ax, label)
            continue

        comp = comp.sort_values(["dataset", "model", "experiment"])
        comp = comp[pd.to_numeric(comp["mean_difference"], errors="coerce").notna()].copy()
        if comp.empty:
            unavailable(ax, f"{metric} unavailable")
            ax.set_title(title, pad=8)
            add_panel_label(ax, label)
            continue
        labels = [
            f"{row.dataset}\n{row.model}\nvs {method_label(row.best_baseline_name)}"
            for row in comp.itertuples(index=False)
        ]
        y = pd.to_numeric(comp["mean_difference"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(comp["ci95_low"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(comp["ci95_high"], errors="coerce").to_numpy(float)
        x = np.arange(len(comp))
        yerr = np.vstack([y - lo, hi - y])
        yerr = np.where(np.isfinite(yerr), np.maximum(0.0, yerr), 0.0)

        ax.bar(x, y, yerr=yerr, capsize=2.5, color=PALETTE["reliability"], alpha=0.84)
        ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Paired difference")
        ax.set_title(title, pad=8)
        max_abs = np.nanmax(np.abs(y)) if len(y) else 1.0
        offset = 0.03 * max(max_abs, 1e-8)
        for idx, row in enumerate(comp.itertuples(index=False)):
            marker = str(row.significance)
            if marker in {"*", "**", "***"} and np.isfinite(y[idx]):
                ax.text(idx, y[idx] + offset, marker, ha="center", va="bottom", fontsize=8)
        add_panel_label(ax, label)

    fig.suptitle("Figure 4.13 | Paired statistical tests against best baselines", y=1.02, fontsize=10)
    save_figure(fig, Path(out_dir) / "fig_4_13_statistical_tests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="statistical_figures")
    parser.add_argument("--include-quick", action="store_true", help="Include quick-test experiments in outputs.")
    args = parser.parse_args()

    set_nature_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = filter_quick(collect_summaries(args.results_dir), include_quick=args.include_quick)
    figure_ablation(summary, out_dir)
    figure_sensitivity(summary, out_dir)
    figure_statistical_tests(summary, out_dir)

    print(f"Saved statistical figures to {out_dir}")


if __name__ == "__main__":
    main()
