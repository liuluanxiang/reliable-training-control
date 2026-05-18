import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import set_nature_style, save_figure, panel_label, PALETTE, LABELS
from src.metrics import spearman_np
from src.results_io import collect_histories as load_histories, collect_summaries as load_summaries


CASE1_METHODS = ["step", "cosine", "plateau", "reliability"]
CASE1_COLORS = {
    "step": "#1F77B4",
    "cosine": "#FF7F0E",
    "plateau": "#2CA02C",
    "reliability": "#C0005B",
}
CASE1_LABELS = {
    "step": "Step decay",
    "cosine": "Cosine annealing",
    "plateau": "ReduceLROnPlateau",
    "reliability": "Ours",
}
EXPERIMENT_COL = "experiment"
DATASET_COL = "dataset"
MODEL_COL = "model"
METHOD_COL = "method"
SEED_COL = "seed"
EPOCH_COL = "epoch"
PB_COL = "pb_signal"
GAP_COL = "gen_gap_loss"
NLL_COL = "val_nll"
CASE1_SETTING_ORDER = [
    ("cifar10", "resnet18"),
    ("cifar10", "resnet50"),
    ("cifar10", "vgg16"),
    ("cifar100", "resnet18"),
    ("cifar100", "resnet50"),
    ("cifar100", "vgg16"),
]


def collect_histories(results_dir):
    frames = []
    for p in Path(results_dir).glob("*/*/seed_*/history.csv"):
        df = pd.read_csv(p)
        if "experiment" not in df.columns:
            df["experiment"] = p.parts[-4]
        if "method" not in df.columns:
            df["method"] = p.parts[-3]
        if "seed" not in df.columns:
            df["seed"] = int(p.parts[-2].replace("seed_", ""))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No history.csv found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def collect_summaries(results_dir):
    master = Path(results_dir) / "master_summary.csv"
    if master.exists():
        return pd.read_csv(master)
    rows = []
    for p in Path(results_dir).glob("*/*/seed_*/summary.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        row.setdefault("experiment", p.parts[-4])
        row.setdefault("method", p.parts[-3])
        row.setdefault("seed", int(p.parts[-2].replace("seed_", "")))
        rows.append(row)
    return pd.DataFrame(rows)


def mean_ci(df, metric):
    rows = []
    for (method, epoch), d in df.groupby(["method", "epoch"]):
        vals = d[metric].dropna().astype(float).values
        if len(vals) == 0:
            continue
        mu = vals.mean()
        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append([method, epoch, mu, mu - ci, mu + ci])
    return pd.DataFrame(rows, columns=["method", "epoch", "mean", "lo", "hi"])


def representative_experiment(df):
    if df.empty or "experiment" not in df.columns:
        return None
    meta_cols = [c for c in ["experiment", "model"] if c in df.columns]
    if meta_cols:
        meta = df[meta_cols].drop_duplicates()
        if {"experiment", "model"}.issubset(meta.columns):
            resnet18 = meta[meta["model"].astype(str).str.lower() == "resnet18"]
            if len(resnet18):
                return resnet18["experiment"].iloc[0]
    values = df["experiment"].dropna().unique()
    return values[0] if len(values) else None


def short_case1_setting_label(dataset, model):
    labels = {
        ("cifar10", "resnet18"): "C10-R18",
        ("cifar10", "resnet50"): "C10-R50",
        ("cifar10", "vgg16"): "C10-V16",
        ("cifar100", "resnet18"): "C100-R18",
        ("cifar100", "resnet50"): "C100-R50",
        ("cifar100", "vgg16"): "C100-V16",
    }
    return labels.get((str(dataset).lower(), str(model).lower()), f"{dataset}-{model}")


def representative_case1_experiment(df):
    required_cols = {EXPERIMENT_COL, METHOD_COL, PB_COL, GAP_COL}
    if df.empty or not required_cols.issubset(df.columns):
        return representative_experiment(df)
    for dataset, model in [("cifar100", "resnet18"), ("cifar10", "resnet18"), ("cifar100", "resnet50"), ("cifar10", "resnet50")]:
        subset = df[
            (df[DATASET_COL].astype(str).str.lower() == dataset)
            & (df[MODEL_COL].astype(str).str.lower() == model)
        ] if {DATASET_COL, MODEL_COL}.issubset(df.columns) else pd.DataFrame()
        if subset.empty:
            continue
        experiments = sorted(subset[EXPERIMENT_COL].dropna().astype(str).unique())
        for exp in experiments:
            exp_df = subset[subset[EXPERIMENT_COL].astype(str) == exp]
            available = set(exp_df[METHOD_COL].astype(str))
            if set(CASE1_METHODS).issubset(available):
                # Prefer CIFAR-100 + ResNet-18 for Figures 4.6 and 4.7 when available.
                return exp
    return representative_experiment(df)


def save_case1_figure(fig, out_dir, filename):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / filename
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _numeric_pair(df, x_col, y_col):
    d = df[[x_col, y_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    return d.dropna()


def _format_p_value(p_value):
    if not np.isfinite(p_value):
        return "p = n/a"
    if p_value < 1e-3:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def plot_mean_std_trajectory(ax, histories, metric_col, ylabel, title, methods):
    for method in methods:
        d = histories[histories[METHOD_COL].astype(str) == method].copy()
        if d.empty or metric_col not in d.columns:
            continue
        d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
        rows = []
        for epoch, group in d.groupby(EPOCH_COL):
            values = group[metric_col].dropna().to_numpy(float)
            if len(values):
                rows.append((float(epoch), float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0))
        if not rows:
            continue
        # Use only observed epochs; early-stopped runs are not extended to the full schedule.
        curve = pd.DataFrame(rows, columns=[EPOCH_COL, "mean", "std"]).sort_values(EPOCH_COL)
        color = CASE1_COLORS[method]
        ax.plot(curve[EPOCH_COL], curve["mean"], color=color, lw=1.8, label=CASE1_LABELS[method])
        ax.fill_between(
            curve[EPOCH_COL].to_numpy(float),
            (curve["mean"] - curve["std"]).to_numpy(float),
            (curve["mean"] + curve["std"]).to_numpy(float),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


def plot_scatter_with_binned_median(ax, data, x_col, y_col, xlabel, ylabel, title):
    d = _numeric_pair(data, x_col, y_col) if {x_col, y_col}.issubset(data.columns) else pd.DataFrame()
    if d.empty:
        ax.text(0.5, 0.5, "data unavailable", ha="center", va="center", transform=ax.transAxes)
        return
    rho, p_value = (np.nan, np.nan)
    if len(d) >= 3:
        rho, p_value = spearmanr(d[x_col], d[y_col])
    ax.scatter(d[x_col], d[y_col], s=16, alpha=0.34, color=CASE1_COLORS["reliability"], edgecolor="none")
    if len(d) >= 12 and d[x_col].nunique() > 1:
        bins = min(10, max(8, int(np.sqrt(len(d)))))
        d = d.sort_values(x_col).copy()
        d["_bin"] = pd.qcut(d[x_col], q=bins, duplicates="drop")
        trend = d.groupby("_bin", observed=True).agg({x_col: "median", y_col: "median"}).dropna()
        trend_direction = float(trend[y_col].iloc[-1] - trend[y_col].iloc[0]) if len(trend) >= 2 else np.nan
        compatible = np.isfinite(rho) and abs(float(rho)) >= 0.15 and np.sign(trend_direction) == np.sign(float(rho))
        # Avoid drawing a trend line when outliers make the binned median direction contradict Spearman rho.
        if len(trend) >= 2 and compatible:
            ax.plot(trend[x_col], trend[y_col], color="#222222", lw=1.5, marker="o", markersize=3.2)
    if len(d) >= 3:
        ax.text(
            0.04,
            0.96,
            f"Spearman rho = {rho:.2f}\n{_format_p_value(float(p_value))}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


def case1_correlation_rows(histories):
    rows = []
    required = {DATASET_COL, MODEL_COL, METHOD_COL, SEED_COL, PB_COL, GAP_COL}
    if histories.empty or not required.issubset(histories.columns):
        return pd.DataFrame(rows)
    for keys, group in histories.groupby([DATASET_COL, MODEL_COL, METHOD_COL, SEED_COL], dropna=False):
        if str(keys[2]) not in CASE1_METHODS:
            continue
        d = _numeric_pair(group, PB_COL, GAP_COL)
        rows.append(
            {
                DATASET_COL: str(keys[0]).lower(),
                MODEL_COL: str(keys[1]).lower(),
                METHOD_COL: str(keys[2]),
                "rho": spearman_np(d[PB_COL], d[GAP_COL]) if len(d) >= 3 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_correlation_summary(ax, corr_df):
    labels = [short_case1_setting_label(dataset, model) for dataset, model in CASE1_SETTING_ORDER]
    x = np.arange(len(labels))
    offsets = np.linspace(-0.24, 0.24, len(CASE1_METHODS))
    missing_by_setting = {}
    for offset, method in zip(offsets, CASE1_METHODS):
        means, stds = [], []
        for dataset, model in CASE1_SETTING_ORDER:
            vals = corr_df.loc[
                (corr_df[DATASET_COL] == dataset)
                & (corr_df[MODEL_COL] == model)
                & (corr_df[METHOD_COL] == method),
                "rho",
            ].dropna().to_numpy(float) if not corr_df.empty else np.array([])
            if len(vals) == 0:
                setting = short_case1_setting_label(dataset, model)
                missing_by_setting.setdefault(setting, []).append(CASE1_LABELS[method])
                print(f"Warning: missing {CASE1_LABELS[method]} data for {setting}; no correlation point plotted.")
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt="o",
            markersize=5,
            capsize=2,
            elinewidth=0.8,
            linewidth=0,
            color=CASE1_COLORS[method],
            label=CASE1_LABELS[method],
            zorder=3,
        )
    for i, label in enumerate(labels):
        missing = missing_by_setting.get(label)
        if missing:
            compact = [name.replace(" decay", "").replace(" annealing", "").replace("ReduceLROnPlateau", "Plateau") for name in missing]
            ax.text(i, -0.99, "missing: " + ", ".join(compact), ha="center", va="bottom", fontsize=6.5, color="#666666", rotation=90)
    ax.axhline(0, color="#B8B8B8", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Dataset-architecture setting")
    ax.set_ylabel(r"Correlation rho($PB_t$, $G_t$)")
    ax.set_title(r"Cross-method comparison of rho($PB_t$, $G_t$)", fontsize=10)
    ax.legend(frameon=False)


def fig_case1(hist, out_dir):
    exp = representative_case1_experiment(hist)
    df = hist[hist[EXPERIMENT_COL] == exp].copy() if exp else pd.DataFrame()
    rel = df[df[METHOD_COL] == "reliability"].copy() if not df.empty else pd.DataFrame()

    # Figure 4.6 includes all methods because PB_t is computable for every control strategy.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    plot_mean_std_trajectory(axes[0], df, PB_COL, r"$PB_t$", r"(a) PAC-Bayes-inspired signal", CASE1_METHODS)
    plot_mean_std_trajectory(axes[1], df, GAP_COL, "Generalization gap", "(b) Empirical generalization gap", CASE1_METHODS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.subplots_adjust(top=0.86, hspace=0.42)
    save_case1_figure(fig, out_dir, "figure_4_6_pb_gap_trajectory_revised.png")

    # Figure 4.7 uses the proposed method only to keep the illustrative scatter plots readable.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    plot_scatter_with_binned_median(axes[0], rel, PB_COL, GAP_COL, r"$PB_t$", "Generalization gap", r"(a) $PB_t$ and generalization gap")
    plot_scatter_with_binned_median(axes[1], rel, PB_COL, NLL_COL, r"$PB_t$", "Validation NLL", r"(b) $PB_t$ and validation NLL")
    save_case1_figure(fig, out_dir, "figure_4_7_pb_association_revised.png")

    corr = case1_correlation_rows(hist)
    # Incomplete settings are kept on the x-axis and annotated so missing methods are explicit.
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 3.9), constrained_layout=True)
    plot_grouped_correlation_summary(ax, corr)
    save_case1_figure(fig, out_dir, "figure_4_8_pb_correlation_summary_revised.png")


def fig_case2(hist, out_dir):
    exp = representative_experiment(hist)
    df = hist[(hist["method"] == "reliability") & (hist["experiment"] == exp)].copy() if exp else pd.DataFrame()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8))
    line_metrics = [
        ("pd_signal", r"Primal-dual signal $PD_t$"),
        ("grad_norm", "Gradient norm"),
        ("update_norm", "Update norm"),
        ("risk_violation_vt", "Risk violation"),
    ]
    for ax, (m, title), letter in zip(axes.ravel()[:4], line_metrics, list("abcd")):
        for seed, d in df.groupby("seed"):
            ax.plot(d["epoch"], d[m], lw=1.0, alpha=0.85)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        panel_label(ax, letter)

    axes[2, 0].scatter(df["pd_signal"], df["grad_norm"], s=8, alpha=0.4, color=PALETTE["reliability"])
    axes[2, 0].set_xlabel(r"$PD_t$"); axes[2, 0].set_ylabel("Gradient norm")
    axes[2, 0].set_title(r"$PD_t$ vs gradient norm")
    panel_label(axes[2, 0], "e")

    exps, counts = [], []
    for exp, d in hist[hist["method"] == "reliability"].groupby("experiment"):
        seed_counts = []
        for seed, s in d.groupby("seed"):
            th = s["pd_signal"].mean() + 2 * s["pd_signal"].std(ddof=0)
            seed_counts.append((s["pd_signal"] > th).sum())
        exps.append(exp); counts.append(np.mean(seed_counts))
    axes[2, 1].bar(range(len(exps)), counts, color=PALETTE["gray"])
    axes[2, 1].set_xticks(range(len(exps))); axes[2, 1].set_xticklabels(exps, rotation=35, ha="right")
    axes[2, 1].set_ylabel("Spike count")
    axes[2, 1].set_title("Instability spikes")
    panel_label(axes[2, 1], "f")

    save_figure(fig, Path(out_dir) / "fig_4_7_case2_primal_dual_stability")


def fig_case3(hist, summary, out_dir):
    exp = representative_experiment(hist)
    df = hist[hist["experiment"] == exp].copy() if exp else hist.copy()
    sdf = summary.copy()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8))

    for method, d in df.groupby("method"):
        m = d.groupby("epoch")["lr"].mean().reset_index()
        axes[0, 0].plot(m["epoch"], m["lr"], lw=1.4, color=PALETTE.get(method, "#333333"), label=LABELS.get(method, method))
    axes[0, 0].set_yscale("log"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Learning rate")
    axes[0, 0].set_title("Learning-rate trajectory")
    axes[0, 0].legend(frameon=False)
    panel_label(axes[0, 0], "a")

    for ax, metric, title, letter in [
        (axes[0, 1], "val_acc", "Validation accuracy", "b"),
        (axes[1, 0], "val_nll", "Validation NLL", "c"),
        (axes[1, 1], "reliability_rbar", r"Smoothed reliability $\bar{R}_t$", "d"),
    ]:
        mdf = mean_ci(df, metric)
        for method, d in mdf.groupby("method"):
            color = PALETTE.get(method, "#333333")
            ax.plot(d["epoch"], d["mean"], lw=1.4, color=color, label=LABELS.get(method, method))
            ax.fill_between(d["epoch"].values, d["lo"].values, d["hi"].values, color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.set_title(title)
        panel_label(ax, letter)

    methods = list(sdf["method"].unique())
    for ax, metric, title, letter in [
        (axes[2, 0], "test_acc", "Test accuracy", "e"),
        (axes[2, 1], "lr_reductions", "LR reduction count", "f"),
    ]:
        vals = [sdf[sdf["method"] == m][metric].mean() for m in methods]
        ax.bar(range(len(methods)), vals, color=[PALETTE.get(m, "#333333") for m in methods])
        ax.set_xticks(range(len(methods))); ax.set_xticklabels([LABELS.get(m, m) for m in methods], rotation=35, ha="right")
        ax.set_ylabel(title); ax.set_title(title)
        panel_label(ax, letter)

    save_figure(fig, Path(out_dir) / "fig_4_8_case3_lr_scheduling")


def fig_case4(summary, results_dir, out_dir):
    exp = representative_experiment(summary)
    sdf = summary.copy()
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 10.0))
    methods = list(sdf["method"].unique())

    metrics = [
        ("test_acc", "Test accuracy", "a"),
        ("test_nll", "Test NLL", "b"),
        ("test_ece", "ECE", "c"),
        ("test_brier", "Brier score", "d"),
        ("test_mce", "MCE", "e"),
        ("test_aurc", "AURC", "f"),
    ]

    for ax, (metric, title, letter) in zip(axes.ravel()[:6], metrics):
        vals = [sdf[sdf["method"] == m][metric].dropna().values for m in methods]
        means = [v.mean() if len(v) else np.nan for v in vals]
        errs = [v.std(ddof=1) if len(v) > 1 else 0.0 for v in vals]
        ax.bar(range(len(methods)), means, yerr=errs, capsize=2, color=[PALETTE.get(m, "#333333") for m in methods])
        ax.set_xticks(range(len(methods))); ax.set_xticklabels([LABELS.get(m, m) for m in methods], rotation=35, ha="right")
        ax.set_ylabel(title); ax.set_title(title)
        panel_label(ax, letter)

    ax = axes[3, 0]
    for method in ["cosine", "plateau", "reliability"]:
        p = Path(results_dir) / exp / method / "seed_1" / "calibration_bins.csv"
        if p.exists():
            d = pd.read_csv(p)
            ax.plot(d["confidence"], d["accuracy"], marker="o", ms=3, lw=1.2, color=PALETTE.get(method, "#333333"), label=LABELS.get(method, method))
    ax.plot([0, 1], [0, 1], "--", lw=1.0, color="#999999")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(frameon=False)
    panel_label(ax, "g")

    ax = axes[3, 1]
    for method in ["cosine", "plateau", "reliability"]:
        p = Path(results_dir) / exp / method / "seed_1" / "risk_coverage_curve.csv"
        if p.exists():
            d = pd.read_csv(p)
            ax.plot(d["coverage"], d["risk"], lw=1.2, color=PALETTE.get(method, "#333333"), label=LABELS.get(method, method))
    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk")
    ax.set_title("Risk-coverage curve")
    ax.legend(frameon=False)
    panel_label(ax, "h")

    save_figure(fig, Path(out_dir) / "fig_4_9_case4_calibration_generalization")


def fig_case5(summary, out_dir):
    sdf = summary.copy()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8))

    metrics = [
        ("test_acc", "Test accuracy"),
        ("test_nll", "Test NLL"),
        ("test_ece", "ECE"),
        ("best_epoch", "Best epoch"),
        ("epochs_run", "Epochs run"),
        ("late_degradation", "Late degradation"),
    ]
    methods = list(sdf["method"].unique())

    for ax, (metric, title), letter in zip(axes.ravel(), metrics, list("abcdef")):
        data = [sdf[sdf["method"] == m][metric].dropna().values for m in methods]
        bp = ax.boxplot(data, tick_labels=[LABELS.get(m, m) for m in methods], patch_artist=True, showfliers=False)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(PALETTE.get(method, "#777777"))
            patch.set_alpha(0.75)
        for i, vals in enumerate(data, start=1):
            if len(vals):
                jitter = np.linspace(-0.08, 0.08, len(vals))
                ax.scatter(i + jitter, vals, s=12, color="black", alpha=0.65, zorder=3)
        ax.set_title(title); ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=35)
        panel_label(ax, letter)

    save_figure(fig, Path(out_dir) / "fig_4_10_case5_seed_stability")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--include-quick", action="store_true", help="Include quick-test experiments in outputs.")
    args = parser.parse_args()

    set_nature_style()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hist = load_histories(args.results_dir, include_quick=args.include_quick)
    summary = load_summaries(args.results_dir, include_quick=args.include_quick)
    if hist.empty:
        print(f"No history.csv files found under {args.results_dir}; no case-study figures generated.")
        return

    fig_case1(hist, out)
    fig_case2(hist, out)
    fig_case3(hist, summary, out)
    fig_case4(summary, args.results_dir, out)
    fig_case5(summary, out)

    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()
