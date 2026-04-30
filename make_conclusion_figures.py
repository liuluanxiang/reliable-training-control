import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import reliability_bins, risk_coverage_curve  # noqa: E402
from src.plot_style import (  # noqa: E402
    METHOD_LABELS, METHOD_ORDER, PALETTE, add_panel_label, add_stat_annotation,
    draw_schematic_sequence, place_legend_safely, save_figure, set_nature_style,
)
from src.results_io import filter_quick  # noqa: E402
from src.stats_utils import format_p_value, mean_ci, paired_ttest, spearman_with_p  # noqa: E402


HIGHER_BETTER = {"test_acc", "test_top5_acc", "aulc_val_acc"}
LOWER_BETTER = {
    "test_nll", "test_ece", "test_brier", "test_mce", "test_aurc", "test_eaurc",
    "final_gen_gap_loss", "late_degradation",
}


def collect_histories(results_dir):
    frames = []
    for path in Path(results_dir).glob("*/*/seed_*/history.csv"):
        try:
            df = pd.read_csv(path)
            parts = path.parts
            if "experiment" not in df.columns:
                df["experiment"] = parts[-4]
            if "method" not in df.columns:
                df["method"] = parts[-3]
            if "seed" not in df.columns:
                df["seed"] = int(parts[-2].replace("seed_", ""))
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
            parts = path.parts
            row.setdefault("experiment", parts[-4])
            row.setdefault("method", parts[-3])
            row.setdefault("seed", int(parts[-2].replace("seed_", "")))
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def ordered_methods(methods):
    known = [m for m in METHOD_ORDER if m in set(methods)]
    extra = sorted([m for m in methods if m not in METHOD_ORDER])
    return known + extra


def method_label(method):
    return METHOD_LABELS.get(method, str(method))


def color_for(method):
    return PALETTE.get(method, PALETTE["gray"])


def unavailable(ax, text="metric unavailable"):
    ax.text(0.5, 0.5, text, ha="center", va="center", color=PALETTE["midgray"], fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def choose_experiment(df):
    if df is None or df.empty or "experiment" not in df.columns:
        return None
    values = [v for v in df["experiment"].dropna().unique()]
    return values[0] if values else None


def subset_first_experiment(df):
    exp = choose_experiment(df)
    if exp is None:
        return df
    return df[df["experiment"] == exp].copy()


def add_schematic(ax, title, nodes, arrows=None):
    draw_schematic_sequence(ax, title, nodes)
    if arrows:
        for text, x, yy in arrows:
            ax.text(x, yy, text, transform=ax.transAxes, ha="center", va="center", fontsize=7, color=PALETTE["gray"])


def trajectory_ci(ax, df, metric, methods=None, title="", ylabel="", reliability_only=False):
    if df.empty or metric not in df.columns or "epoch" not in df.columns:
        unavailable(ax)
        ax.set_title(title)
        return
    data = df.copy()
    if reliability_only:
        data = data[data["method"] == "reliability"]
    if methods is not None:
        data = data[data["method"].isin(methods)]
    if data.empty:
        unavailable(ax)
        ax.set_title(title)
        return
    for method in ordered_methods(data["method"].dropna().unique()):
        d = data[data["method"] == method]
        rows = []
        for epoch, g in d.groupby("epoch"):
            mu, lo, hi = mean_ci(g[metric])
            rows.append((epoch, mu, lo, hi))
        if not rows:
            continue
        arr = pd.DataFrame(rows, columns=["epoch", "mean", "lo", "hi"]).sort_values("epoch")
        x = arr["epoch"].astype(float).to_numpy()
        mean = arr["mean"].astype(float).to_numpy()
        lo = arr["lo"].astype(float).to_numpy()
        hi = arr["hi"].astype(float).to_numpy()
        lw = 2.0 if method == "reliability" else 1.2
        alpha = 0.22 if method == "reliability" else 0.12
        ax.plot(x, mean, color=color_for(method), lw=lw, label=method_label(method), zorder=3 if method == "reliability" else 2)
        ax.fill_between(x, lo, hi, color=color_for(method), alpha=alpha, linewidth=0)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or metric)
    place_legend_safely(ax, preferred="best", outside_if_needed=not reliability_only, ncol=1)


def scatter_with_regression(ax, df, x_col, y_col, title, xlabel, ylabel):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        unavailable(ax)
        ax.set_title(title)
        return
    d = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        unavailable(ax, "insufficient data")
        ax.set_title(title)
        return
    x = d[x_col].astype(float).to_numpy()
    y = d[y_col].astype(float).to_numpy()
    ax.scatter(x, y, s=13, alpha=0.35, color=PALETTE["reliability"], edgecolor="none")
    try:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, slope * xs + intercept, color=PALETTE["axis"], lw=1.2)
    except Exception:
        pass
    rho, p = spearman_with_p(x, y)
    annotation_loc = "top-left" if np.isfinite(rho) and rho >= 0 else "top-right"
    add_stat_annotation(ax, f"Spearman rho = {rho:.2f}\n{format_p_value(p)}", loc=annotation_loc)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def dot_plot(ax, df, metric, title, ylabel="", methods=None, ci=True, sd=False, higher=True, significance=False):
    if df.empty or metric not in df.columns or "method" not in df.columns:
        unavailable(ax)
        ax.set_title(title)
        return
    data = df[["method", "seed", metric] if "seed" in df.columns else ["method", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])
    if methods is not None:
        data = data[data["method"].isin(methods)]
    methods = ordered_methods(data["method"].dropna().unique())
    if not methods:
        unavailable(ax)
        ax.set_title(title)
        return
    rng = np.random.default_rng(4)
    for i, method in enumerate(methods):
        vals = data.loc[data["method"] == method, metric].astype(float).to_numpy()
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=20, color=color_for(method), alpha=0.72,
                   edgecolor="white", linewidth=0.35, zorder=3)
        mean = float(np.mean(vals))
        if sd and len(vals) > 1:
            lo, hi = mean - float(np.std(vals, ddof=1)), mean + float(np.std(vals, ddof=1))
        elif ci:
            mean, lo, hi = mean_ci(vals)
        else:
            lo = hi = mean
        ax.errorbar([i], [mean], yerr=[[mean - lo], [hi - mean]], fmt="_", ms=16, mew=1.6,
                    color=PALETTE["axis"], capsize=2.5, zorder=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_label(m) for m in methods], rotation=30, ha="right")
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel or metric)
    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = 0.16 * (ymax - ymin) if significance else 0.08 * (ymax - ymin)
        ax.set_ylim(ymin, ymax + pad)
    if significance and "reliability" in methods:
        p = ours_vs_best_baseline_p(df, metric, higher=higher)
        if np.isfinite(p):
            add_stat_annotation(ax, f"Ours vs best baseline\n{format_p_value(p)}", loc="top-right")


def ours_vs_best_baseline_p(summary, metric, higher=True):
    if summary.empty or "seed" not in summary.columns or metric not in summary.columns:
        return np.nan
    data = summary[["method", "seed", metric]].dropna()
    if "reliability" not in set(data["method"]):
        return np.nan
    baselines = [m for m in data["method"].unique() if m != "reliability"]
    if not baselines:
        return np.nan
    means = data[data["method"].isin(baselines)].groupby("method")[metric].mean()
    if means.empty:
        return np.nan
    best = means.idxmax() if higher else means.idxmin()
    ours = data[data["method"] == "reliability"].set_index("seed")[metric]
    base = data[data["method"] == best].set_index("seed")[metric]
    seeds = ours.index.intersection(base.index)
    if len(seeds) < 2:
        return np.nan
    return paired_ttest(ours.loc[seeds].to_numpy(), base.loc[seeds].to_numpy())


def pb_seed_correlations(hist):
    rows = []
    required = {"experiment", "dataset", "model", "seed", "method", "pb_signal", "gen_gap_loss", "val_nll"}
    if hist.empty or not required.issubset(hist.columns):
        return pd.DataFrame(rows)
    df = hist[hist["method"] == "reliability"].copy()
    for keys, d in df.groupby(["experiment", "dataset", "model", "seed"]):
        rho_g, p_g = spearman_with_p(d["pb_signal"], d["gen_gap_loss"])
        rho_nll, p_nll = spearman_with_p(d["pb_signal"], d["val_nll"])
        rows.append({
            "experiment": keys[0], "dataset": keys[1], "model": keys[2], "seed": keys[3],
            "rho_pb_gengap": rho_g, "p_pb_gengap": p_g,
            "rho_pb_valnll": rho_nll, "p_pb_valnll": p_nll,
        })
    return pd.DataFrame(rows)


def figure_46(hist, out_dir):
    df = subset_first_experiment(hist)
    rel = df[df.get("method", "") == "reliability"] if not df.empty and "method" in df.columns else pd.DataFrame()
    fig, axes = plt.subplots(3, 2, figsize=(7.8, 8.2), constrained_layout=True)
    axes = axes.ravel()
    add_schematic(axes[0], "PB_t links model movement to reliability", [r"$\theta_0$", r"$\theta_t$", "Val NLL", r"$PB_t$"])
    trajectory_ci(axes[1], rel, "pb_signal", reliability_only=True, title="PB_t rises with reliability pressure", ylabel=r"$PB_t$")
    trajectory_ci(axes[2], rel, "gen_gap_loss", reliability_only=True, title="Generalization gap tracks the signal", ylabel="Loss gap")
    scatter_with_regression(axes[3], rel, "pb_signal", "gen_gap_loss", "PB_t aligns with loss gap", r"$PB_t$", r"$G_t$")
    scatter_with_regression(axes[4], rel, "pb_signal", "val_nll", "PB_t aligns with validation NLL", r"$PB_t$", "Val NLL")
    corr = pb_seed_correlations(hist)
    if corr.empty:
        unavailable(axes[5])
    else:
        axes[5].axhline(0, color=PALETTE["lightgray"], lw=0.8)
        labels = []
        rng = np.random.default_rng(3)
        for i, (exp, d) in enumerate(corr.groupby("experiment")):
            vals = d["rho_pb_gengap"].dropna().astype(float).to_numpy()
            labels.append(exp)
            axes[5].scatter(np.full(len(vals), i) + rng.normal(0, 0.04, len(vals)), vals,
                            s=20, color=PALETTE["reliability"], alpha=0.75, edgecolor="white", linewidth=0.3)
            if len(vals):
                axes[5].plot([i - 0.18, i + 0.18], [np.mean(vals), np.mean(vals)], color=PALETTE["axis"], lw=1.5)
        axes[5].set_xticks(range(len(labels)))
        axes[5].set_xticklabels(labels, rotation=30, ha="right")
        axes[5].set_ylabel(r"$\rho(PB_t, G_t)$")
        axes[5].set_title("Positive association holds across runs")
    for ax, lab in zip(axes, "abcdef"):
        add_panel_label(ax, lab)
    fig.suptitle("Figure 4.6 | PB_t tracks generalization reliability", y=1.02, fontsize=10)
    save_figure(fig, Path(out_dir) / "fig_4_6_conclusion_pb_generalization")


def pd_seed_spikes(hist):
    rows = []
    if hist.empty or "pd_signal" not in hist.columns:
        return pd.DataFrame(rows)
    df = hist[hist["method"] == "reliability"] if "method" in hist.columns else hist
    group_cols = [c for c in ["experiment", "dataset", "model", "seed"] if c in df.columns]
    for keys, d in df.groupby(group_cols):
        vals = pd.to_numeric(d["pd_signal"], errors="coerce")
        th = vals.mean() + 2 * vals.std(ddof=0)
        spikes = vals > th
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["spike_count"] = int(spikes.sum())
        row["spike_area"] = float((vals[spikes] - th).clip(lower=0).sum()) if spikes.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def figure_47(hist, out_dir):
    df = subset_first_experiment(hist)
    rel = df[df.get("method", "") == "reliability"] if not df.empty and "method" in df.columns else pd.DataFrame()
    fig, axes = plt.subplots(3, 2, figsize=(7.8, 8.2), constrained_layout=True)
    axes = axes.ravel()
    add_schematic(axes[0], "PD_t summarizes optimization pressure", [r"$V_t$", r"$H_t$", r"$U_t$", r"$PD_t$"])
    trajectory_ci(axes[1], rel, "pd_signal", reliability_only=True, title="PD_t exposes instability phases", ylabel=r"$PD_t$")
    if rel.empty or not {"grad_norm", "update_norm", "epoch"}.issubset(rel.columns):
        unavailable(axes[2])
    else:
        seed = rel["seed"].dropna().unique()[0] if "seed" in rel.columns and len(rel["seed"].dropna()) else None
        d = rel[rel["seed"] == seed] if seed is not None else rel
        sc = axes[2].scatter(d["grad_norm"], d["update_norm"], c=d["epoch"], cmap="viridis", s=18, alpha=0.8)
        plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.055, label="Epoch")
        axes[2].set_xlabel(r"$H_t$ gradient norm")
        axes[2].set_ylabel(r"$U_t$ update norm")
        axes[2].set_title("Gradient-update phases separate")
    trajectory_ci(axes[3], rel, "risk_violation_vt", reliability_only=True, title="Risk violation co-varies with PD_t", ylabel=r"$V_t$")
    if rel.empty or not {"pd_signal", "epoch", "seed"}.issubset(rel.columns):
        unavailable(axes[4])
    else:
        labels = []
        y = 0
        for (exp, seed), d in rel.groupby(["experiment", "seed"]):
            vals = pd.to_numeric(d["pd_signal"], errors="coerce")
            th = vals.mean() + 2 * vals.std(ddof=0)
            spikes = d.loc[vals > th, "epoch"]
            axes[4].vlines(spikes, y - 0.35, y + 0.35, color=PALETTE["reliability"], lw=1.0)
            labels.append(f"{exp}:s{seed}")
            y += 1
            if y >= 18:
                break
        axes[4].set_yticks(range(len(labels)))
        axes[4].set_yticklabels(labels, fontsize=6)
        axes[4].set_xlabel("Epoch")
        axes[4].set_title("Spike events localize instability")
    spikes = pd_seed_spikes(hist)
    if spikes.empty:
        unavailable(axes[5])
    else:
        rng = np.random.default_rng(4)
        labels = []
        for i, (exp, d) in enumerate(spikes.groupby("experiment")):
            vals = d["spike_count"].to_numpy(float)
            labels.append(exp)
            axes[5].scatter(np.full(len(vals), i) + rng.normal(0, 0.04, len(vals)), vals,
                            s=20, color=PALETTE["reliability"], alpha=0.75, edgecolor="white", linewidth=0.3)
            axes[5].plot([i - 0.18, i + 0.18], [np.mean(vals), np.mean(vals)], color=PALETTE["axis"], lw=1.5)
        axes[5].set_xticks(range(len(labels)))
        axes[5].set_xticklabels(labels, rotation=30, ha="right")
        axes[5].set_ylabel("Spike count")
        axes[5].set_title("Instability burden varies by run")
    for ax, lab in zip(axes, "abcdef"):
        add_panel_label(ax, lab)
    fig.suptitle("Figure 4.7 | PD_t captures optimization instability", y=1.02, fontsize=10)
    save_figure(fig, Path(out_dir) / "fig_4_7_conclusion_pd_stability")


def figure_48(hist, summary, out_dir):
    df = subset_first_experiment(hist)
    sdf = summary.copy()
    fig, axes = plt.subplots(3, 2, figsize=(7.9, 8.3), constrained_layout=True)
    axes = axes.ravel()
    add_schematic(axes[0], "Reliability control converts signals into actions",
                  [r"$PB_t + PD_t$", r"$R_t$", r"$\bar{R}_t$", "LR drop\n/ stop"])
    trajectory_ci(axes[1], df, "lr", title="Ours changes LR at distinct phases", ylabel="Learning rate")
    axes[1].set_yscale("log")
    trajectory_ci(axes[2], df, "val_acc", title="Accuracy trajectory remains competitive", ylabel="Val accuracy")
    trajectory_ci(axes[3], df, "val_nll", title="Validation risk decreases smoothly", ylabel="Val NLL")
    if df.empty or not {"method", "seed", "epoch"}.issubset(df.columns):
        unavailable(axes[4])
    else:
        y = 0
        labels = []
        for (method, seed), d in df.groupby(["method", "seed"]):
            if y >= 24:
                break
            if "lr_reduced" in d.columns:
                reduced = d.loc[d["lr_reduced"].astype(str).str.lower().isin(["true", "1"]), "epoch"]
            else:
                lr = pd.to_numeric(d["lr"], errors="coerce")
                reduced = d.loc[lr.diff().fillna(0) < 0, "epoch"]
            axes[4].vlines(reduced, y - 0.35, y + 0.35, color=color_for(method), lw=1.0)
            axes[4].scatter([d["epoch"].max()], [y], marker="|", color=PALETTE["axis"], s=35)
            labels.append(f"{method_label(method)} s{seed}")
            y += 1
        axes[4].set_yticks(range(len(labels)))
        axes[4].set_yticklabels(labels, fontsize=5.8)
        axes[4].set_xlabel("Epoch")
        axes[4].set_title("Control events align to training phases", pad=8)
    dot_plot(axes[5], sdf, "test_acc", "Endpoint accuracy favors Ours", "Test accuracy",
             higher=True, significance=True)
    for ax, lab in zip(axes, "abcdef"):
        add_panel_label(ax, lab)
    fig.suptitle("Figure 4.8 | The controller adapts learning rate at meaningful phases", y=1.02, fontsize=10)
    save_figure(fig, Path(out_dir) / "fig_4_8_conclusion_lr_control")


def read_calibration(results_dir, experiment, method):
    rows = []
    for path in Path(results_dir).glob(f"{experiment}/{method}/seed_*/calibration_bins.csv"):
        try:
            d = pd.read_csv(path)
            d["seed"] = path.parent.name.replace("seed_", "")
            rows.append(d)
        except Exception:
            continue
    if rows:
        df = pd.concat(rows, ignore_index=True)
        group_key = "bin_low" if "bin_low" in df.columns else "confidence"
        return df.groupby(group_key, as_index=False)[["confidence", "accuracy"]].mean(numeric_only=True)
    for seed_dir in Path(results_dir).glob(f"{experiment}/{method}/seed_*"):
        logits_path = seed_dir / "test_logits.pt"
        targets_path = seed_dir / "test_targets.pt"
        if logits_path.exists() and targets_path.exists():
            try:
                logits = torch.load(logits_path, map_location="cpu")
                targets = torch.load(targets_path, map_location="cpu")
                return pd.DataFrame(reliability_bins(logits, targets))
            except Exception:
                continue
    return pd.DataFrame()


def read_risk_coverage(results_dir, experiment, method):
    curves = []
    grid = np.linspace(0.02, 1.0, 100)
    for path in Path(results_dir).glob(f"{experiment}/{method}/seed_*/risk_coverage_curve.csv"):
        try:
            d = pd.read_csv(path).dropna()
            if {"coverage", "risk"}.issubset(d.columns) and len(d) > 2:
                curves.append(np.interp(grid, d["coverage"].to_numpy(float), d["risk"].to_numpy(float)))
        except Exception:
            continue
    if curves:
        return pd.DataFrame({"coverage": grid, "risk": np.mean(curves, axis=0)})
    for seed_dir in Path(results_dir).glob(f"{experiment}/{method}/seed_*"):
        logits_path = seed_dir / "test_logits.pt"
        targets_path = seed_dir / "test_targets.pt"
        if logits_path.exists() and targets_path.exists():
            try:
                logits = torch.load(logits_path, map_location="cpu")
                targets = torch.load(targets_path, map_location="cpu")
                coverage, risk = risk_coverage_curve(logits, targets)
                return pd.DataFrame({"coverage": coverage, "risk": risk})
            except Exception:
                continue
    return pd.DataFrame()


def figure_49(summary, results_dir, out_dir):
    sdf = summary.copy()
    exp = choose_experiment(summary)
    fig, axes = plt.subplots(3, 2, figsize=(7.9, 8.3), constrained_layout=True)
    axes = axes.ravel()
    dot_plot(axes[0], sdf, "test_acc", "Accuracy is preserved", "Test accuracy", higher=True, significance=True)
    dot_plot(axes[1], sdf, "test_nll", "Predictive risk improves", "Test NLL", higher=False, significance=True)
    dot_plot(axes[2], sdf, "test_ece", "Calibration error decreases", "ECE", higher=False, significance=True)
    dot_plot(axes[3], sdf, "test_brier", "Probabilistic reliability improves", "Brier score", higher=False, significance=True)
    if exp is None:
        unavailable(axes[4])
        unavailable(axes[5])
    else:
        for method in ["cosine", "plateau", "reliability"]:
            d = read_calibration(results_dir, exp, method)
            if not d.empty and {"confidence", "accuracy"}.issubset(d.columns):
                lw = 2.0 if method == "reliability" else 1.2
                axes[4].plot(d["confidence"], d["accuracy"], marker="o", ms=3, lw=lw,
                             color=color_for(method), label=method_label(method))
        axes[4].plot([0, 1], [0, 1], "--", lw=1.0, color=PALETTE["midgray"])
        axes[4].set_xlim(0, 1)
        axes[4].set_ylim(0, 1)
        axes[4].set_xlabel("Confidence")
        axes[4].set_ylabel("Accuracy")
        axes[4].set_title("Reliability diagram moves toward diagonal", pad=8)
        place_legend_safely(axes[4], preferred="lower right", outside_if_needed="always")
        for method in ["cosine", "plateau", "reliability"]:
            d = read_risk_coverage(results_dir, exp, method)
            if not d.empty and {"coverage", "risk"}.issubset(d.columns):
                lw = 2.0 if method == "reliability" else 1.2
                axes[5].plot(d["coverage"], d["risk"], lw=lw, color=color_for(method), label=method_label(method))
        axes[5].set_xlabel("Coverage")
        axes[5].set_ylabel("Risk")
        axes[5].set_title("Selective risk falls at matched coverage", pad=8)
        place_legend_safely(axes[5], preferred="upper right", outside_if_needed="always")
    for ax, lab in zip(axes, "abcdef"):
        add_panel_label(ax, lab)
    fig.suptitle("Figure 4.9 | The controller improves calibration and reliability beyond accuracy", y=1.02, fontsize=10)
    save_figure(fig, Path(out_dir) / "fig_4_9_conclusion_calibration_reliability")


def figure_410(summary, out_dir):
    sdf = summary.copy()
    fig, axes = plt.subplots(3, 2, figsize=(7.8, 8.2), constrained_layout=True)
    axes = axes.ravel()
    dot_plot(axes[0], sdf, "test_acc", "Accuracy is less seed-sensitive", "Test accuracy", sd=True)
    dot_plot(axes[1], sdf, "test_nll", "Risk variance is reduced", "Test NLL", sd=True, higher=False)
    dot_plot(axes[2], sdf, "test_ece", "Calibration varies less", "ECE", sd=True, higher=False)
    dot_plot(axes[3], sdf, "best_epoch", "Best epoch is more stable", "Best epoch", sd=True, higher=False)
    cv_metrics = ["test_acc", "test_nll", "test_ece", "best_epoch", "epochs_run"]
    if sdf.empty or "method" not in sdf.columns:
        unavailable(axes[4])
    else:
        methods = ordered_methods(sdf["method"].dropna().unique())
        mat = np.full((len(methods), len(cv_metrics)), np.nan)
        for i, method in enumerate(methods):
            d = sdf[sdf["method"] == method]
            for j, metric in enumerate(cv_metrics):
                if metric in d.columns:
                    vals = pd.to_numeric(d[metric], errors="coerce").dropna().to_numpy(float)
                    if len(vals) > 1 and np.mean(np.abs(vals)) != 0:
                        mat[i, j] = np.std(vals, ddof=1) / abs(np.mean(vals))
        im = axes[4].imshow(mat, cmap="magma_r", aspect="auto")
        axes[4].set_xticks(range(len(cv_metrics)))
        axes[4].set_xticklabels(cv_metrics, rotation=35, ha="right")
        axes[4].set_yticks(range(len(methods)))
        axes[4].set_yticklabels([method_label(m) for m in methods])
        axes[4].set_title("Coefficient-of-variation heatmap")
        plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.055)
    if sdf.empty or not {"method", "seed", "test_acc"}.issubset(sdf.columns):
        unavailable(axes[5])
    else:
        data = sdf[["method", "seed", "test_acc"]].dropna()
        methods = ordered_methods(data["method"].unique())
        seeds = sorted(data["seed"].unique())
        mat = np.full((len(methods), len(seeds)), np.nan)
        for j, seed in enumerate(seeds):
            d = data[data["seed"] == seed].copy()
            d["rank"] = d["test_acc"].rank(ascending=False, method="min")
            for i, method in enumerate(methods):
                vals = d.loc[d["method"] == method, "rank"]
                if len(vals):
                    mat[i, j] = vals.iloc[0]
        im = axes[5].imshow(mat, cmap="viridis_r", aspect="auto")
        axes[5].set_xticks(range(len(seeds)))
        axes[5].set_xticklabels([str(s) for s in seeds])
        axes[5].set_yticks(range(len(methods)))
        axes[5].set_yticklabels([method_label(m) for m in methods])
        axes[5].set_xlabel("Seed")
        axes[5].set_title("Rank stability across seeds")
        plt.colorbar(im, ax=axes[5], fraction=0.046, pad=0.055, label="Rank")
    for ax, lab in zip(axes, "abcdef"):
        add_panel_label(ax, lab)
    fig.suptitle("Figure 4.10 | The controller improves seed-level stability", y=1.02, fontsize=10)
    save_figure(fig, Path(out_dir) / "fig_4_10_conclusion_seed_stability")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures_conclusion")
    parser.add_argument("--include-quick", action="store_true", help="Include quick-test experiments in outputs.")
    args = parser.parse_args()

    set_nature_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = filter_quick(collect_histories(args.results_dir), include_quick=args.include_quick)
    summary = filter_quick(collect_summaries(args.results_dir), include_quick=args.include_quick)

    figure_46(hist, out_dir)
    figure_47(hist, out_dir)
    figure_48(hist, summary, out_dir)
    figure_49(summary, args.results_dir, out_dir)
    figure_410(summary, out_dir)

    print(f"Saved conclusion figures to {out_dir}")


if __name__ == "__main__":
    main()
