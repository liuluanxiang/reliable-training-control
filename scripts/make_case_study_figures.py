import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import set_nature_style, save_figure, panel_label, PALETTE, LABELS
from src.metrics import spearman_np
from src.results_io import collect_histories as load_histories, collect_summaries as load_summaries


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


def fig_case1(hist, out_dir):
    exp = representative_experiment(hist)
    df = hist[(hist["method"] == "reliability") & (hist["experiment"] == exp)].copy() if exp else pd.DataFrame()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8))

    for seed, d in df.groupby("seed"):
        axes[0, 0].plot(d["epoch"], d["pb_signal"], lw=1.0, alpha=0.85)
        axes[0, 1].plot(d["epoch"], d["gen_gap_loss"], lw=1.0, alpha=0.85)
    axes[0, 0].set_title(r"PAC-Bayes signal $PB_t$")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel(r"$PB_t$")
    axes[0, 1].set_title("Loss generalization gap")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Gap")
    panel_label(axes[0, 0], "a"); panel_label(axes[0, 1], "b")

    axes[1, 0].scatter(df["pb_signal"], df["gen_gap_loss"], s=8, alpha=0.4, color=PALETTE["reliability"])
    axes[1, 0].set_xlabel(r"$PB_t$"); axes[1, 0].set_ylabel("Generalization gap")
    axes[1, 0].set_title(r"$PB_t$ vs generalization gap")
    panel_label(axes[1, 0], "c")

    axes[1, 1].scatter(df["pb_signal"], df["val_nll"], s=8, alpha=0.4, color=PALETTE["gray"])
    axes[1, 1].set_xlabel(r"$PB_t$"); axes[1, 1].set_ylabel("Validation NLL")
    axes[1, 1].set_title(r"$PB_t$ vs validation NLL")
    panel_label(axes[1, 1], "d")

    cols = ["pb_signal", "gen_gap_loss", "val_nll", "val_ece", "reliability_rbar"]
    corr = df[cols].corr(numeric_only=True).values
    im = axes[2, 0].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    axes[2, 0].set_xticks(range(len(cols))); axes[2, 0].set_xticklabels(cols, rotation=45, ha="right")
    axes[2, 0].set_yticks(range(len(cols))); axes[2, 0].set_yticklabels(cols)
    axes[2, 0].set_title("Correlation heatmap")
    fig.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)
    panel_label(axes[2, 0], "e")

    exps, vals = [], []
    for exp, d in hist[hist["method"] == "reliability"].groupby("experiment"):
        if len(d) >= 3:
            exps.append(exp)
            vals.append(spearman_np(d["pb_signal"], d["gen_gap_loss"]))
    axes[2, 1].bar(range(len(exps)), vals, color=PALETTE["reliability"])
    axes[2, 1].set_xticks(range(len(exps))); axes[2, 1].set_xticklabels(exps, rotation=35, ha="right")
    axes[2, 1].set_ylabel(r"$\rho(PB_t, G_t)$")
    axes[2, 1].set_title("Cross-experiment association")
    panel_label(axes[2, 1], "f")

    save_figure(fig, Path(out_dir) / "fig_4_6_case1_pac_bayes_signal_validity")


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
