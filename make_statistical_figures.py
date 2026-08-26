"""将多种子统计结果绘制为论文补充实验图。"""

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
    PALETTE, set_nature_style,
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
    "reliability_pb_only",
    "reliability_pd_only",
    "reliability_no_smoothing",
    "reliability_val_loss_only",
]

METHOD_AXIS_LABELS = {
    "reliability": "Ours /\nFull",
    "reliability_full": "Ours /\nFull",
    "reliability_pb_only": "PB-only",
    "reliability_pd_only": "PD-only",
    "reliability_no_smoothing": "No\nsmoothing",
    "reliability_no_ema": "No\nsmoothing",
    "reliability_val_loss_only": "Val-loss\nonly",
    "reliability_monitor_only": "Monitor-\nonly",
}

SHORT_SETTING_LABELS = {
    ("cifar10", "resnet18"): "C10-R18",
    ("cifar10", "resnet50"): "C10-R50",
    ("cifar10", "vgg16"): "C10-V16",
    ("cifar100", "resnet18"): "C100-R18",
    ("cifar100", "resnet50"): "C100-R50",
    ("cifar100", "vgg16"): "C100-V16",
}


def collect_summaries(results_dir):
    """从主表或逐运行 JSON 收集正式实验摘要。"""

    results_dir = Path(results_dir)
    master = results_dir / "master_summary.csv"
    if master.exists():
        try:
            return pd.read_csv(master)
        except Exception:
            pass
    rows = []
    for path in results_dir.rglob("summary.json"):
        if any(part in {"_archive", "smoke"} for part in path.parts):
            continue
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
    """从实验名称提取 alpha/beta 敏感性坐标。"""

    match = re.search(r"alpha(\d{3})_beta(\d{3})", str(name))
    if not match:
        return np.nan, np.nan
    return int(match.group(1)) / 100.0, int(match.group(2)) / 100.0


def add_alpha_beta(summary):
    """为可解析的敏感性运行添加数值超参数列。"""

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
    """在缺少结果时绘制明确的不可用占位面板。"""

    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=PALETTE["midgray"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def method_label(method):
    """返回方法的短显示名称。"""

    return METHOD_LABELS.get(str(method), str(method))


def method_axis_label(method):
    """返回适合坐标轴空间的换行方法标签。"""

    return METHOD_AXIS_LABELS.get(str(method), method_label(method))


def method_color(method):
    """从共享调色板选择方法颜色。"""

    return PALETTE.get(method, PALETTE["gray"])


def short_setting_label(dataset, model):
    """生成数据集/模型组合的紧凑标签。"""

    return SHORT_SETTING_LABELS.get((str(dataset).lower(), str(model).lower()), f"{dataset}-{model}")


def save_stat_figure(fig, out_base):
    """按统计图规范同时保存矢量和位图版本。"""

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.06, wspace=0.12, hspace=0.16)
    except Exception:
        pass
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)


def write_caption(out_dir, name, text):
    """把图注写为与图同名的文本文件。"""

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / name).write_text(text.strip() + "\n", encoding="utf-8")


def dot_plot(ax, df, metric, title, ylabel):
    """绘制各消融方法的均值点和 95% 置信区间。"""

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
    ax.set_xticklabels([method_axis_label(m) for m in methods], rotation=0, ha="center")
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))


def lr_epoch_panel(ax, df):
    """绘制停止轮数与学习率降低行为的联合面板。"""

    if df.empty or "method" not in df.columns:
        unavailable(ax, "ablation results unavailable")
        ax.set_title("(d) Control effort and training length", pad=8)
        return
    methods = [m for m in ABLATION_METHODS if m in set(df["method"])]
    if not methods:
        unavailable(ax, "ablation results unavailable")
        ax.set_title("(d) Control effort and training length", pad=8)
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
    ax.set_xticklabels([method_axis_label(m) for m in methods], rotation=0, ha="center")
    ax.set_ylabel("LR reductions")
    ax2.set_ylabel("Epochs run")
    ax.set_title("(d) Control effort and training length", pad=8)
    handles, labels = [], []
    for axis in [ax, ax2]:
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=6.5)


def figure_ablation(summary, out_dir):
    """绘制算法组件消融的性能与控制行为面板。"""

    exp_type = summary.get("experiment_type", pd.Series("", index=summary.index)).astype(str) if not summary.empty else pd.Series(dtype=str)
    df = summary[exp_type.isin(["ablation", "pd_component_ablation"])].copy() if not summary.empty else pd.DataFrame()
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.5), constrained_layout=True)
    axes = axes.ravel()

    dot_plot(axes[0], df, "test_acc", "(a) Endpoint accuracy across variants", "Test accuracy")
    dot_plot(axes[1], df, "test_ece", "(b) Calibration error across variants", "ECE")
    dot_plot(axes[2], df, "test_brier", "(c) Brier score across variants", "Brier score")
    lr_epoch_panel(axes[3], df)

    fig.suptitle("Ablation analysis of controller design choices", y=1.02, fontsize=10)
    save_stat_figure(fig, Path(out_dir) / "fig_4_11_ablation_study")


def heatmap(ax, df, metric, title, cmap, fmt="{:.3f}"):
    """将 alpha/beta 网格指标绘制为带数值标注的热图。"""

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
    norm = im.norm
    cmap_obj = im.cmap
    for yi in range(len(betas)):
        for xi in range(len(alphas)):
            if np.isfinite(matrix[yi, xi]):
                rgba = cmap_obj(norm(matrix[yi, xi]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if luminance > 0.58 else "white"
                ax.text(
                    xi,
                    yi,
                    fmt.format(matrix[yi, xi]),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=text_color,
                )
    if 0.50 in alphas and 0.90 in betas:
        xi = alphas.index(0.50)
        yi = betas.index(0.90)
        ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor="#D62728", linewidth=2.0))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.045)


def figure_sensitivity(summary, out_dir):
    """绘制 alpha/beta 敏感性热图。"""

    df = add_alpha_beta(summary)
    exp_type = df.get("experiment_type", pd.Series("", index=df.index)).astype(str) if not df.empty else pd.Series(dtype=str)
    df = df[exp_type == "sensitivity"].copy() if not df.empty else pd.DataFrame()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.5), constrained_layout=True)
    axes = axes.ravel()

    heatmap(axes[0], df, "test_acc", "(a) Test accuracy", "viridis")
    heatmap(axes[1], df, "test_ece", "(b) ECE", "magma_r")
    heatmap(axes[2], df, "test_brier", "(c) Brier score", "magma_r")
    heatmap(axes[3], df, "final_gen_gap_loss", "(d) Generalization gap", "magma_r")

    fig.suptitle("Hyperparameter sensitivity analysis", y=1.02, fontsize=10)
    save_stat_figure(fig, Path(out_dir) / "fig_4_12_sensitivity_analysis")
    write_caption(
        out_dir,
        "fig_4_12_sensitivity_analysis_caption.txt",
        "Figure 4.12. Hyperparameter sensitivity analysis. The red box marks the default setting.",
    )


def figure_statistical_tests(summary, out_dir):
    """绘制相对最优基线的效应和置信区间。"""

    if summary.empty:
        return

    exp_type = summary.get("experiment_type", pd.Series("", index=summary.index)).astype(str)
    df = summary[~exp_type.isin(["quick", "ablation", "sensitivity"])].copy()
    metrics = [
        ("test_acc", "(a) Test accuracy", "a"),
        ("test_nll", "(b) Test NLL", "b"),
        ("test_ece", "(c) ECE", "c"),
        ("test_brier", "(d) Brier score", "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.9), constrained_layout=True)
    axes = axes.ravel()

    for ax, (metric, title, label) in zip(axes, metrics):
        comp = compare_ours_vs_best_baseline(df, metric, ["dataset", "model", "experiment"])
        if comp.empty:
            unavailable(ax, f"{metric} unavailable")
            ax.set_title(title, pad=8)
            continue

        comp = comp.sort_values(["dataset", "model", "experiment"])
        comp = comp[pd.to_numeric(comp["mean_difference"], errors="coerce").notna()].copy()
        if comp.empty:
            unavailable(ax, f"{metric} unavailable")
            ax.set_title(title, pad=8)
            continue
        labels = [short_setting_label(row.dataset, row.model) for row in comp.itertuples(index=False)]
        y = pd.to_numeric(comp["mean_difference"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(comp["ci95_low"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(comp["ci95_high"], errors="coerce").to_numpy(float)
        x = np.arange(len(comp))
        yerr = np.vstack([y - lo, hi - y])
        yerr = np.where(np.isfinite(yerr), np.maximum(0.0, yerr), 0.0)

        ax.bar(x, y, yerr=yerr, capsize=2.5, color=PALETTE["reliability"], alpha=0.84)
        ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Paired difference")
        ax.set_title(title, pad=8)
        max_abs = np.nanmax(np.abs(y)) if len(y) else 1.0
        offset = 0.03 * max(max_abs, 1e-8)
        for idx, row in enumerate(comp.itertuples(index=False)):
            marker = str(row.significance)
            if marker in {"*", "**", "***"} and np.isfinite(y[idx]):
                if y[idx] >= 0:
                    ax.text(idx, y[idx] + offset, marker, ha="center", va="bottom", fontsize=8)
                else:
                    ax.text(idx, y[idx] - offset, marker, ha="center", va="top", fontsize=8)

    fig.suptitle("Paired differences between Ours and the best baseline", y=1.02, fontsize=10)
    save_stat_figure(fig, Path(out_dir) / "fig_4_13_statistical_tests")
    write_caption(
        out_dir,
        "fig_4_13_statistical_tests_caption.txt",
        (
            "Figure 4.13. Paired differences between Ours and the best baseline. "
            "For test accuracy, positive paired differences favor Ours. For NLL, ECE, "
            "and Brier score, negative paired differences favor Ours. Significance stars: "
            "* p < 0.05, ** p < 0.01, *** p < 0.001."
        ),
    )


def main():
    """加载正式结果并批量输出 PDF/PNG 图及图注。"""

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
