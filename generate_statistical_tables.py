import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stats_utils import (  # noqa: E402
    compare_ours_vs_best_baseline, format_p_value, mean_std,
)
from src.results_io import filter_quick  # noqa: E402


HIGHER_BETTER = {"test_acc", "test_top5_acc", "aulc_val_acc"}
LOWER_BETTER = {
    "test_nll", "test_ece", "test_mce", "test_brier", "test_aurc", "test_eaurc",
    "final_gen_gap_loss", "late_degradation", "epochs_run", "lr_reductions",
}

METHOD_LABELS = {
    "step": "Step",
    "cosine": "Cosine",
    "plateau": "Plateau",
    "reliability": "Ours / Full",
    "reliability_pb_only": "PB-only",
    "reliability_pd_only": "PD-only",
    "reliability_no_smoothing": "No smoothing",
    "reliability_val_loss_only": "Val-loss only",
}

ABLATION_METHODS = [
    "reliability",
    "reliability_pb_only",
    "reliability_pd_only",
    "reliability_no_smoothing",
    "reliability_val_loss_only",
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


def method_label(method):
    return METHOD_LABELS.get(str(method), str(method))


def metric_higher(metric):
    if metric in LOWER_BETTER:
        return False
    return True


def fmt_number(value, digits=4):
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_mean_std(values, digits=4):
    mean, std = mean_std(values)
    if not np.isfinite(mean):
        return ""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def parse_sensitivity_experiment(name):
    match = re.search(r"alpha(\d{3})_beta(\d{3})", str(name))
    if not match:
        return np.nan, np.nan
    alpha = int(match.group(1)) / 100.0
    beta = int(match.group(2)) / 100.0
    return alpha, beta


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


def best_indices(df, metric_cols):
    best = {}
    for col in metric_cols:
        vals = df[col].astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
        vals = pd.to_numeric(vals, errors="coerce")
        if not vals.notna().any():
            continue
        raw_metric = metric_cols[col]
        best[col] = vals.idxmax() if metric_higher(raw_metric) else vals.idxmin()
    return best


def markdown_table(df, path, best_map=None, note=None):
    md = df.copy().astype(object)
    if best_map:
        for col, idx in best_map.items():
            if idx in md.index and col in md.columns and str(md.loc[idx, col]):
                md.loc[idx, col] = f"**{md.loc[idx, col]}**"

    def esc(value):
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|")

    lines = []
    lines.append("| " + " | ".join(md.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(md.columns)) + " |")
    for _, row in md.iterrows():
        lines.append("| " + " | ".join(esc(row[col]) for col in md.columns) + " |")
    if note:
        lines.append("")
        lines.append(note)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_ablation(summary):
    columns = [
        "Dataset", "Model", "Method", "Test Acc", "Test NLL", "ECE", "Brier",
        "Gen Gap", "AULC", "LR Reductions", "Epochs Run",
    ]
    metric_map = {
        "Test Acc": "test_acc",
        "Test NLL": "test_nll",
        "ECE": "test_ece",
        "Brier": "test_brier",
        "Gen Gap": "final_gen_gap_loss",
        "AULC": "aulc_val_acc",
        "LR Reductions": "lr_reductions",
        "Epochs Run": "epochs_run",
    }
    if summary.empty or "method" not in summary.columns:
        return pd.DataFrame(columns=columns), {}

    exp_type = summary.get("experiment_type", pd.Series("", index=summary.index)).astype(str)
    df = summary[summary["method"].isin(ABLATION_METHODS) & (exp_type == "ablation")].copy()
    rows = []
    for (dataset, model, method), group in df.groupby(["dataset", "model", "method"], dropna=False):
        row = {"Dataset": dataset, "Model": model, "Method": method_label(method)}
        for label, metric in metric_map.items():
            row[label] = fmt_mean_std(group[metric]) if metric in group.columns else ""
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    return out, best_indices(out, metric_map)


def table_sensitivity(summary):
    columns = ["Alpha", "Beta", "Test Acc", "Test NLL", "ECE", "Brier", "Gen Gap", "AULC", "Epochs Run"]
    metric_map = {
        "Test Acc": "test_acc",
        "Test NLL": "test_nll",
        "ECE": "test_ece",
        "Brier": "test_brier",
        "Gen Gap": "final_gen_gap_loss",
        "AULC": "aulc_val_acc",
        "Epochs Run": "epochs_run",
    }
    if summary.empty:
        return pd.DataFrame(columns=columns), {}

    df = add_alpha_beta(summary)
    exp_type = df.get("experiment_type", pd.Series("", index=df.index)).astype(str)
    df = df[(exp_type == "sensitivity") & (df.get("method", pd.Series("", index=df.index)) == "reliability")].copy()
    rows = []
    for (alpha, beta), group in df.groupby(["controller_alpha", "controller_beta"], dropna=False):
        row = {"Alpha": fmt_number(alpha, 2), "Beta": fmt_number(beta, 2)}
        for label, metric in metric_map.items():
            row[label] = fmt_mean_std(group[metric]) if metric in group.columns else ""
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    return out, best_indices(out, metric_map)


def table_statistical_tests(summary):
    columns = [
        "Dataset", "Model", "Metric", "Ours", "Best Baseline", "Best Baseline Name",
        "Mean Difference", "Cohen d", "Paired t-test p", "Wilcoxon p", "Significance",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)

    df = summary.copy()
    exp_type = df.get("experiment_type", pd.Series("", index=df.index)).astype(str)
    df = df[~exp_type.isin(["quick", "ablation", "sensitivity"])]

    metric_labels = {
        "test_acc": "Test Acc",
        "test_top5_acc": "Test Top-5 Acc",
        "aulc_val_acc": "AULC",
        "test_nll": "Test NLL",
        "test_ece": "ECE",
        "test_mce": "MCE",
        "test_brier": "Brier",
        "test_aurc": "AURC",
        "test_eaurc": "EAURC",
        "final_gen_gap_loss": "Gen Gap",
        "late_degradation": "Late Degradation",
    }
    rows = []
    for metric, label in metric_labels.items():
        comp = compare_ours_vs_best_baseline(df, metric, ["dataset", "model", "experiment"])
        for _, row in comp.iterrows():
            rows.append({
                "Dataset": row["dataset"],
                "Model": row["model"],
                "Metric": label,
                "Ours": fmt_number(row["ours"]),
                "Best Baseline": fmt_number(row["best_baseline"]),
                "Best Baseline Name": method_label(row["best_baseline_name"]),
                "Mean Difference": fmt_number(row["mean_difference"]),
                "Cohen d": fmt_number(row["cohen_d"]),
                "Paired t-test p": format_p_value(row["paired_ttest_p"]),
                "Wilcoxon p": format_p_value(row["wilcoxon_p"]),
                "Significance": row["significance"],
            })
    return pd.DataFrame(rows, columns=columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="statistical_tables")
    parser.add_argument("--include-quick", action="store_true", help="Include quick-test experiments in outputs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = filter_quick(collect_summaries(args.results_dir), include_quick=args.include_quick)

    ablation, ablation_best = table_ablation(summary)
    ablation.to_csv(out_dir / "table_4_8_ablation_study.csv", index=False)
    markdown_table(ablation, out_dir / "table_4_8_ablation_study.md", ablation_best)

    sensitivity, sensitivity_best = table_sensitivity(summary)
    sensitivity.to_csv(out_dir / "table_4_9_sensitivity_alpha_beta.csv", index=False)
    markdown_table(sensitivity, out_dir / "table_4_9_sensitivity_alpha_beta.md", sensitivity_best)

    tests = table_statistical_tests(summary)
    tests.to_csv(out_dir / "table_4_10_statistical_tests.csv", index=False)
    tests_md = tests.copy()
    if not tests_md.empty and {"Ours", "Significance"}.issubset(tests_md.columns):
        higher_labels = {"Test Acc", "Test Top-5 Acc", "AULC"}
        for idx, row in tests_md.iterrows():
            metric = str(row.get("Metric", ""))
            ours = pd.to_numeric(row.get("Ours"), errors="coerce")
            baseline = pd.to_numeric(row.get("Best Baseline"), errors="coerce")
            marker = str(row["Significance"])
            ours_text = str(row["Ours"])
            baseline_text = str(row["Best Baseline"])
            marker_suffix = f" {marker}" if marker in {"*", "**", "***"} else ""
            if np.isfinite(ours) and np.isfinite(baseline):
                ours_best = ours >= baseline if metric in higher_labels else ours <= baseline
                if ours_best:
                    ours_text = f"**{ours_text}**{marker_suffix}"
                else:
                    baseline_text = f"**{baseline_text}**"
                    ours_text = f"{ours_text}{marker_suffix}"
            tests_md.at[idx, "Ours"] = ours_text
            tests_md.at[idx, "Best Baseline"] = baseline_text
    markdown_table(
        tests_md,
        out_dir / "table_4_10_statistical_tests.md",
        note="Statistical tests compare Ours with the best baseline using paired seeds. * p < 0.05, ** p < 0.01, *** p < 0.001.",
    )

    print(f"Saved statistical tables to {out_dir}")


if __name__ == "__main__":
    main()
