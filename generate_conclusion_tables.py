import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import area_under_curve  # noqa: E402
from src.plot_style import METHOD_LABELS, METHOD_ORDER  # noqa: E402
from src.results_io import filter_quick  # noqa: E402
from src.stats_utils import format_p_value, paired_ttest, p_stars, spearman_with_p  # noqa: E402


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


def method_label(method):
    return METHOD_LABELS.get(method, str(method))


def ordered_methods(methods):
    present = set(methods)
    return [m for m in METHOD_ORDER if m in present] + sorted([m for m in present if m not in METHOD_ORDER])


def fmt_num(value, digits=4):
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_mean_std(values, digits=4):
    vals = pd.Series(values).dropna().astype(float)
    if len(vals) == 0:
        return ""
    if len(vals) == 1:
        return f"{vals.iloc[0]:.{digits}f}"
    return f"{vals.mean():.{digits}f} ± {vals.std(ddof=1):.{digits}f}"


def metric_direction(metric):
    if metric in HIGHER_BETTER:
        return "higher"
    if metric in LOWER_BETTER:
        return "lower"
    if metric.endswith("_std") or metric in {"best_epoch", "epochs_run", "lr_reductions", "first_lr_drop_epoch"}:
        return "lower"
    return None


def best_method_for_metric(group, metric, direction):
    if direction is None or metric not in group.columns:
        return None
    vals = group.groupby("method")[metric].mean(numeric_only=True).dropna()
    if vals.empty:
        return None
    return vals.idxmax() if direction == "higher" else vals.idxmin()


def best_baseline_for_metric(group, metric, direction):
    if direction is None or metric not in group.columns:
        return None
    baselines = group[group["method"] != "reliability"]
    if baselines.empty:
        return None
    vals = baselines.groupby("method")[metric].mean(numeric_only=True).dropna()
    if vals.empty:
        return None
    return vals.idxmax() if direction == "higher" else vals.idxmin()


def paired_p_for_ours(group, metric, baseline):
    if baseline is None or "seed" not in group.columns or metric not in group.columns:
        return np.nan
    ours = group[group["method"] == "reliability"].set_index("seed")[metric]
    base = group[group["method"] == baseline].set_index("seed")[metric]
    seeds = ours.index.intersection(base.index)
    if len(seeds) < 2:
        return np.nan
    return paired_ttest(ours.loc[seeds].to_numpy(), base.loc[seeds].to_numpy())


def is_ours_better(group, metric, baseline, direction):
    if baseline is None or direction is None:
        return False
    means = group.groupby("method")[metric].mean(numeric_only=True)
    if "reliability" not in means or baseline not in means:
        return False
    if direction == "higher":
        return means["reliability"] > means[baseline]
    return means["reliability"] < means[baseline]


def table_43(hist):
    rows = []
    required = {"method", "dataset", "model", "seed", "pb_signal", "gen_gap_loss", "val_nll"}
    if hist.empty or not required.issubset(hist.columns):
        return pd.DataFrame(columns=[
            "Dataset", "Model", "Seed", "rho(PB_t,G_t)", "p-value (G_t)",
            "rho(PB_t,Val NLL)", "p-value (Val NLL)", "PB AUC", "Final PB",
        ])
    df = hist[hist["method"] == "reliability"].copy()
    for (dataset, model, seed), d in df.groupby(["dataset", "model", "seed"]):
        rho_g, p_g = spearman_with_p(d["pb_signal"], d["gen_gap_loss"])
        rho_nll, p_nll = spearman_with_p(d["pb_signal"], d["val_nll"])
        rows.append({
            "Dataset": dataset,
            "Model": model,
            "Seed": seed,
            "rho(PB_t,G_t)": rho_g,
            "p-value (G_t)": p_g,
            "rho(PB_t,Val NLL)": rho_nll,
            "p-value (Val NLL)": p_nll,
            "PB AUC": area_under_curve(d["pb_signal"].to_numpy(float)),
            "Final PB": pd.to_numeric(d["pb_signal"], errors="coerce").dropna().iloc[-1] if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def table_44(hist):
    rows = []
    required = {"method", "dataset", "model", "seed", "pd_signal", "grad_norm", "update_norm", "risk_violation_vt"}
    if hist.empty or not required.issubset(hist.columns):
        return pd.DataFrame(columns=[
            "Dataset", "Model", "Seed", "rho(PD_t,H_t)", "rho(PD_t,U_t)", "rho(PD_t,V_t)",
            "PD AUC", "Spike count", "Spike area",
        ])
    df = hist[hist["method"] == "reliability"].copy()
    for (dataset, model, seed), d in df.groupby(["dataset", "model", "seed"]):
        pd_vals = pd.to_numeric(d["pd_signal"], errors="coerce")
        threshold = pd_vals.mean() + 2.0 * pd_vals.std(ddof=0)
        spikes = pd_vals > threshold
        rho_h, _ = spearman_with_p(pd_vals, d["grad_norm"])
        rho_u, _ = spearman_with_p(pd_vals, d["update_norm"])
        rho_v, _ = spearman_with_p(pd_vals, d["risk_violation_vt"])
        rows.append({
            "Dataset": dataset,
            "Model": model,
            "Seed": seed,
            "rho(PD_t,H_t)": rho_h,
            "rho(PD_t,U_t)": rho_u,
            "rho(PD_t,V_t)": rho_v,
            "PD AUC": area_under_curve(pd_vals.to_numpy(float)),
            "Spike count": int(spikes.sum()),
            "Spike area": float((pd_vals[spikes] - threshold).clip(lower=0).sum()) if spikes.any() else 0.0,
        })
    return pd.DataFrame(rows)


def aggregate_rows(summary, metrics):
    rows = []
    columns = ["Dataset", "Model", "Method"] + [label for _, label in metrics]
    if summary.empty or "method" not in summary.columns:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    for (dataset, model), group in summary.groupby(["dataset", "model"]):
        best_by_metric = {
            metric: best_method_for_metric(group, metric, metric_direction(metric))
            for metric, _ in metrics
        }
        best_baseline_by_metric = {
            metric: best_baseline_for_metric(group, metric, metric_direction(metric))
            for metric, _ in metrics
        }
        p_by_metric = {
            metric: paired_p_for_ours(group, metric, best_baseline_by_metric[metric])
            for metric, _ in metrics
        }
        for method in ordered_methods(group["method"].dropna().unique()):
            d = group[group["method"] == method]
            csv_row = {"Dataset": dataset, "Model": model, "Method": method_label(method)}
            md_row = {"Dataset": dataset, "Model": model, "Method": method_label(method)}
            for metric, label in metrics:
                value = fmt_mean_std(d[metric]) if metric in d.columns else ""
                csv_row[label] = value
                md_value = value
                if value:
                    if method == best_by_metric.get(metric):
                        md_value = f"**{md_value}**"
                    p = p_by_metric.get(metric, np.nan)
                    baseline = best_baseline_by_metric.get(metric)
                    direction = metric_direction(metric)
                    if (
                        method == "reliability"
                        and np.isfinite(p)
                        and p < 0.05
                        and is_ours_better(group, metric, baseline, direction)
                    ):
                        md_value = f"{md_value}†"
                md_row[label] = md_value
            rows.append((csv_row, md_row))
    csv_df = pd.DataFrame([r[0] for r in rows], columns=columns)
    md_df = pd.DataFrame([r[1] for r in rows], columns=columns)
    return csv_df, md_df


def table_45(summary):
    metrics = [
        ("test_acc", "Test Acc"),
        ("test_nll", "Test NLL"),
        ("test_ece", "ECE"),
        ("final_gen_gap_loss", "Gen Gap"),
        ("aulc_val_acc", "AULC"),
        ("best_epoch", "Best Epoch"),
        ("epochs_run", "Epochs Run"),
        ("lr_reductions", "LR Reductions"),
        ("first_lr_drop_epoch", "First LR Drop Epoch"),
    ]
    return aggregate_rows(summary, metrics)


def table_46(summary):
    metrics = [
        ("test_acc", "Test Acc"),
        ("test_nll", "Test NLL"),
        ("test_ece", "ECE"),
        ("test_mce", "MCE"),
        ("test_brier", "Brier"),
        ("test_aurc", "AURC"),
        ("test_eaurc", "EAURC"),
        ("test_confidence_mean", "Confidence Mean"),
        ("test_entropy_mean", "Entropy Mean"),
        ("final_gen_gap_loss", "Gen Gap"),
        ("late_degradation", "Late Degradation"),
    ]
    return aggregate_rows(summary, metrics)


def table_47(summary):
    columns = [
        "Dataset", "Model", "Method", "Test Acc Std", "Test NLL Std", "ECE Std", "Brier Std",
        "Best Epoch Std", "Epochs Run Std", "Stability Rank",
    ]
    if summary.empty or "method" not in summary.columns:
        empty = pd.DataFrame(columns=columns)
        return empty, empty
    rows = []
    for (dataset, model), group in summary.groupby(["dataset", "model"]):
        raw = []
        for method in ordered_methods(group["method"].dropna().unique()):
            d = group[group["method"] == method]
            row = {"Dataset": dataset, "Model": model, "Method": method_label(method), "_method": method}
            score = 0.0
            for metric, label in [
                ("test_acc", "Test Acc Std"),
                ("test_nll", "Test NLL Std"),
                ("test_ece", "ECE Std"),
                ("test_brier", "Brier Std"),
                ("best_epoch", "Best Epoch Std"),
                ("epochs_run", "Epochs Run Std"),
            ]:
                vals = pd.to_numeric(d[metric], errors="coerce").dropna() if metric in d.columns else pd.Series(dtype=float)
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
                row[label] = std
                if np.isfinite(std):
                    score += std
            row["_score"] = score
            raw.append(row)
        scores = pd.Series({r["_method"]: r["_score"] for r in raw})
        ranks = scores.rank(ascending=True, method="min")
        for r in raw:
            r["Stability Rank"] = int(ranks[r["_method"]]) if r["_method"] in ranks else np.nan
            rows.append(r)
    csv_rows = []
    md_rows = []
    for r in rows:
        csv_row = {k: r[k] for k in columns[:3]}
        md_row = {k: r[k] for k in columns[:3]}
        for label in columns[3:-1]:
            csv_row[label] = fmt_num(r[label])
            md_row[label] = fmt_num(r[label])
        csv_row["Stability Rank"] = r["Stability Rank"]
        md_row["Stability Rank"] = r["Stability Rank"]
        csv_rows.append(csv_row)
        md_rows.append(md_row)
    csv_df = pd.DataFrame(csv_rows, columns=columns)
    md_df = pd.DataFrame(md_rows, columns=columns).astype(object)
    for (dataset, model), idx in md_df.groupby(["Dataset", "Model"]).groups.items():
        block = md_df.loc[idx]
        for col in columns[3:]:
            vals = pd.to_numeric(block[col], errors="coerce")
            if vals.notna().any():
                best_idx = vals.idxmin()
                md_df.loc[best_idx, col] = f"**{md_df.loc[best_idx, col]}**"
    return csv_df, md_df


def write_markdown(df, path, note=True):
    def esc(value):
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|")

    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(esc(row[h]) for h in headers) + " |")
    if note:
        lines.append("")
        lines.append("Best values are bolded within each Dataset/Model block. † marks Ours significantly better than the best baseline at p < 0.05.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="tables_conclusion")
    parser.add_argument("--include-quick", action="store_true", help="Include quick-test experiments in outputs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = filter_quick(collect_histories(args.results_dir), include_quick=args.include_quick)
    summary = filter_quick(collect_summaries(args.results_dir), include_quick=args.include_quick)

    t43 = table_43(hist)
    t43.to_csv(out_dir / "table_4_3_pac_bayes_signal_validity.csv", index=False)

    t44 = table_44(hist)
    t44.to_csv(out_dir / "table_4_4_primal_dual_stability.csv", index=False)

    t45_csv, t45_md = table_45(summary)
    t45_csv.to_csv(out_dir / "table_4_5_lr_scheduling.csv", index=False)
    write_markdown(t45_md, out_dir / "table_4_5_lr_scheduling.md")

    t46_csv, t46_md = table_46(summary)
    t46_csv.to_csv(out_dir / "table_4_6_calibration_reliability.csv", index=False)
    write_markdown(t46_md, out_dir / "table_4_6_calibration_reliability.md")

    t47_csv, t47_md = table_47(summary)
    t47_csv.to_csv(out_dir / "table_4_7_seed_stability.csv", index=False)
    write_markdown(t47_md, out_dir / "table_4_7_seed_stability.md", note=False)

    print(f"Saved conclusion tables to {out_dir}")


if __name__ == "__main__":
    main()
