"""汇总大修实验并生成共享协议、物理运行与成对检验结果。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.stats_utils import (  # noqa: E402
    bootstrap_ci, cohens_d, holm_bonferroni, paired_ttest, wilcoxon_test,
)


REVISION = ROOT / "revision_experiments"
RESULTS = REVISION / "results"
STATISTICS = RESULTS / "statistics"
METHODS = ["step", "cosine", "plateau", "reliability_val_loss_only", "reliability"]
BASELINES = METHODS[:-1]
METRICS = [
    "test_acc", "test_nll", "test_ece", "test_brier", "test_aurc",
    "final_gen_gap_loss", "epochs_run", "runtime_seconds",
]
HIGHER_BETTER = {"test_acc"}
REQUIRED_COLUMNS = [
    "experiment", "experiment_type", "dataset", "model", "method", "seed", "budget_mode",
    "status", "data_protocol_version", "prior_sigma", *METRICS,
]
STAT_COLUMNS = [
    "experiment", "dataset", "model", "metric", "higher_is_better", "baseline", "n_pairs",
    "ours_mean", "baseline_mean", "mean_difference_ours_minus_baseline", "ci95_low", "ci95_high",
    "sd_difference", "cohens_dz", "analysis_scope", "paired_t_p_raw", "wilcoxon_p_raw",
    "paired_t_p_holm", "wilcoxon_p_holm",
]


def read_json(path):
    """读取单个运行的 JSON 摘要。"""

    return json.loads(path.read_text(encoding="utf-8"))


def collect_results():
    """从统一结果树读取完成运行，并按科学哈希去重。"""

    rows = []
    seen_hashes = set()
    for root in [ROOT / "results_v2", RESULTS]:
        for path in root.rglob("summary.json"):
            if any(part in {"_archive", "smoke"} for part in path.parts):
                continue
            try:
                row = read_json(path)
            except Exception:
                continue
            config_path = path.with_name("config_used.json")
            config = read_json(config_path) if config_path.exists() else {}
            protocol = row.get("data_protocol_version", config.get("data_protocol_version"))
            scientific_hash = row.get("scientific_config_hash")
            if (
                protocol != "deterministic_validation_v2"
                or row.get("status") != "COMPLETED"
                or not scientific_hash
                or scientific_hash in seen_hashes
            ):
                continue
            seen_hashes.add(scientific_hash)
            controller = config.get("controller", {})
            row.update({
                "summary_path": str(path), "analysis_source": "physical_run",
                "prior_sigma": controller.get("prior_sigma"),
                "lambda_g": controller.get("lambda_g"), "lambda_u": controller.get("lambda_u"),
                "data_protocol_version": protocol,
            })
            rows.append(row)
    frame = pd.DataFrame(rows)
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame


def shared_analysis_rows(physical):
    """将可共享的物理运行映射到预注册分析任务，同时保留来源。"""

    if physical.empty:
        return physical.copy()
    main = physical[
        (physical["experiment_type"] == "revision_main")
        & (physical["dataset"] == "cifar10")
        & (physical["model"] == "resnet18")
    ].copy()
    shared = []

    pd_full = main[main["method"] == "reliability"].copy()
    pd_full["experiment"] = "revision_pd_component_ablation"
    pd_full["experiment_type"] = "pd_component_ablation"
    pd_full["method"] = "reliability_pd_full"
    shared.append(pd_full)

    budget = main[main["method"].isin(METHODS)].copy()
    budget["experiment"] = "revision_budget_adaptive_cifar10_resnet18"
    budget["experiment_type"] = "budget_comparison"
    budget["budget_mode"] = "adaptive"
    shared.append(budget)

    sigma = main[main["method"] == "reliability"].copy()
    sigma["experiment"] = "revision_sigma_1p0"
    sigma["experiment_type"] = "sigma_sensitivity"
    sigma["prior_sigma"] = 0.05
    shared.append(sigma)

    shared = [frame for frame in shared if not frame.empty]
    if not shared:
        return physical.iloc[0:0].copy()
    result = pd.concat(shared, ignore_index=True)
    result["analysis_source"] = "shared_corrected_main_run"
    return result


def aggregate(frame, group_columns, path):
    """按指定实验维度汇总均值、标准差与样本数。"""

    metric_columns = [metric for metric in METRICS if metric in frame.columns]
    columns = [*group_columns, "seeds"]
    if frame.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    numeric = frame.copy()
    for metric in metric_columns:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    rows = []
    for keys, group in numeric.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["seeds"] = group["seed"].nunique()
        for metric in metric_columns:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def paired_statistics(frame):
    """按共同种子比较本方法与各基线，并执行多重检验校正。"""

    eligible_types = {"revision_main", "external_tiny_imagenet", "external_architecture"}
    data = frame[frame["experiment_type"].isin(eligible_types)].copy()
    rows = []
    group_columns = ["experiment", "dataset", "model"]
    for keys, group in data.groupby(group_columns, dropna=False):
        ours = group[group["method"] == "reliability"]
        if ours.empty:
            continue
        for baseline_name in BASELINES:
            baseline = group[group["method"] == baseline_name]
            for metric in METRICS:
                if metric not in group.columns:
                    continue
                paired = ours[["seed", metric]].merge(
                    baseline[["seed", metric]], on="seed", suffixes=("_ours", "_baseline")
                ).dropna()
                if paired.empty:
                    continue
                ours_values = paired[f"{metric}_ours"].to_numpy(dtype=float)
                baseline_values = paired[f"{metric}_baseline"].to_numpy(dtype=float)
                differences = ours_values - baseline_values
                ci_low, ci_high = bootstrap_ci(differences, n_boot=10000, seed=123)
                rows.append({
                    **dict(zip(group_columns, keys)), "metric": metric,
                    "higher_is_better": metric in HIGHER_BETTER, "baseline": baseline_name,
                    "n_pairs": len(paired), "ours_mean": ours_values.mean(),
                    "baseline_mean": baseline_values.mean(),
                    "mean_difference_ours_minus_baseline": differences.mean(),
                    "ci95_low": ci_low, "ci95_high": ci_high,
                    "sd_difference": differences.std(ddof=1) if len(differences) > 1 else np.nan,
                    "cohens_dz": cohens_d(ours_values, baseline_values),
                    "analysis_scope": "headline_5_seed" if len(paired) == 5 else "incomplete_supporting",
                    "paired_t_p_raw": paired_ttest(ours_values, baseline_values),
                    "wilcoxon_p_raw": wilcoxon_test(ours_values, baseline_values),
                })
    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    stats["paired_t_p_holm"] = np.nan
    stats["wilcoxon_p_holm"] = np.nan
    for _, indices in stats.groupby([*group_columns, "metric"], dropna=False).groups.items():
        indices = list(indices)
        stats.loc[indices, "paired_t_p_holm"] = holm_bonferroni(stats.loc[indices, "paired_t_p_raw"])
        stats.loc[indices, "wilcoxon_p_holm"] = holm_bonferroni(stats.loc[indices, "wilcoxon_p_raw"])
    return stats


def write_outputs(physical):
    """写出运行清单、聚合表、统计检验和缺失项报告。"""

    RESULTS.mkdir(parents=True, exist_ok=True)
    STATISTICS.mkdir(parents=True, exist_ok=True)
    shared = shared_analysis_rows(physical)
    analysis = pd.concat([physical, shared], ignore_index=True) if not shared.empty else physical
    physical.to_csv(RESULTS / "master_summary_protocol_v2.csv", index=False)
    analysis.to_csv(RESULTS / "analysis_inventory.csv", index=False)

    specs = [
        ("pd_component_ablation", "pd_component_ablation", ["method"], "pd_runs.csv", "pd_summary.csv"),
        ("sigma_sensitivity", "sigma_sensitivity", ["prior_sigma"], "sigma_runs.csv", "sigma_summary.csv"),
        ("budget_comparison", "budget_comparison", ["budget_mode", "method"], "budget_runs.csv", "budget_summary.csv"),
    ]
    for directory, experiment_type, groups, runs_name, summary_name in specs:
        output = RESULTS / directory
        output.mkdir(parents=True, exist_ok=True)
        selected = analysis[analysis["experiment_type"] == experiment_type]
        selected.to_csv(output / runs_name, index=False)
        aggregate(selected, groups, output / summary_name)

    external = analysis[analysis["experiment_type"].isin({"external_tiny_imagenet", "external_architecture"})]
    external_dir = RESULTS / "external_validation"
    external_dir.mkdir(parents=True, exist_ok=True)
    external.to_csv(external_dir / "external_validation_runs.csv", index=False)
    aggregate(external, ["dataset", "model", "method"], external_dir / "external_validation_summary.csv")

    stats = paired_statistics(analysis)
    for column in STAT_COLUMNS:
        if column not in stats.columns:
            stats[column] = pd.Series(dtype="object")
    stats = stats[STAT_COLUMNS]
    stats.to_csv(STATISTICS / "paired_results_raw.csv", index=False)
    if stats.empty:
        empty = pd.DataFrame(columns=STAT_COLUMNS)
        for name in ["paired_results_holm.csv", "effect_sizes.csv", "confidence_intervals.csv", "consistency_summary.csv"]:
            empty.to_csv(STATISTICS / name, index=False)
    else:
        stats.to_csv(STATISTICS / "paired_results_holm.csv", index=False)
        stats[["experiment", "dataset", "model", "metric", "baseline", "n_pairs",
               "mean_difference_ours_minus_baseline", "sd_difference", "cohens_dz", "analysis_scope"]].to_csv(
            STATISTICS / "effect_sizes.csv", index=False
        )
        stats[["experiment", "dataset", "model", "metric", "baseline", "n_pairs",
               "mean_difference_ours_minus_baseline", "ci95_low", "ci95_high"]].to_csv(
            STATISTICS / "confidence_intervals.csv", index=False
        )
        consistency = stats.assign(
            favorable=np.where(stats["higher_is_better"],
                               stats["mean_difference_ours_minus_baseline"] > 0,
                               stats["mean_difference_ours_minus_baseline"] < 0)
        ).groupby(["metric", "baseline"], as_index=False).agg(
            comparisons=("favorable", "size"), favorable_comparisons=("favorable", "sum"),
            significant_holm=("paired_t_p_holm", lambda values: int((values < 0.05).sum())),
        )
        consistency.to_csv(STATISTICS / "consistency_summary.csv", index=False)

    completed = len(physical)
    matched = int((stats["n_pairs"] == 5).sum()) if not stats.empty else 0
    (STATISTICS / "STATISTICAL_REPORT.md").write_text(
        "# Statistical Report\n\n"
        f"- Corrected-protocol physical runs found: {completed}/260.\n"
        f"- Fully matched five-seed metric comparisons available: {matched}.\n"
        "- Difference orientation: proposed minus baseline.\n"
        "- Effect size: paired Cohen's d_z = mean(pair differences) / SD(pair differences).\n"
        "- Confidence intervals: deterministic 10,000-resample paired bootstrap.\n"
        "- Holm families: four proposed-vs-baseline comparisons within each experiment setting and metric.\n",
        encoding="utf-8",
    )
    return completed, len(analysis)


def main():
    """校验结果协议后生成完整大修分析产物。"""

    parser = argparse.ArgumentParser(description="Aggregate and statistically analyze protocol-v2 revision runs.")
    parser.parse_args()
    completed, analysis_rows = write_outputs(collect_results())
    print(f"Corrected-protocol physical summaries: {completed}/260")
    print(f"Analysis rows after documented main-run sharing: {analysis_rows}")


if __name__ == "__main__":
    main()
