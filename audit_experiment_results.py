"""审计每个计划运行的产物、摘要字段和论文统计输出完整性。"""

import argparse
import csv
import json
import sys
from pathlib import Path


RESULT_ARTIFACTS = [
    "history.csv",
    "summary.json",
    "test_logits.pt",
    "test_targets.pt",
    "calibration_bins.csv",
    "risk_coverage_curve.csv",
]

REQUIRED_SUMMARY_FIELDS = [
    "experiment",
    "experiment_type",
    "dataset",
    "model",
    "method",
    "seed",
    "epochs_run",
    "best_epoch",
    "best_val_acc",
    "final_val_acc",
    "test_acc",
    "test_top5_acc",
    "test_nll",
    "test_ece",
    "test_mce",
    "test_brier",
    "test_aurc",
    "test_eaurc",
    "final_gen_gap_loss",
    "late_degradation",
    "aulc_val_acc",
    "runtime_seconds",
]

MAIN_SUPPLEMENT_CONFIGS = [
    "configs/cifar10_resnet18_main.json",
    "configs/cifar100_resnet18_main.json",
    "configs/cifar10_resnet50_supplement.json",
    "configs/cifar100_resnet50_supplement.json",
    "configs/cifar10_vgg16_supplement.json",
    "configs/cifar100_vgg16_supplement.json",
]

ABLATION_CONFIGS = [
    "configs/ablation_cifar10_resnet18.json",
]

STATISTICAL_TABLE_FILES = [
    "table_4_8_ablation_study.csv",
    "table_4_8_ablation_study.md",
    "table_4_9_sensitivity_alpha_beta.csv",
    "table_4_9_sensitivity_alpha_beta.md",
    "table_4_10_statistical_tests.csv",
    "table_4_10_statistical_tests.md",
]

STATISTICAL_FIGURE_FILES = [
    "fig_4_11_ablation_study.pdf",
    "fig_4_11_ablation_study.png",
    "fig_4_12_sensitivity_analysis.pdf",
    "fig_4_12_sensitivity_analysis.png",
    "fig_4_13_statistical_tests.pdf",
    "fig_4_13_statistical_tests.png",
]


def read_json(path):
    """读取单个 JSON 产物。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_paths_for_suite(suites):
    """将审计套件名称展开为去重后的配置路径。"""

    paths = []
    selected = set(suites)
    if "all" in selected or "main" in selected:
        paths.extend(MAIN_SUPPLEMENT_CONFIGS)
    if "all" in selected or "ablation" in selected:
        paths.extend(ABLATION_CONFIGS)
    if "all" in selected or "sensitivity" in selected:
        paths.extend(str(path) for path in sorted(Path("configs/sensitivity").glob("*.json")))
    return paths


def planned_runs(config_paths, results_dir, experiment_prefix, experiment_suffix):
    """从配置构造期望存在的 ``method x seed`` 运行清单。"""

    runs = []
    for config_path in config_paths:
        cfg = read_json(config_path)
        experiment = f"{experiment_prefix}{cfg['experiment_name']}{experiment_suffix}"
        for method in cfg["methods"]:
            for seed in cfg["seeds"]:
                runs.append(
                    {
                        "config": str(config_path),
                        "experiment": experiment,
                        "experiment_type": str(cfg.get("experiment_type", "")),
                        "dataset": cfg["dataset"],
                        "model": cfg["model"],
                        "method": str(method),
                        "seed": int(seed),
                        "path": Path(results_dir) / experiment / str(method) / f"seed_{int(seed)}",
                    }
                )
    return runs


def nonempty(path):
    """检查文件存在且至少含一个字节。"""

    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def validate_summary(run, require_artifacts=True):
    """验证运行摘要字段、状态、数值范围和配套产物。"""

    issues = []
    out_dir = run["path"]
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        issues.append(("missing_summary", summary_path, "summary.json is missing"))
        return issues

    try:
        summary = read_json(summary_path)
    except Exception as exc:
        issues.append(("invalid_summary_json", summary_path, str(exc)))
        return issues

    for field in REQUIRED_SUMMARY_FIELDS:
        if field not in summary:
            issues.append(("missing_summary_field", summary_path, field))

    expected = {
        "experiment": run["experiment"],
        "experiment_type": run["experiment_type"],
        "dataset": run["dataset"],
        "model": run["model"],
        "method": run["method"],
        "seed": run["seed"],
    }
    for field, value in expected.items():
        if field in summary and summary[field] != value:
            issues.append(("summary_field_mismatch", summary_path, f"{field}: expected {value}, got {summary[field]}"))

    numeric_positive = ["epochs_run", "runtime_seconds"]
    for field in numeric_positive:
        value = summary.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
            issues.append(("invalid_summary_value", summary_path, f"{field}: {value}"))

    if require_artifacts:
        for artifact in RESULT_ARTIFACTS:
            artifact_path = out_dir / artifact
            if not nonempty(artifact_path):
                issues.append(("missing_or_empty_artifact", artifact_path, artifact))

    return issues


def read_master_keys(master_path):
    """读取主汇总表中的实验唯一键。"""

    keys = set()
    if not Path(master_path).exists():
        return keys
    with Path(master_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                seed = int(row.get("seed", ""))
            except ValueError:
                continue
            keys.add((row.get("experiment", ""), row.get("method", ""), seed))
    return keys


def validate_master(runs, results_dir):
    """确认所有有效逐运行摘要都已进入主汇总表。"""

    issues = []
    master_path = Path(results_dir) / "master_summary.csv"
    if not nonempty(master_path):
        return [("missing_master_summary", master_path, "master_summary.csv is missing or empty")]

    keys = read_master_keys(master_path)
    for run in runs:
        key = (run["experiment"], run["method"], run["seed"])
        if key not in keys:
            issues.append(("missing_master_row", master_path, f"{key[0]} | {key[1]} | seed={key[2]}"))
    return issues


def csv_data_rows(path):
    """返回 CSV 的数据行数，不计表头。"""

    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    return max(0, len(rows) - 1)


def validate_statistical_outputs(tables_dir, figures_dir, suites):
    """检查所选套件要求的表格、图片和图注是否非空。"""

    issues = []
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)

    for name in STATISTICAL_TABLE_FILES:
        path = tables_dir / name
        if not nonempty(path):
            issues.append(("missing_or_empty_statistical_table", path, name))

    for name in STATISTICAL_FIGURE_FILES:
        path = figures_dir / name
        if not nonempty(path):
            issues.append(("missing_or_empty_statistical_figure", path, name))

    selected = set(suites)
    if "all" in selected or "ablation" in selected:
        path = tables_dir / "table_4_8_ablation_study.csv"
        if path.exists() and csv_data_rows(path) == 0:
            issues.append(("empty_ablation_statistical_table", path, "no ablation rows"))
    if "all" in selected or "sensitivity" in selected:
        path = tables_dir / "table_4_9_sensitivity_alpha_beta.csv"
        if path.exists() and csv_data_rows(path) == 0:
            issues.append(("empty_sensitivity_statistical_table", path, "no sensitivity rows"))
    if "all" in selected or "main" in selected:
        path = tables_dir / "table_4_10_statistical_tests.csv"
        if path.exists() and csv_data_rows(path) == 0:
            issues.append(("empty_statistical_tests_table", path, "no statistical test rows"))

    return issues


def write_reports(report_dir, runs, issues):
    """写出机器可读 CSV 与人工可读 Markdown 审计报告。"""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    serializable_runs = []
    for run in runs:
        row = dict(run)
        row["path"] = str(row["path"])
        serializable_runs.append(row)

    serializable_issues = [
        {"kind": kind, "path": str(path), "detail": detail}
        for kind, path, detail in issues
    ]
    report = {
        "planned_runs": len(runs),
        "issues": len(issues),
        "runs": serializable_runs,
        "issue_rows": serializable_issues,
    }
    (report_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (report_dir / "audit_issues.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kind", "path", "detail"])
        writer.writeheader()
        writer.writerows(serializable_issues)


def has_valid_summary(run):
    """快速判断运行是否拥有可解析且标记完成的摘要。"""

    try:
        read_json(run["path"] / "summary.json")
        return True
    except Exception:
        return False


def main():
    """解析审计范围，执行全部验证并用退出码表示成功或失败。"""

    parser = argparse.ArgumentParser(description="Audit experiment result completeness.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--suite",
        action="append",
        choices=["all", "main", "ablation", "sensitivity"],
        default=[],
        help="Expected config group. Repeatable. Defaults to main.",
    )
    parser.add_argument("--config", action="append", default=[], help="Additional config path to audit.")
    parser.add_argument("--experiment-prefix", default="")
    parser.add_argument(
        "--experiment-suffix",
        default=None,
        help="Defaults to __<results-dir-name>, matching the project run scripts.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Do not require non-summary artifacts.")
    parser.add_argument("--no-check-master", action="store_true", help="Do not require master_summary.csv rows.")
    parser.add_argument("--check-statistical", action="store_true")
    parser.add_argument("--statistical-tables-dir", default="")
    parser.add_argument("--statistical-figures-dir", default="")
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--warn-only", action="store_true", help="Print issues but exit with code 0.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_tag = results_dir.name
    suites = args.suite if args.suite else ([] if args.config else ["main"])
    experiment_suffix = args.experiment_suffix
    if experiment_suffix is None:
        experiment_suffix = f"__{run_tag}"

    config_paths = config_paths_for_suite(suites) + args.config
    runs = planned_runs(config_paths, results_dir, args.experiment_prefix, experiment_suffix)

    issues = []
    for run in runs:
        issues.extend(validate_summary(run, require_artifacts=not args.summary_only))

    if not args.no_check_master:
        issues.extend(validate_master(runs, results_dir))

    if args.check_statistical:
        tables_dir = args.statistical_tables_dir or str(Path("statistical_tables") / run_tag)
        figures_dir = args.statistical_figures_dir or str(Path("statistical_figures") / run_tag)
        issues.extend(validate_statistical_outputs(tables_dir, figures_dir, suites))

    report_dir = args.report_dir or str(results_dir / "audit")
    write_reports(report_dir, runs, issues)

    completed = sum(1 for run in runs if has_valid_summary(run))
    print(f"Audit results dir: {results_dir}")
    print(f"Expected runs: {len(runs)}")
    print(f"Runs with summary.json present: {completed}/{len(runs)}")
    print(f"Issues: {len(issues)}")
    print(f"Report: {Path(report_dir) / 'audit_report.json'}")
    print(f"Issues CSV: {Path(report_dir) / 'audit_issues.csv'}")

    if issues:
        print("")
        print("First issues:")
        for kind, path, detail in issues[:25]:
            print(f"- {kind}: {path} ({detail})")
        if len(issues) > 25:
            print(f"- ... {len(issues) - 25} more")

    if issues and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
