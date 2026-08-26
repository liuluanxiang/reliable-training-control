"""按预定义实验套件顺序运行配置、统计分析、案例图表与完整性审计。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MAIN_CONFIGS = [
    "configs/cifar10_resnet18_main.json",
    "configs/cifar100_resnet18_main.json",
    "configs/cifar10_resnet50_supplement.json",
    "configs/cifar100_resnet50_supplement.json",
    "configs/cifar10_vgg16_supplement.json",
    "configs/cifar100_vgg16_supplement.json",
]

SUPPLEMENTARY_CIFAR100_CONFIGS = [
    "configs/cifar100_resnet50_supplement.json",
    "configs/cifar100_vgg16_supplement.json",
]

SUITES = {
    "main": {
        "tag_prefix": "main_supplement",
        "configs": MAIN_CONFIGS,
        "audit": ["--suite", "main"],
        "case_postprocess": True,
    },
    "supplementary-cifar100": {
        "tag_prefix": "supplementary_cifar100",
        "configs": SUPPLEMENTARY_CIFAR100_CONFIGS,
        "audit": [
            "--config",
            "configs/cifar100_resnet50_supplement.json",
            "--config",
            "configs/cifar100_vgg16_supplement.json",
        ],
        "case_postprocess": True,
    },
    "ablation": {
        "tag_prefix": "ablation",
        "configs": ["configs/ablation_cifar10_resnet18.json"],
        "audit": ["--suite", "ablation"],
        "case_postprocess": False,
    },
    "sensitivity": {
        "tag_prefix": "sensitivity",
        "configs_glob": "configs/sensitivity/*.json",
        "audit": ["--suite", "sensitivity"],
        "case_postprocess": False,
    },
}


def run_command(cmd: list[str], dry_run: bool) -> None:
    """记录并执行子命令；dry-run 模式只打印计划。"""

    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def postprocess(
    python: str,
    results_dir: Path,
    run_tag: str,
    audit_args: list[str],
    include_case_outputs: bool,
    dry_run: bool,
) -> None:
    """对训练结果生成表格/图形，并执行结果协议审计。"""

    tables_dir = Path("tables") / run_tag
    figures_dir = Path("figures") / run_tag
    statistical_tables_dir = Path("statistical_tables") / run_tag
    statistical_figures_dir = Path("statistical_figures") / run_tag

    if include_case_outputs:
        run_command([python, "scripts/generate_case_study_tables.py", "--results-dir", str(results_dir), "--out-dir", str(tables_dir)], dry_run)
        run_command([python, "scripts/make_case_study_figures.py", "--results-dir", str(results_dir), "--out-dir", str(figures_dir)], dry_run)

    run_command([python, "generate_statistical_tables.py", "--results-dir", str(results_dir), "--out-dir", str(statistical_tables_dir)], dry_run)
    run_command([python, "make_statistical_figures.py", "--results-dir", str(results_dir), "--out-dir", str(statistical_figures_dir)], dry_run)

    audit_cmd = [
        python,
        "audit_experiment_results.py",
        "--results-dir",
        str(results_dir),
        *audit_args,
        "--check-statistical",
        "--statistical-tables-dir",
        str(statistical_tables_dir),
        "--statistical-figures-dir",
        str(statistical_figures_dir),
    ]
    run_command(audit_cmd, dry_run)


def resolve_configs(spec: dict[str, object]) -> list[str]:
    """展开套件配置并检查每个配置文件都存在。"""

    if "configs" in spec:
        return list(spec["configs"])  # type: ignore[arg-type]
    pattern = str(spec["configs_glob"])
    return [str(path.relative_to(ROOT)) for path in sorted(ROOT.glob(pattern))]


def main() -> None:
    """解析套件、结果策略与后处理选项并启动批次。"""

    parser = argparse.ArgumentParser(description="Run experiment batches without shell-specific launcher scripts.")
    parser.add_argument("suite", choices=sorted(SUITES), help="Batch suite to run.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for child commands.")
    parser.add_argument("--results-root", default="results", help="Root directory for batch outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    spec = SUITES[args.suite]
    run_tag = f"{spec['tag_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(args.results_root) / run_tag
    progress_log = Path(args.results_root) / f"progress_{run_tag}.log"
    experiment_suffix = f"__{run_tag}"

    progress_state = Path(args.results_root) / "progress_state.json"
    dashboard_path = Path(args.results_root) / "progress_dashboard.html"

    for config in resolve_configs(spec):
        run_command(
            [
                args.python,
                "run_experiment.py",
                "--config",
                config,
                "--results-dir",
                str(results_dir),
                "--experiment-suffix",
                experiment_suffix,
                "--existing",
                "error",
                "--progress",
                "batch",
                "--progress-log",
                str(progress_log),
                "--progress-state",
                str(progress_state),
                "--dashboard-path",
                str(dashboard_path),
            ],
            args.dry_run,
        )

    postprocess(
        args.python,
        results_dir,
        run_tag,
        list(spec["audit"]),  # type: ignore[arg-type]
        bool(spec.get("case_postprocess")),
        args.dry_run,
    )


if __name__ == "__main__":
    main()
