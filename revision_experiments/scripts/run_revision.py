"""构建并执行大修补充实验队列，持续写入队列清单与实时状态。"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision_experiments"
PYTHON = Path(sys.executable)
RESULTS = REVISION / "results"
CURRENT_STATE = RESULTS / "current_run_state.json"
CURRENT_DASHBOARD = RESULTS / "current_run_dashboard.html"
CURRENT_LOG = RESULTS / "current_run.log"


def read_json(path):
    """读取 UTF-8 JSON 配置或状态文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def main_runs():
    """根据仓库审计缺口生成主实验补跑任务。"""

    missing_path = REVISION / "audit" / "missing_main_runs.csv"
    if not missing_path.exists():
        raise RuntimeError("Run audit_repository.py before planning the revision queue.")
    with missing_path.open("r", encoding="utf-8", newline="") as handle:
        missing = list(csv.DictReader(handle))
    configs = {}
    for path in (REVISION / "configs" / "main_5seed").glob("*.json"):
        cfg = read_json(path)
        configs[(cfg["dataset"], cfg["model"])] = path
    return [{"group": "main", "config": configs[(row["dataset"], row["model"])],
             "method": row["method"], "seed": int(row["seed"])} for row in missing]


def pd_runs():
    """生成 PD 组成项消融任务。"""

    path = REVISION / "configs" / "pd_ablation" / "cifar10_resnet18.json"
    cfg = read_json(path)
    return [{"group": "pd", "config": path, "method": method, "seed": seed}
            for method in cfg["methods"] for seed in cfg["seeds"]
            if method != "reliability_pd_full"]


def sigma_runs():
    """生成先验尺度 sigma 敏感性任务。"""

    runs = []
    for path in sorted((REVISION / "configs" / "sigma_sensitivity").glob("*.json")):
        cfg = read_json(path)
        is_default = float(cfg["controller"]["prior_sigma"]) == 0.05
        runs.extend({"group": "sigma", "config": path, "method": "reliability", "seed": seed}
                    for seed in cfg["seeds"] if not is_default)
    return runs


def budget_runs():
    """生成固定预算公平对比任务。"""

    runs = []
    for path in sorted((REVISION / "configs" / "budget_comparison").glob("*.json")):
        cfg = read_json(path)
        for method in cfg["methods"]:
            for seed in cfg["seeds"]:
                main_matrix_adaptive = cfg["budget_mode"] == "adaptive"
                if not main_matrix_adaptive:
                    runs.append({"group": "budget", "config": path, "method": method, "seed": seed})
    return runs


def external_runs(group):
    """生成 Tiny ImageNet 或 DeiT 外部有效性任务。"""

    path = REVISION / "configs" / group / "full.json"
    cfg = read_json(path)
    return [{"group": group, "config": path, "method": method, "seed": seed}
            for method in cfg["methods"] for seed in cfg["seeds"]]


def plan(selected):
    """按用户选择拼接任务组，并分配全局队列序号。"""

    builders = {
        "main": main_runs, "pd": pd_runs, "sigma": sigma_runs, "budget": budget_runs,
        "tiny_imagenet": lambda: external_runs("tiny_imagenet"),
        "deit_cifar100": lambda: external_runs("deit_cifar100"),
    }
    rows = []
    for group in selected:
        rows.extend(builders[group]())
    for index, row in enumerate(rows, start=1):
        cfg = read_json(row["config"])
        row.update({"run_index": index, "dataset": cfg["dataset"], "model": cfg["model"],
                    "budget_mode": cfg.get("budget_mode", "adaptive"), "max_epochs": cfg["epochs"],
                    "status": "PENDING"})
    return rows


def write_manifest(rows):
    """把本次队列写成 CSV，供仪表盘和完成时间估算读取。"""

    path = REVISION / "results" / "run_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_index", "group", "dataset", "model", "method", "seed", "budget_mode",
              "max_epochs", "config", "status", "started_at", "completed_at", "return_code"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def execute(rows, existing, fail_fast=False):
    """逐任务调用主实验入口，并把当前任务状态转发给仪表盘。"""

    scheduled = REVISION / "configs" / "_scheduled"
    scheduled.mkdir(parents=True, exist_ok=True)
    failures = 0
    for row in rows:
        source = read_json(row["config"])
        source["methods"] = [row["method"]]
        source["seeds"] = [row["seed"]]
        config_name = f"{row['group']}_{source['dataset']}_{source['model']}_{row['method']}_seed{row['seed']}"
        config_path = scheduled / f"{config_name}.json"
        config_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
        command = [str(PYTHON), str(ROOT / "run_experiment.py"), "--config", str(config_path),
                   "--existing", existing, "--progress", "batch",
                   "--progress-interval", "20", "--progress-seconds", "5",
                   "--progress-state", str(CURRENT_STATE),
                   "--dashboard-path", str(CURRENT_DASHBOARD),
                   "--progress-log", str(CURRENT_LOG)]
        print("Running:", " ".join(command), flush=True)
        row["status"] = "RUNNING"
        row["started_at"] = datetime.now().astimezone().isoformat()
        write_manifest(rows)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        row["return_code"] = completed.returncode
        row["completed_at"] = datetime.now().astimezone().isoformat()
        row["status"] = "COMPLETED" if completed.returncode == 0 else "FAILED"
        write_manifest(rows)
        subprocess.run(
            [str(PYTHON), str(REVISION / "scripts" / "estimate_runtime.py")],
            cwd=ROOT, check=False,
        )
        if completed.returncode != 0:
            failures += 1
            if fail_fast:
                break
    if failures:
        raise RuntimeError(f"Revision queue completed with {failures} failed run(s); see run_manifest.csv.")


def main():
    """解析任务组与已有结果策略，然后执行或仅打印队列。"""

    parser = argparse.ArgumentParser(description="Plan or execute corrected-protocol revision runs.")
    parser.add_argument("--group", action="append", choices=[
        "main", "pd", "sigma", "budget", "tiny_imagenet", "deit_cifar100"
    ])
    parser.add_argument("--execute", action="store_true", help="Launch sequential GPU runs; default is plan-only.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed run.")
    parser.add_argument("--existing", choices=["error", "skip", "rerun-partial"], default="rerun-partial")
    args = parser.parse_args()
    selected = args.group or ["main", "pd", "budget", "sigma", "tiny_imagenet", "deit_cifar100"]
    rows = plan(selected)
    manifest = write_manifest(rows)
    print(f"Planned missing runs: {len(rows)}")
    print(f"Manifest: {manifest}")
    if args.execute:
        execute(rows, args.existing, fail_fast=args.fail_fast)
    else:
        print("Plan only. Run smoke tests and runtime benchmarks before passing --execute.")


if __name__ == "__main__":
    main()
