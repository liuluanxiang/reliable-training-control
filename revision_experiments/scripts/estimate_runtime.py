"""结合实测吞吐、历史停止轮数和已完成任务估算补充实验工期。"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision_experiments"
BENCHMARKS = REVISION / "results" / "runtime_benchmarks.csv"
BENCHMARK_HISTORY = REVISION / "results" / "runtime_benchmark_history.csv"
OUTPUT = REVISION / "results" / "runtime_estimate.csv"
GPU_NAME = "NVIDIA GeForce RTX 5070 Ti"


def read_csv(path):
    """容错读取 CSV；文件不存在时返回空列表。"""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    """读取 UTF-8 JSON 结果文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def historical_rows():
    """从 ``results_v2`` 收集可用于校准的正式历史运行。"""

    rows = []
    for path in (ROOT / "results_v2").rglob("summary.json"):
        try:
            row = read_json(path)
        except Exception:
            continue
        if row.get("experiment_type") in {"main", "supplement", "ablation"}:
            rows.append(row)
    return rows


def measured_throughput(history):
    """从基准历史中提取各模型族的稳健每 epoch 耗时。"""

    values = {}
    for row in read_csv(BENCHMARKS):
        try:
            seconds = float(row["seconds_per_epoch"])
        except (ValueError, TypeError):
            continue
        try:
            wall = float(row["benchmark_wall_clock_seconds"])
            warmup = max(0.0, wall - seconds * int(row["timed_epochs"]))
        except (ValueError, TypeError, KeyError):
            warmup = seconds
        values[row["family"]] = (seconds, warmup, "current_5_epoch_benchmark")

    aliases = {
        ("cifar10", "resnet18"): "cifar10_resnet18",
        ("cifar100", "resnet18"): "cifar10_resnet18",
        ("cifar10", "resnet50"): "cifar100_resnet50",
        ("cifar100", "resnet50"): "cifar100_resnet50",
        ("cifar10", "vgg16"): "cifar100_vgg16",
        ("cifar100", "vgg16"): "cifar100_vgg16",
        ("tiny_imagenet", "resnet18"): "tiny_imagenet_resnet18",
        ("cifar100", "deit_tiny_patch16_224"): "cifar100_deit_tiny",
    }
    result = {}
    for setting, family in aliases.items():
        if family in values:
            result[setting] = values[family]
            continue
        samples = [float(row["runtime_seconds"]) / float(row["epochs_run"]) for row in history
                   if (row.get("dataset"), row.get("model")) == setting and row.get("epochs_run", 0)]
        if samples:
            seconds = sum(samples) / len(samples)
            result[setting] = (seconds, seconds, "historical_same_gpu_fallback")
    return result


def mean_stopping(history, dataset, model, method, default_epochs):
    """从匹配历史运行估计平均停止轮数。"""

    samples = [float(row["epochs_run"]) for row in history
               if row.get("dataset") == dataset and row.get("model") == model and row.get("method") == method]
    return sum(samples) / len(samples) if samples else float(default_epochs)


def corrected_completed_rows():
    """汇总完成运行并按实验唯一键去重。"""

    rows = []
    seen_hashes = set()
    for root in [ROOT / "results_v2", REVISION / "results"]:
        for path in root.rglob("summary.json"):
            if any(part in {"_archive", "smoke"} for part in path.parts):
                continue
            try:
                row = read_json(path)
                config_path = path.with_name("config_used.json")
                config = read_json(config_path) if config_path.exists() else {}
            except Exception:
                continue
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
            row["_prior_sigma"] = config.get("controller", {}).get("prior_sigma")
            rows.append(row)
    return rows


def unique_count(rows, predicate, key_fields=("method", "seed")):
    """统计满足条件的唯一运行键数量。"""

    return len({tuple(row.get(field) for field in key_fields) for row in rows if predicate(row)})


def estimate_group(name, final_slots, shared_slots, total_runs, completed_runs, throughput_sample, epochs):
    """估算单任务组的剩余 GPU 小时与墙钟时间。"""

    seconds_epoch = throughput_sample[0] if throughput_sample else None
    warmup_seconds = throughput_sample[1] if throughput_sample else None
    remaining_runs = max(0, total_runs - completed_runs)
    if seconds_epoch is None:
        hours = "UNRESOLVED"
        mean = "UNRESOLVED"
        status = "AWAITING_BENCHMARK"
    else:
        mean = (warmup_seconds + seconds_epoch * max(0, epochs - 1)) / 3600.0
        hours = remaining_runs * mean
        status = "CONSERVATIVE_MAX_EPOCH_BUDGET"
    return {
        "experiment_group": name, "final_analysis_slots": final_slots,
        "shared_from_main_slots": shared_slots, "total_physical_runs": total_runs,
        "completed_physical_runs": completed_runs, "remaining_physical_runs": remaining_runs,
        "max_epochs": epochs, "mean_hours_per_run": mean,
        "remaining_gpu_hours": hours, "eta_status": status,
    }


def main():
    """重新计算各新增实验组与总队列的预计完成时间。"""

    now = datetime.now().astimezone()
    history = historical_rows()
    throughput = measured_throughput(history)
    missing_main = read_csv(REVISION / "audit" / "missing_main_runs.csv")
    completed = corrected_completed_rows()
    completed_main_keys = {
        (row.get("dataset"), row.get("model"), row.get("method"), row.get("seed"))
        for row in completed if row.get("experiment_type") == "revision_main"
    }
    pending_main = [row for row in missing_main if (
        row["dataset"], row["model"], row["method"], int(row["seed"])
    ) not in completed_main_keys]

    main_hours = 0.0
    main_resolved = True
    for row in pending_main:
        measured = throughput.get((row["dataset"], row["model"]))
        if measured is None:
            main_resolved = False
            continue
        main_hours += (measured[1] + measured[0] * 119) / 3600.0
    estimates = [{
        "experiment_group": "Corrected main matrix", "final_analysis_slots": 150,
        "shared_from_main_slots": 0, "total_physical_runs": 150,
        "completed_physical_runs": len(completed_main_keys),
        "remaining_physical_runs": len(pending_main), "max_epochs": 120,
        "mean_hours_per_run": main_hours / max(1, len(pending_main)) if main_resolved else "UNRESOLVED",
        "remaining_gpu_hours": main_hours if main_resolved else "UNRESOLVED",
        "eta_status": "CONSERVATIVE_MAX_EPOCH_BUDGET" if main_resolved else "AWAITING_BENCHMARK",
    }]

    rn18 = throughput.get(("cifar10", "resnet18"))
    pd_completed = unique_count(completed, lambda row:
        row.get("experiment_type") == "pd_component_ablation" and row.get("method") != "reliability_pd_full")
    budget_completed = unique_count(completed, lambda row:
        row.get("experiment_type") == "budget_comparison" and row.get("budget_mode") == "fixed")
    sigma_completed = unique_count(completed, lambda row:
        row.get("experiment_type") == "sigma_sensitivity" and row.get("_prior_sigma") != 0.05,
        key_fields=("_prior_sigma", "seed"))
    estimates.extend([
        estimate_group("PD component ablation", 25, 5, 20, pd_completed, rn18, 120),
        estimate_group("Budget comparison", 50, 25, 25, budget_completed, rn18, 120),
        estimate_group("Sigma sensitivity", 20, 5, 15, sigma_completed, rn18, 120),
    ])
    tiny = throughput.get(("tiny_imagenet", "resnet18"))
    deit = throughput.get(("cifar100", "deit_tiny_patch16_224"))
    tiny_completed = unique_count(completed, lambda row: row.get("experiment_type") == "external_tiny_imagenet")
    deit_completed = unique_count(completed, lambda row: row.get("experiment_type") == "external_architecture")
    estimates.extend([
        estimate_group("Tiny ImageNet / ResNet-18", 25, 0, 25, tiny_completed, tiny, 120),
        estimate_group("CIFAR-100 / DeiT-Tiny", 25, 0, 25, deit_completed, deit, 100),
    ])

    known_hours = sum(float(row["remaining_gpu_hours"]) for row in estimates
                      if row["remaining_gpu_hours"] != "UNRESOLVED")
    benchmark_hours = 0.0
    benchmark_rows = read_csv(BENCHMARK_HISTORY) or read_csv(BENCHMARKS)
    for row in benchmark_rows:
        try:
            if row.get("benchmark_wall_clock_seconds") not in {None, "", "UNRESOLVED"}:
                benchmark_hours += float(row["benchmark_wall_clock_seconds"]) / 3600.0
            else:
                benchmark_hours += float(row["seconds_per_epoch"]) * (
                    int(row["warmup_epochs"]) + int(row["timed_epochs"])
                ) / 3600.0
        except (ValueError, TypeError):
            continue
    final_slots = sum(int(row["final_analysis_slots"]) for row in estimates)
    shared_slots = sum(int(row["shared_from_main_slots"]) for row in estimates)
    total_runs = sum(int(row["total_physical_runs"]) for row in estimates)
    completed_runs = sum(int(row["completed_physical_runs"]) for row in estimates)
    remaining_runs = sum(int(row["remaining_physical_runs"]) for row in estimates)
    manifest_rows = read_csv(REVISION / "results" / "run_manifest.csv")
    failed_runs = sum(row.get("status") == "FAILED" for row in manifest_rows)
    running_runs = sum(row.get("status") == "RUNNING" for row in manifest_rows)
    if completed_runs >= total_runs and total_runs:
        stage = "formal corrected-protocol queue complete"
    elif running_runs or completed_runs:
        stage = "formal corrected-protocol long-run queue in progress"
    else:
        stage = "corrective implementation and smoke verification complete; long-run queue not launched"
    unresolved_runs = sum(int(row["remaining_physical_runs"]) for row in estimates
                          if row["remaining_gpu_hours"] == "UNRESOLVED")
    fieldnames = ["experiment_group", "final_analysis_slots", "shared_from_main_slots",
                  "total_physical_runs", "completed_physical_runs", "remaining_physical_runs",
                  "max_epochs", "mean_hours_per_run",
                  "remaining_gpu_hours", "eta_status"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(estimates)

    table = ["| Experiment group | Analysis slots | Shared main slots | Physical total | Completed | Remaining | Max epochs | Mean h/run | Remaining GPU-h | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in estimates:
        mean_h = row["mean_hours_per_run"]
        hours = row["remaining_gpu_hours"]
        mean_text = mean_h if isinstance(mean_h, str) else f"{mean_h:.3f}"
        hours_text = hours if isinstance(hours, str) else f"{hours:.1f}"
        table.append(f"| {row['experiment_group']} | {row['final_analysis_slots']} | {row['shared_from_main_slots']} | {row['total_physical_runs']} | {row['completed_physical_runs']} | {row['remaining_physical_runs']} | {row['max_epochs']} | {mean_text} | {hours_text} | {row['eta_status']} |")
    known_24 = known_hours / 24.0
    known_20 = known_hours / 20.0
    known_date_24 = now + timedelta(hours=known_hours)
    known_date_20 = now + timedelta(days=known_20)
    sources = sorted({source for _, _, source in throughput.values()})
    (REVISION / "RUNTIME_ESTIMATE.md").write_text(
        "# Runtime Estimate\n\n"
        f"Generated: {now.isoformat()}  \nGPU: {GPU_NAME}  \nThroughput source: {', '.join(sources)}\n\n"
        + "\n".join(table)
        + f"\n\n## Totals\n\n- Final analysis inventory: {final_slots} run slots.\n"
          f"- Historical slots reusable under protocol v2: 0.\n"
          f"- Cross-analysis slots supplied by corrected main runs: {shared_slots}.\n"
          f"- Unique new physical runs: {total_runs}; completed: {completed_runs}; remaining: {remaining_runs}.\n"
          f"- Remaining runs awaiting a throughput benchmark: {unresolved_runs}.\n"
          f"- Computable subtotal: {known_hours:.1f} GPU-h = {known_24:.2f} calendar days at 24 h/day or {known_20:.2f} days at 20 h/day.\n"
          f"- Completed Stage 0 benchmark GPU-hours: approximately {benchmark_hours:.2f}.\n"
          f"- Earliest known-family completion at continuous execution: {known_date_24.isoformat()}.\n"
          f"- Known-family completion at 20 h/day: {known_date_20.isoformat()}.\n"
          + (f"- Conservative full completion at continuous execution: {known_date_24.isoformat()}.\n"
             if unresolved_runs == 0 else
             "- Full completion datetime: UNRESOLVED until all remaining throughput benchmarks finish.\n")
          + "\nEach per-run estimate includes the measured first-epoch warm-up cost plus steady-state timed epochs. The estimate uses every run's maximum epoch budget because corrected early-stopping trajectories do not yet exist. It is therefore a conservative upper-bound planning estimate, not a promise.\n",
        encoding="utf-8",
    )
    (REVISION / "REVISION_STATUS.md").write_text(
        "# Revision Status\n\n"
        f"Updated: {now.isoformat()}\n\n"
        f"- Stage: {stage}.\n"
        f"- Historical reusable slots under protocol v2: 0.\n- Completed corrected physical runs: {completed_runs}.\n"
        f"- Pending unique physical runs: {remaining_runs}.\n"
        f"- Failed revision runs in current manifest: {failed_runs}.\n"
        f"- Current measured revision GPU-hours (benchmarks): {benchmark_hours:.2f}.\n"
        f"- Estimated remaining currently computable GPU-hours: {known_hours:.1f}.\n"
        + (f"- Conservative continuous completion datetime: {known_date_24.isoformat()}.\n" if unresolved_runs == 0
           else "- Full completion datetime: awaiting remaining external throughput benchmarks.\n")
        + "- Corrected protocol: deterministic validation transforms and independent reliability LR/stop counters.\n",
        encoding="utf-8",
    )
    print(f"Currently computable remaining GPU-hours: {known_hours:.1f}")
    print(f"Physical runs: {completed_runs}/{total_runs} completed; {remaining_runs} remaining")
    print(f"Remaining runs awaiting benchmark: {unresolved_runs}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
