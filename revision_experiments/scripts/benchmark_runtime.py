"""在当前硬件上短跑各模型族，测量可靠性控制器与 epoch 吞吐。"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_experiment import evaluate, make_optimizer, sync_if_cuda, train_one_epoch
from src.controller import ControllerConfig, ReliabilityController
from src.data import build_loaders
from src.models import build_model
from src.reproducibility import set_seed
from src.signals import ReliabilitySignals, SignalConfig


OUTPUT = ROOT / "revision_experiments" / "results" / "runtime_benchmarks.csv"
HISTORY = ROOT / "revision_experiments" / "results" / "runtime_benchmark_history.csv"
FAMILIES = {
    "cifar10_resnet18": ("cifar10", "resnet18", "configs/cifar10_resnet18_main.json"),
    "cifar100_resnet50": ("cifar100", "resnet50", "configs/cifar100_resnet50_supplement.json"),
    "cifar100_vgg16": ("cifar100", "vgg16", "configs/cifar100_vgg16_supplement.json"),
    "tiny_imagenet_resnet18": (
        "tiny_imagenet", "resnet18", "revision_experiments/configs/tiny_imagenet/full.json"
    ),
    "cifar100_deit_tiny": (
        "cifar100", "deit_tiny_patch16_224", "revision_experiments/configs/deit_cifar100/full.json"
    ),
}


def measure_family(name, warmup_epochs, timed_epochs, seed):
    """预热后计时指定模型族，并返回可用于工期估算的记录。"""

    dataset, model_name, config_rel = FAMILIES[name]
    cfg = json.loads((ROOT / config_rel).read_text(encoding="utf-8"))
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Runtime benchmarks require CUDA; CPU fallback would produce a misleading ETA.")

    train_loader, val_loader, _, num_classes = build_loaders(
        dataset, str(ROOT / "data"), int(cfg["batch_size"]), int(cfg["num_workers"]),
        float(cfg["val_ratio"]), seed,
        image_size=cfg.get("image_size"), normalization=cfg.get("normalization", "dataset"),
    )
    checkpoint = cfg.get("pretrained_checkpoint")
    if checkpoint and not Path(checkpoint).is_absolute():
        checkpoint = str(ROOT / checkpoint)
    model = build_model(model_name, num_classes, pretrained_checkpoint=checkpoint).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg, model)
    signals = ReliabilitySignals(model, len(train_loader.dataset), SignalConfig(
        alpha=cfg["controller"]["alpha"], beta=cfg["controller"]["beta"],
        delta=cfg["controller"]["delta"], prior_sigma=cfg["controller"]["prior_sigma"],
        lambda_g=cfg["controller"]["lambda_g"], lambda_u=cfg["controller"]["lambda_u"],
    ))
    controller = ReliabilityController(ControllerConfig(
        gamma=cfg["controller"]["gamma"], min_lr=cfg["controller"]["min_lr"],
        p_lr=cfg["controller"]["p_lr"], p_stop=cfg["controller"]["p_stop"],
        epsilon=cfg["controller"]["epsilon"],
    ))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    epoch_seconds = []
    controller_seconds = []
    all_epoch_seconds = []

    total_epochs = warmup_epochs + timed_epochs
    for epoch in range(1, total_epochs + 1):
        sync_if_cuda(device)
        started = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, signals)
        val_stats = evaluate(model, val_loader, criterion, device)
        sync_if_cuda(device)
        sig = signals.compute(
            model, train_stats["train_nll"], val_stats["nll"],
            train_stats["grad_norm"], train_stats["update_norm"], "reliability",
        )
        decision_started = time.perf_counter()
        controller.step(optimizer, sig["reliability_rbar"], val_stats["nll"], epoch)
        sync_if_cuda(device)
        decision_elapsed = time.perf_counter() - decision_started
        control_elapsed = (
            sig["pb_computation_seconds"] + sig["pd_computation_seconds"]
            + sig["standardization_smoothing_seconds"] + max(0.0, decision_elapsed)
        )
        elapsed = time.perf_counter() - started
        all_epoch_seconds.append(elapsed)
        if epoch > warmup_epochs:
            epoch_seconds.append(elapsed)
            controller_seconds.append(control_elapsed)
        print(
            f"[{name}] epoch {epoch}/{total_epochs}: {elapsed:.3f}s "
            f"(controller {control_elapsed:.4f}s)", flush=True,
        )

    return {
        "family": name,
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "warmup_epochs": warmup_epochs,
        "timed_epochs": timed_epochs,
        "seconds_per_epoch": sum(epoch_seconds) / len(epoch_seconds),
        "controller_seconds_per_epoch": sum(controller_seconds) / len(controller_seconds),
        "controller_overhead_percent": 100.0 * sum(controller_seconds) / sum(epoch_seconds),
        "benchmark_wall_clock_seconds": sum(all_epoch_seconds),
        "peak_vram_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "gpu": torch.cuda.get_device_name(device),
        "measured_at": datetime.now().astimezone().isoformat(),
        "status": "MEASURED",
        "notes": "Full train+validation epochs; first epoch excluded as warm-up.",
    }


def write_rows(rows):
    """覆盖写入每个模型族的最新基准结果。"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if OUTPUT.exists():
        with OUTPUT.open("r", encoding="utf-8", newline="") as handle:
            existing = [row for row in csv.DictReader(handle) if row.get("family") in FAMILIES]
    by_family = {row["family"]: row for row in existing}
    by_family.update({row["family"]: row for row in rows})
    for row in by_family.values():
        try:
            row["controller_overhead_percent"] = (
                100.0 * float(row["controller_seconds_per_epoch"]) / float(row["seconds_per_epoch"])
            )
        except (ValueError, TypeError, ZeroDivisionError):
            row["controller_overhead_percent"] = "UNRESOLVED"
    fieldnames = [
        "family", "dataset", "model", "seed", "warmup_epochs", "timed_epochs",
        "seconds_per_epoch", "controller_seconds_per_epoch", "controller_overhead_percent",
        "benchmark_wall_clock_seconds", "peak_vram_mb", "gpu",
        "measured_at", "status", "notes",
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_family.values())


def append_history(rows):
    """追加带时间戳的基准历史，供稳健工期估算使用。"""

    measured = [row for row in rows if row.get("status") == "MEASURED"]
    if not measured:
        return
    fields = [
        "family", "dataset", "model", "seed", "warmup_epochs", "timed_epochs",
        "seconds_per_epoch", "controller_seconds_per_epoch", "controller_overhead_percent",
        "benchmark_wall_clock_seconds", "peak_vram_mb", "gpu", "measured_at", "status", "notes",
    ]
    exists = HISTORY.exists()
    with HISTORY.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(measured)


def main():
    """选择模型族执行基准并更新当前值与追加历史。"""

    parser = argparse.ArgumentParser(description="Measure RTX throughput for revision experiment families.")
    parser.add_argument("--family", action="append", choices=[*FAMILIES])
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--timed-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    selected = args.family or [*FAMILIES]
    rows = []
    for name in selected:
        rows.append(measure_family(name, args.warmup_epochs, args.timed_epochs, args.seed))
    append_history(rows)
    write_rows(rows)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
