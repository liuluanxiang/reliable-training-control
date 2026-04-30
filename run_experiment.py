import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.reproducibility import set_seed
from src.data import build_loaders
from src.models import build_model
from src.signals import SignalConfig, ReliabilitySignals
from src.controller import ControllerConfig, ReliabilityController
from src.metrics import (
    AverageMeter, accuracy, topk_accuracy, expected_calibration_error,
    maximum_calibration_error, brier_score, confidence_stats, reliability_bins,
    risk_coverage_curve, aurc_eaurc, area_under_curve, slope_last
)


def make_optimizer(cfg, model):
    if cfg["optimizer"].lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )
    if cfg["optimizer"].lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {cfg['optimizer']}")


def make_scheduler(method, cfg, optimizer):
    epochs = int(cfg["epochs"])
    if method == "step":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(epochs * 0.5), int(epochs * 0.75)],
            gamma=0.1,
        )
    if method == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if method == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=cfg["controller"]["min_lr"],
        )
    return None


def resolve_device(cfg):
    requested = str(cfg.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requested CUDA, but this Python environment cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build in the selected virtual environment."
        )
    return device


def train_one_epoch(model, loader, optimizer, criterion, device, signals):
    model.train()
    signals.begin_epoch(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top5_meter = AverageMeter()
    last_grad_norm = 0.0
    non_blocking = device.type == "cuda"

    for x, y in loader:
        x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        last_grad_norm = signals.gradient_norm(model)
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy(logits.detach(), y), x.size(0))
        top5_meter.update(topk_accuracy(logits.detach(), y, k=5), x.size(0))

    update_norm = signals.update_norm(model)
    return {
        "train_nll": loss_meter.avg,
        "train_acc": acc_meter.avg,
        "train_top5_acc": top5_meter.avg,
        "grad_norm": last_grad_norm,
        "update_norm": update_norm,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_logits=False):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top5_meter = AverageMeter()
    all_logits = []
    all_targets = []
    non_blocking = device.type == "cuda"

    for x, y in loader:
        x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy(logits, y), x.size(0))
        top5_meter.update(topk_accuracy(logits, y, k=5), x.size(0))

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    ece = expected_calibration_error(logits, targets)
    mce = maximum_calibration_error(logits, targets)
    brier = brier_score(logits, targets)
    aurc, eaurc = aurc_eaurc(logits, targets)
    cstats = confidence_stats(logits)

    out = {
        "nll": loss_meter.avg,
        "acc": acc_meter.avg,
        "top5_acc": top5_meter.avg,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "aurc": aurc,
        "eaurc": eaurc,
        **cstats,
    }
    if return_logits:
        out["logits"] = logits
        out["targets"] = targets
    return out


def run_one(cfg, method, seed):
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = resolve_device(cfg)

    exp_name = cfg["experiment_name"]
    out_dir = Path(cfg["results_dir"]) / exp_name / method / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, num_classes = build_loaders(
        cfg["dataset"],
        cfg["data_root"],
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        float(cfg["val_ratio"]),
        seed,
    )

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    model = build_model(cfg["model"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(method, cfg, optimizer)

    sig_cfg = SignalConfig(
        alpha=cfg["controller"]["alpha"],
        beta=cfg["controller"]["beta"],
        delta=cfg["controller"]["delta"],
        prior_sigma=cfg["controller"]["prior_sigma"],
        lambda_g=cfg["controller"]["lambda_g"],
        lambda_u=cfg["controller"]["lambda_u"],
    )
    signals = ReliabilitySignals(model, len(train_loader.dataset), sig_cfg)

    controller = ReliabilityController(
        ControllerConfig(
            gamma=cfg["controller"]["gamma"],
            min_lr=cfg["controller"]["min_lr"],
            p_lr=cfg["controller"]["p_lr"],
            p_stop=cfg["controller"]["p_stop"],
            epsilon=cfg["controller"]["epsilon"],
        )
    )

    history = []
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    start = time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, signals)
        val_stats = evaluate(model, val_loader, criterion, device, return_logits=False)

        sig = signals.compute(
            model=model,
            train_nll=train_stats["train_nll"],
            val_nll=val_stats["nll"],
            grad_norm=train_stats["grad_norm"],
            update_norm=train_stats["update_norm"],
        )

        lr_before = optimizer.param_groups[0]["lr"]
        action = {
            "lr_reduced": False,
            "early_stop": False,
            "stop_reason": "",
            "bad_rel": 0,
            "bad_val": 0,
            "first_lr_drop_epoch": -1,
            "lr_reductions": 0,
        }

        if method == "reliability":
            action = controller.step(optimizer, sig["reliability_rbar"], val_stats["nll"], epoch)
        elif method in ["step", "cosine"]:
            scheduler.step()
        elif method == "plateau":
            scheduler.step(val_stats["nll"])

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row = {
            "experiment": exp_name,
            "dataset": cfg["dataset"],
            "model": cfg["model"],
            "method": method,
            "seed": seed,
            "epoch": epoch,
            "lr": lr_before,
            **train_stats,
            "val_nll": val_stats["nll"],
            "val_acc": val_stats["acc"],
            "val_top5_acc": val_stats["top5_acc"],
            "val_ece": val_stats["ece"],
            "val_mce": val_stats["mce"],
            "val_brier": val_stats["brier"],
            "val_aurc": val_stats["aurc"],
            "val_eaurc": val_stats["eaurc"],
            "val_confidence_mean": val_stats["confidence_mean"],
            "val_entropy_mean": val_stats["entropy_mean"],
            "val_margin_mean": val_stats["margin_mean"],
            "gen_gap_loss": val_stats["nll"] - train_stats["train_nll"],
            "gen_gap_acc": train_stats["train_acc"] - val_stats["acc"],
            **sig,
            **action,
            "lr_after": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"{exp_name} | {method} | seed={seed} | epoch={epoch:03d} | "
            f"val_acc={val_stats['acc']:.4f} | val_nll={val_stats['nll']:.4f} | "
            f"rbar={sig['reliability_rbar']:.3f} | lr={lr_before:.5g}"
        )

        if action["early_stop"]:
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_stats = evaluate(model, test_loader, criterion, device, return_logits=True)
    logits = test_stats.pop("logits")
    targets = test_stats.pop("targets")

    torch.save(logits, out_dir / "test_logits.pt")
    torch.save(targets, out_dir / "test_targets.pt")

    pd.DataFrame(reliability_bins(logits, targets)).to_csv(out_dir / "calibration_bins.csv", index=False)
    cov, risk = risk_coverage_curve(logits, targets)
    pd.DataFrame({"coverage": cov, "risk": risk}).to_csv(out_dir / "risk_coverage_curve.csv", index=False)

    summary = {
        "experiment": exp_name,
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "method": method,
        "seed": seed,
        "epochs_run": int(hist_df["epoch"].max()),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "final_val_acc": float(hist_df["val_acc"].iloc[-1]),
        "test_acc": float(test_stats["acc"]),
        "test_top5_acc": float(test_stats["top5_acc"]),
        "test_nll": float(test_stats["nll"]),
        "test_ece": float(test_stats["ece"]),
        "test_mce": float(test_stats["mce"]),
        "test_brier": float(test_stats["brier"]),
        "test_aurc": float(test_stats["aurc"]),
        "test_eaurc": float(test_stats["eaurc"]),
        "test_confidence_mean": float(test_stats["confidence_mean"]),
        "test_confidence_std": float(test_stats["confidence_std"]),
        "test_margin_mean": float(test_stats["margin_mean"]),
        "test_entropy_mean": float(test_stats["entropy_mean"]),
        "final_gen_gap_loss": float(hist_df["gen_gap_loss"].iloc[-1]),
        "final_gen_gap_acc": float(hist_df["gen_gap_acc"].iloc[-1]),
        "late_degradation": float(best_val_acc - hist_df["val_acc"].iloc[-1]),
        "aulc_val_acc": float(area_under_curve(hist_df["val_acc"].values)),
        "val_acc_last10_slope": float(slope_last(hist_df["val_acc"].values, 10)),
        "pb_final": float(hist_df["pb_signal"].iloc[-1]),
        "pd_final": float(hist_df["pd_signal"].iloc[-1]),
        "rbar_final": float(hist_df["reliability_rbar"].iloc[-1]),
        "lr_reductions": int(hist_df["lr_reduced"].sum()),
        "first_lr_drop_epoch": int(hist_df.loc[hist_df["lr_reduced"] == True, "epoch"].min()) if hist_df["lr_reduced"].any() else -1,
        "runtime_seconds": float(time.time() - start),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    all_rows = []

    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            all_rows.append(run_one(cfg, method, int(seed)))

    exp_dir = Path(cfg["results_dir"]) / cfg["experiment_name"]
    pd.DataFrame(all_rows).to_csv(exp_dir / "summary_by_seed.csv", index=False)

    master_path = Path(cfg["results_dir"]) / "master_summary.csv"
    new_df = pd.DataFrame(all_rows)
    if master_path.exists():
        old = pd.read_csv(master_path)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(["experiment", "method", "seed"], keep="last")
    else:
        combined = new_df
    combined.to_csv(master_path, index=False)


if __name__ == "__main__":
    main()
