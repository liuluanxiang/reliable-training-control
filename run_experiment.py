"""统一实验入口：执行单配置的训练、控制、评估、审计与结果落盘。

配置文件决定数据集、模型、方法和随机种子。本模块将每个 ``method x seed``
视为独立科学运行，并把逐 epoch 轨迹、最佳验证检查点、一次性测试结果以及
复现哈希写入结果目录。
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
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
from src.results_io import infer_experiment_type


RESULT_ARTIFACTS = [
    "best_checkpoint.pt",
    "config_used.json",
    "run_metadata.json",
    "history.csv",
    "summary.json",
    "test_logits.pt",
    "test_targets.pt",
    "calibration_bins.csv",
    "risk_coverage_curve.csv",
]


def canonical_hash(value):
    """对 JSON 可序列化对象计算与键顺序无关的 SHA-256。"""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scientific_run_hash(cfg, method, seed):
    """计算科学配置哈希，排除不影响数值结果的调度/路径字段。"""

    orchestration_only = {
        "experiment_name", "experiment_type", "methods", "seeds", "results_dir"
    }
    payload = {key: value for key, value in cfg.items() if key not in orchestration_only}
    payload.update({"method": method, "seed": int(seed)})
    return canonical_hash(payload)


def file_sha256(path):
    """分块计算外部文件哈希；用于记录 DeiT 预训练权重的确切版本。"""

    if not path:
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_hash(state):
    """将参数名、dtype、shape 和原始字节纳入模型状态哈希。"""

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def split_hashes(train_loader, val_loader):
    """返回完整划分哈希与验证索引哈希，支持跨运行核对数据协议。"""

    train_indices = list(getattr(train_loader.dataset, "indices", range(len(train_loader.dataset))))
    val_indices = list(getattr(val_loader.dataset, "indices", range(len(val_loader.dataset))))
    return canonical_hash({"train": train_indices, "validation": val_indices}), canonical_hash(val_indices)


def git_commit():
    """读取当前提交号；非 Git 环境下保留明确的不可解析标记。"""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNRESOLVED"


def now_text():
    """返回供实时状态和日志显示的本地时间。"""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds):
    """将秒数格式化为紧凑且适合仪表盘显示的持续时间。"""

    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def make_run_id():
    """生成批次级时间戳标识，主要用于归档旧结果。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sync_if_cuda(device):
    """仅在 CUDA 上同步异步工作，用于准确计时和进度估计。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def result_artifacts(out_dir):
    """列举已有运行产物，作为覆盖/跳过策略的输入。"""

    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    protected = [name for name in RESULT_ARTIFACTS if (out_dir / name).exists()]
    other_files = [
        path.name
        for path in out_dir.iterdir()
        if path.is_file() and path.name not in protected
    ]
    return protected + sorted(other_files)


def load_existing_summary(out_dir):
    """容错读取已完成摘要；缺失或损坏时返回 ``None``。"""

    path = Path(out_dir) / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def existing_scientific_hash(out_dir, summary):
    """读取旧结果的科学哈希，并兼容尚未直接记录哈希的历史结果。"""

    stored = (summary or {}).get("scientific_config_hash")
    if stored:
        return stored
    config_path = Path(out_dir) / "config_used.json"
    if not config_path.exists() or not summary:
        return None
    try:
        existing_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        return scientific_run_hash(existing_cfg, summary["method"], summary["seed"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def planned_output_dir(cfg, method, seed):
    """按照 ``experiment/method/seed`` 约定生成运行目录。"""

    return Path(cfg["results_dir"]) / cfg["experiment_name"] / method / f"seed_{seed}"


def planned_runs(cfg, existing_policy):
    """在训练前审计所有运行，标记将运行、跳过、归档或报错的项目。"""

    runs = []
    run_index = 0
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            run_index += 1
            out_dir = planned_output_dir(cfg, method, int(seed))
            artifacts = result_artifacts(out_dir)
            has_complete_artifact_set = set(RESULT_ARTIFACTS).issubset(artifacts)
            if not artifacts:
                status = "will_run"
            elif existing_policy == "skip":
                if has_complete_artifact_set:
                    summary = load_existing_summary(out_dir)
                    actual_hash = existing_scientific_hash(out_dir, summary)
                    expected_hash = scientific_run_hash(cfg, method, int(seed))
                    status = "skip_completed" if actual_hash == expected_hash else "error_config_mismatch"
                else:
                    status = "skip_partial"
            elif existing_policy == "rerun-partial":
                if has_complete_artifact_set:
                    summary = load_existing_summary(out_dir)
                    actual_hash = existing_scientific_hash(out_dir, summary)
                    expected_hash = scientific_run_hash(cfg, method, int(seed))
                    status = "skip_completed" if actual_hash == expected_hash else "archive_mismatch_then_run"
                else:
                    status = "archive_partial_then_run"
            elif existing_policy == "error":
                status = "error_existing"
            elif existing_policy == "archive":
                status = "archive_then_run"
            elif existing_policy == "overwrite":
                status = "overwrite"
            else:
                status = "unknown"
            runs.append({
                "run_index": run_index,
                "experiment": cfg["experiment_name"],
                "method": method,
                "seed": int(seed),
                "output_dir": str(out_dir),
                "artifacts": artifacts,
                "status": status,
            })
    return runs


def archive_existing_output(out_dir, archive_root):
    """把现有目录整体移入带批次标识的归档树，保留原始证据。"""

    out_dir = Path(out_dir)
    archive_root = Path(archive_root)
    relative = out_dir.relative_to(out_dir.parents[2])
    target = archive_root / relative
    if target.exists():
        suffix = 1
        while True:
            candidate = target.with_name(f"{target.name}_{suffix:02d}")
            if not candidate.exists():
                target = candidate
                break
            suffix += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out_dir), str(target))
    return target


def write_dashboard(dashboard_path, state_path):
    """从静态模板生成指向实时 JSON 状态的实验仪表盘。"""

    dashboard_path = Path(dashboard_path)
    state_path = Path(state_path)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    rel_state = os.path.relpath(state_path, dashboard_path.parent).replace("\\", "/")
    template_path = Path(__file__).resolve().parent / "src" / "progress_dashboard_template.html"
    html = template_path.read_text(encoding="utf-8").replace("__STATE_URL__", rel_state)
    dashboard_path.write_text(html, encoding="utf-8")


class ProgressLogger:
    """同步维护控制台日志、追加日志文件和仪表盘 JSON 状态。"""

    def __init__(self, mode="epoch", interval=50, seconds=30.0, log_path=None):
        """初始化日志频率、可选文件句柄和实时状态骨架。"""

        self.mode = mode
        self.interval = max(1, int(interval))
        self.seconds = max(1.0, float(seconds))
        self._fh = None
        self.state_path = None
        self.dashboard_path = None
        self.state = {
            "status": "initializing",
            "updated_at": now_text(),
            "messages": [],
            "planned_runs": [],
            "completed_runs": 0,
            "current": {},
            "finished_runs": [],
        }
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8", buffering=1)

    def enabled(self):
        """返回是否启用任何进度输出。"""

        return self.mode != "none"

    def batch_enabled(self):
        """返回是否启用 mini-batch 级心跳。"""

        return self.mode == "batch"

    def configure_state(self, state_path, dashboard_path=None):
        """绑定状态文件，并保留队列脚本写入的顶层通知字段。"""

        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        preserved = {}
        if self.state_path.exists():
            try:
                previous = json.loads(self.state_path.read_text(encoding="utf-8"))
                for key in ["main_notice", "pipeline"]:
                    if key in previous:
                        preserved[key] = previous[key]
            except Exception:
                preserved = {}
        self.state.update(preserved)
        if dashboard_path:
            self.dashboard_path = Path(dashboard_path)
            write_dashboard(self.dashboard_path, self.state_path)
        self.write_state()

    def update_state(self, **kwargs):
        """更新批次级状态并立即落盘。"""

        self.state.update(kwargs)
        self.state["updated_at"] = now_text()
        self.write_state()

    def update_current(self, **kwargs):
        """合并当前运行/epoch/batch 的局部进度。"""

        current = dict(self.state.get("current", {}))
        current.update(kwargs)
        self.state["current"] = current
        self.state["updated_at"] = now_text()
        self.write_state()

    def record_finished(self, row):
        """登记一个已完成或已跳过的运行。"""

        finished = self.state.setdefault("finished_runs", [])
        finished.append(row)
        self.state["completed_runs"] = len(finished)
        self.state["updated_at"] = now_text()
        self.write_state()

    def write_state(self):
        """写出仪表盘读取的 JSON 快照。"""

        if self.state_path is None:
            return
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def log(self, message):
        """输出消息，并只在状态中保留最近 120 条以限制文件增长。"""

        if not self.enabled():
            return
        print(message, flush=True)
        if self._fh is not None:
            self._fh.write(message + "\n")
        messages = self.state.setdefault("messages", [])
        messages.append(f"{now_text()} | {message}")
        del messages[:-120]
        self.state["updated_at"] = now_text()
        self.write_state()

    def close(self):
        """关闭可选的追加日志句柄。"""

        if self._fh is not None:
            self._fh.close()
            self._fh = None


def make_optimizer(cfg, model):
    """按配置建立 SGD 或 AdamW，并统一传入权重衰减。"""

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
    """为传统基线建立调度器；可靠性方法由自定义控制器接管。"""

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


def is_reliability_controller_method(method):
    """判断方法是否属于完整算法或其可靠性消融版本。"""

    return method == "reliability" or str(method).startswith("reliability_")


def resolve_device(cfg):
    """解析运行设备，并在显式 CUDA 请求不可满足时尽早失败。"""

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


def train_one_epoch(model, loader, optimizer, criterion, device, signals, progress=None, context=""):
    """训练一个 epoch，并采集 PB/PD 所需的梯度与参数更新范数。

    梯度范数在最后一个 mini-batch 的 ``optimizer.step`` 之前测量；更新范数
    则在整个 epoch 完成后相对 ``begin_epoch`` 快照测量。这样每个 epoch 只
    增加一次全参数扫描，且两个量对应明确的时间边界。
    """

    model.train()
    signals.begin_epoch(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top5_meter = AverageMeter()
    last_grad_norm = 0.0
    non_blocking = device.type == "cuda"
    total_batches = len(loader)
    epoch_start = time.time()
    last_log = epoch_start

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # 只取 epoch 末批次梯度，控制额外计算成本并固定采样位置。
        if batch_idx == total_batches:
            last_grad_norm = signals.gradient_norm(model)
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy(logits.detach(), y), x.size(0))
        top5_meter.update(topk_accuracy(logits.detach(), y, k=5), x.size(0))

        now = time.time()
        should_log = (
            progress is not None
            and progress.batch_enabled()
            and (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % progress.interval == 0
                or now - last_log >= progress.seconds
            )
        )
        if should_log:
            sync_if_cuda(device)
            now = time.time()
            elapsed = now - epoch_start
            batches_per_sec = batch_idx / max(elapsed, 1e-12)
            eta_epoch = (total_batches - batch_idx) / max(batches_per_sec, 1e-12)
            progress.update_current(
                phase="training",
                batch=batch_idx,
                total_batches=total_batches,
                batch_elapsed=format_duration(elapsed),
                eta_epoch=format_duration(eta_epoch),
                train_loss=loss_meter.avg,
                train_acc=acc_meter.avg,
            )
            progress.log(
                f"{context} | train batch={batch_idx}/{total_batches} | "
                f"elapsed={format_duration(elapsed)} | eta_epoch={format_duration(eta_epoch)} | "
                f"loss={loss_meter.avg:.4f} | acc={acc_meter.avg:.4f}"
            )
            last_log = now

    update_norm = signals.update_norm(model)
    return {
        "train_nll": loss_meter.avg,
        "train_acc": acc_meter.avg,
        "train_top5_acc": top5_meter.avg,
        "grad_norm": last_grad_norm,
        "update_norm": update_norm,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_logits=False, progress=None, context="eval"):
    """在无梯度模式下计算分类、校准和选择性预测指标。

    logits 和 targets 始终先汇总到 CPU，以便所有派生指标使用完全相同的样本
    顺序；只有最终测试需要保存原始张量时才通过 ``return_logits`` 返回它们。
    """

    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top5_meter = AverageMeter()
    all_logits = []
    all_targets = []
    non_blocking = device.type == "cuda"
    total_batches = len(loader)
    eval_start = time.time()
    last_log = eval_start

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(accuracy(logits, y), x.size(0))
        top5_meter.update(topk_accuracy(logits, y, k=5), x.size(0))

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

        now = time.time()
        should_log = (
            progress is not None
            and progress.batch_enabled()
            and (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % progress.interval == 0
                or now - last_log >= progress.seconds
            )
        )
        if should_log:
            sync_if_cuda(device)
            now = time.time()
            elapsed = now - eval_start
            batches_per_sec = batch_idx / max(elapsed, 1e-12)
            eta_eval = (total_batches - batch_idx) / max(batches_per_sec, 1e-12)
            progress.update_current(
                phase="evaluating",
                batch=batch_idx,
                total_batches=total_batches,
                batch_elapsed=format_duration(elapsed),
                eta_eval=format_duration(eta_eval),
                eval_nll=loss_meter.avg,
                eval_acc=acc_meter.avg,
            )
            progress.log(
                f"{context} | eval batch={batch_idx}/{total_batches} | "
                f"elapsed={format_duration(elapsed)} | eta_eval={format_duration(eta_eval)} | "
                f"nll={loss_meter.avg:.4f} | acc={acc_meter.avg:.4f}"
            )
            last_log = now

    # 先拼接全数据集预测，再计算 ECE/AURC，避免按 batch 平均造成统计偏差。
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


def run_one(cfg, method, seed, progress, run_index=1, total_runs=1, existing_policy="skip", run_id=None):
    """执行一个 ``method x seed`` 的完整、可审计实验运行。

    生命周期依次为：旧结果审计、数据与模型建立、逐 epoch 训练/验证、仅按
    验证准确率选择检查点、最终一次测试、派生文件与摘要落盘。训练期间从不
    读取测试指标。
    """

    exp_name = cfg["experiment_name"]
    exp_type = str(cfg.get("experiment_type", "unspecified") or "unspecified")
    out_dir = planned_output_dir(cfg, method, seed)
    artifacts = result_artifacts(out_dir)
    progress.update_current(
        run_index=run_index,
        total_runs=total_runs,
        experiment=exp_name,
        method=method,
        seed=seed,
        phase="checking_existing_results",
        epoch=0,
        total_epochs=int(cfg["epochs"]),
        batch=0,
        total_batches=0,
    )

    # 先处理已有产物，避免无提示覆盖或把不完整运行误判为已完成。
    if artifacts:
        if existing_policy in ["skip", "rerun-partial"]:
            existing_summary = load_existing_summary(out_dir)
            expected_hash = scientific_run_hash(cfg, method, seed)
            actual_hash = existing_scientific_hash(out_dir, existing_summary)
            if existing_summary is not None and actual_hash == expected_hash:
                progress.log(
                    f"[run {run_index}/{total_runs}] skip existing completed result | "
                    f"experiment={exp_name} | method={method} | seed={seed} | "
                    f"artifacts={','.join(artifacts)}"
                )
                existing_summary.setdefault("experiment", exp_name)
                existing_summary.setdefault("experiment_type", exp_type)
                existing_summary.setdefault("dataset", cfg["dataset"])
                existing_summary.setdefault("model", cfg["model"])
                existing_summary.setdefault("method", method)
                existing_summary.setdefault("seed", seed)
                existing_summary["skipped_existing"] = True
                progress.record_finished({
                    "run_index": run_index,
                    "experiment": exp_name,
                    "method": method,
                    "seed": seed,
                    "status": "skipped_existing",
                })
                return existing_summary

            if existing_summary is not None and existing_policy == "skip":
                raise RuntimeError(
                    f"Existing completed result in {out_dir} does not match the requested scientific "
                    f"configuration (expected {expected_hash}, found {actual_hash or 'UNAVAILABLE'}). "
                    "Use --existing rerun-partial or --existing archive."
                )

            if existing_policy == "skip":
                progress.log(
                    f"[run {run_index}/{total_runs}] skip partial existing output | "
                    f"experiment={exp_name} | method={method} | seed={seed} | "
                    f"artifacts={','.join(artifacts)} | use --existing rerun-partial to archive and rerun"
                )
                progress.record_finished({
                    "run_index": run_index,
                    "experiment": exp_name,
                    "method": method,
                    "seed": seed,
                    "status": "skipped_partial",
                })
                return None

            archive_root = Path(cfg["results_dir"]) / "_archive" / (run_id or make_run_id())
            archive_dir = archive_existing_output(out_dir, archive_root)
            progress.log(
                f"[run {run_index}/{total_runs}] archived partial/mismatched existing output | "
                f"from={out_dir} | to={archive_dir} | artifacts={','.join(artifacts)}"
            )

        if existing_policy == "error":
            raise FileExistsError(
                f"Existing result artifacts found in {out_dir}: {artifacts}. "
                "Use --existing skip, --existing rerun-partial, --existing archive, or --existing overwrite."
            )

        if existing_policy == "archive":
            archive_root = Path(cfg["results_dir"]) / "_archive" / (run_id or make_run_id())
            archive_dir = archive_existing_output(out_dir, archive_root)
            progress.log(
                f"[run {run_index}/{total_runs}] archived existing output | "
                f"from={out_dir} | to={archive_dir}"
            )
        elif existing_policy == "overwrite":
            progress.log(
                f"[run {run_index}/{total_runs}] overwrite allowed | output_dir={out_dir} | "
                f"artifacts={','.join(artifacts)}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_hash(cfg)
    scientific_config_hash = scientific_run_hash(cfg, method, seed)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = resolve_device(cfg)

    train_loader, val_loader, test_loader, num_classes = build_loaders(
        cfg["dataset"],
        cfg["data_root"],
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        float(cfg["val_ratio"]),
        seed,
        image_size=cfg.get("image_size"),
        normalization=cfg.get("normalization", "dataset"),
    )

    progress.log(
        f"[run {run_index}/{total_runs}] start | experiment={exp_name} | method={method} | "
        f"seed={seed} | epochs={int(cfg['epochs'])} | train_batches={len(train_loader)} | "
        f"val_batches={len(val_loader)} | test_batches={len(test_loader)}"
    )
    progress.update_current(
        phase="training",
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        test_batches=len(test_loader),
        total_batches=len(train_loader),
    )
    progress.log(f"Using device: {device}")
    if device.type == "cuda":
        progress.log(f"CUDA device: {torch.cuda.get_device_name(device)}")

    # 模型在固定随机种子后建立；初始化、数据划分和外部权重都记录哈希。
    model = build_model(
        cfg["model"], num_classes, pretrained_checkpoint=cfg.get("pretrained_checkpoint")
    ).to(device)
    pretrained_checkpoint_hash = file_sha256(cfg.get("pretrained_checkpoint"))
    initialization_hash = tensor_state_hash(model.state_dict())
    split_hash, validation_split_hash = split_hashes(train_loader, val_loader)
    commit = git_commit()
    metadata = {
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "method": method,
        "seed": seed,
        "budget_mode": cfg.get("budget_mode", "adaptive"),
        "data_protocol_version": cfg.get("data_protocol_version", "legacy_augmented_validation_v1"),
        "variant": method,
        "config_hash": config_hash,
        "scientific_config_hash": scientific_config_hash,
        "pretrained_checkpoint_sha256": pretrained_checkpoint_hash,
        "split_hash": split_hash,
        "validation_split_hash": validation_split_hash,
        "initialization_hash": initialization_hash,
        "data_order_seed": seed,
        "augmentation_seed": seed,
        "git_commit": commit,
        "test_evaluation_count": 0,
        "test_evaluation_timestamp": "",
        "checkpoint_selected_before_test": False,
        "test_used_during_training": False,
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(method, cfg, optimizer)

    # 论文算法参数由同一 controller 配置块注入信号计算器与动作控制器。
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
    epoch_times = []
    controller_times = []
    checkpoint_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 单个 epoch 的严格顺序：训练 -> 验证 -> 信号 -> 调度决策 -> 检查点。
    for epoch in range(1, int(cfg["epochs"]) + 1):
        sync_if_cuda(device)
        epoch_start = time.time()
        epoch_context = (
            f"[run {run_index}/{total_runs}] {exp_name} | {method} | "
            f"seed={seed} | epoch={epoch:03d}/{int(cfg['epochs']):03d}"
        )
        progress.update_current(
            phase="training",
            epoch=epoch,
            total_epochs=int(cfg["epochs"]),
            batch=0,
            total_batches=len(train_loader),
            epoch_time="running",
            eta_run="calculating",
        )
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, signals,
            progress=progress, context=epoch_context
        )
        progress.update_current(
            phase="evaluating",
            batch=0,
            total_batches=len(val_loader),
        )
        val_stats = evaluate(
            model, val_loader, criterion, device, return_logits=False,
            progress=progress, context=epoch_context
        )

        sync_if_cuda(device)
        sig = signals.compute(
            model=model,
            train_nll=train_stats["train_nll"],
            val_nll=val_stats["nll"],
            grad_norm=train_stats["grad_norm"],
            update_norm=train_stats["update_norm"],
            variant=method,
        )

        lr_before = optimizer.param_groups[0]["lr"]
        action = {
            "lr_reduced": False,
            "early_stop": False,
            "stop_reason": "",
            "bad_rel": 0,
            "bad_rel_lr": 0,
            "bad_rel_stop": 0,
            "bad_val": 0,
            "first_lr_drop_epoch": -1,
            "lr_reductions": 0,
        }

        # 所有基线与本方法只在验证结束后更新下一 epoch 的学习率。
        decision_start = time.perf_counter()
        if is_reliability_controller_method(method):
            action = controller.step(optimizer, sig["reliability_rbar"], val_stats["nll"], epoch)
        elif method in ["step", "cosine"]:
            scheduler.step()
        elif method == "plateau":
            scheduler.step(val_stats["nll"])
        sync_if_cuda(device)
        controller_decision_seconds = time.perf_counter() - decision_start
        controller_seconds = (
            sig["pb_computation_seconds"]
            + sig["pd_computation_seconds"]
            + sig["standardization_smoothing_seconds"]
            + controller_decision_seconds
        )
        controller_times.append(controller_seconds)

        # 模型选择唯一依据是验证准确率，测试集尚未被访问。
        if val_stats["acc"] > best_val_acc:
            checkpoint_start = time.perf_counter()
            best_val_acc = val_stats["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint_seconds += time.perf_counter() - checkpoint_start

        row = {
            "experiment": exp_name,
            "experiment_type": exp_type,
            "dataset": cfg["dataset"],
            "model": cfg["model"],
            "method": method,
            "seed": seed,
            "budget_mode": cfg.get("budget_mode", "adaptive"),
            "data_protocol_version": cfg.get("data_protocol_version", "legacy_augmented_validation_v1"),
            "scientific_config_hash": scientific_config_hash,
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
            "controller_total_seconds_epoch": controller_seconds,
            "controller_decision_seconds": controller_decision_seconds,
        }
        history.append(row)

        sync_if_cuda(device)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_left = int(cfg["epochs"]) - epoch
        eta_run = avg_epoch_time * epochs_left
        progress.update_current(
            phase="epoch_complete",
            batch=len(val_loader),
            total_batches=len(val_loader),
            epoch_time=format_duration(epoch_time),
            avg_epoch_time=format_duration(avg_epoch_time),
            eta_run=format_duration(eta_run),
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            val_acc=val_stats["acc"],
            val_nll=val_stats["nll"],
            lr_before=lr_before,
            lr_after=optimizer.param_groups[0]["lr"],
        )

        progress.log(
            f"[run {run_index}/{total_runs}] {exp_name} | {method} | seed={seed} | "
            f"epoch={epoch:03d}/{int(cfg['epochs']):03d} | "
            f"val_acc={val_stats['acc']:.4f} | val_nll={val_stats['nll']:.4f} | "
            f"rbar={sig['reliability_rbar']:.3f} | lr={lr_before:.5g}->{optimizer.param_groups[0]['lr']:.5g} | "
            f"epoch_time={format_duration(epoch_time)} | avg_epoch={format_duration(avg_epoch_time)} | "
            f"eta_run={format_duration(eta_run)} | best_val_acc={best_val_acc:.4f}@{best_epoch}"
        )

        # fixed 模式保留控制信号和动作记录，但不允许提前缩短公平训练预算。
        if action["early_stop"] and str(cfg.get("budget_mode", "adaptive")).lower() != "fixed":
            progress.log(
                f"[run {run_index}/{total_runs}] early stop | experiment={exp_name} | "
                f"method={method} | seed={seed} | epoch={epoch} | reason={action['stop_reason']}"
            )
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    # 恢复验证集选出的最佳状态后再触碰测试集，阻断测试信息泄漏。
    if best_state is not None:
        checkpoint_start = time.perf_counter()
        model.load_state_dict(best_state)
        torch.save(
            {"epoch": best_epoch, "model_state_dict": best_state, "config_hash": config_hash},
            out_dir / "best_checkpoint.pt",
        )
        checkpoint_seconds += time.perf_counter() - checkpoint_start

    metadata["checkpoint_selected_before_test"] = best_state is not None
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    progress.log(
        f"[run {run_index}/{total_runs}] test start | experiment={exp_name} | method={method} | seed={seed}"
    )
    progress.update_current(
        phase="testing",
        batch=0,
        total_batches=len(test_loader),
    )
    # 每个运行只执行这一次测试评估，次数和时间戳同时写入元数据。
    test_stats = evaluate(
        model, test_loader, criterion, device, return_logits=True,
        progress=progress,
        context=f"[run {run_index}/{total_runs}] {exp_name} | {method} | seed={seed} | test",
    )
    logits = test_stats.pop("logits")
    targets = test_stats.pop("targets")
    metadata["test_evaluation_count"] = 1
    metadata["test_evaluation_timestamp"] = datetime.now().astimezone().isoformat()

    torch.save(logits, out_dir / "test_logits.pt")
    torch.save(targets, out_dir / "test_targets.pt")

    pd.DataFrame(reliability_bins(logits, targets)).to_csv(out_dir / "calibration_bins.csv", index=False)
    cov, risk = risk_coverage_curve(logits, targets)
    pd.DataFrame({"coverage": cov, "risk": risk}).to_csv(out_dir / "risk_coverage_curve.csv", index=False)

    # summary.json 汇总论文表格所需指标、资源开销和完整复现证据。
    summary = {
        "experiment": exp_name,
        "experiment_type": exp_type,
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
        "controller_alpha": float(cfg["controller"]["alpha"]),
        "controller_beta": float(cfg["controller"]["beta"]),
        "pb_final": float(hist_df["pb_signal"].iloc[-1]),
        "pd_final": float(hist_df["pd_signal"].iloc[-1]),
        "rbar_final": float(hist_df["reliability_rbar"].iloc[-1]),
        "lr_reductions": int(hist_df["lr_reduced"].sum()),
        "first_lr_drop_epoch": int(hist_df.loc[hist_df["lr_reduced"] == True, "epoch"].min()) if hist_df["lr_reduced"].any() else -1,
        "runtime_seconds": float(time.time() - start),
        "training_wall_clock_seconds": float(sum(epoch_times)),
        "avg_seconds_per_epoch": float(sum(epoch_times) / max(1, len(epoch_times))),
        "gpu_hours": float((time.time() - start) / 3600.0) if device.type == "cuda" else 0.0,
        "peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        "peak_gpu_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        "controller_total_seconds": float(sum(controller_times)),
        "controller_avg_seconds_per_epoch": float(sum(controller_times) / max(1, len(controller_times))),
        "pb_computation_seconds": float(hist_df["pb_computation_seconds"].sum()),
        "pd_computation_seconds": float(hist_df["pd_computation_seconds"].sum()),
        "standardization_smoothing_seconds": float(hist_df["standardization_smoothing_seconds"].sum()),
        "controller_decision_seconds": float(hist_df["controller_decision_seconds"].sum()),
        "checkpoint_overhead_seconds": float(checkpoint_seconds),
        "config_hash": config_hash,
        "scientific_config_hash": scientific_config_hash,
        "pretrained_checkpoint_sha256": pretrained_checkpoint_hash,
        "split_hash": split_hash,
        "initialization_hash": initialization_hash,
        "git_commit": commit,
        "budget_mode": cfg.get("budget_mode", "adaptive"),
        "data_protocol_version": cfg.get("data_protocol_version", "legacy_augmented_validation_v1"),
        "final_lr": float(hist_df["lr_after"].iloc[-1]),
        "stopping_epoch": int(hist_df["epoch"].iloc[-1]),
        "test_evaluation_count": 1,
        "test_evaluation_timestamp": metadata["test_evaluation_timestamp"],
        "checkpoint_selected_before_test": metadata["checkpoint_selected_before_test"],
        "test_used_during_training": False,
        "status": "COMPLETED",
        "run_id": run_id or "",
    }
    summary["controller_overhead_percent"] = (
        100.0 * summary["controller_total_seconds"]
        / max(summary["training_wall_clock_seconds"], 1e-12)
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata["status"] = "COMPLETED"
    metadata["completed_at"] = datetime.now().astimezone().isoformat()
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress.log(
        f"[run {run_index}/{total_runs}] done | experiment={exp_name} | method={method} | seed={seed} | "
        f"epochs_run={summary['epochs_run']} | test_acc={summary['test_acc']:.4f} | "
        f"runtime={format_duration(summary['runtime_seconds'])}"
    )
    progress.record_finished({
        "run_index": run_index,
        "experiment": exp_name,
        "method": method,
        "seed": seed,
        "status": "completed",
        "test_acc": summary["test_acc"],
        "runtime": format_duration(summary["runtime_seconds"]),
    })
    return summary


def main():
    """解析命令行、审计运行矩阵并顺序执行配置中的全部方法与种子。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--progress",
        choices=["none", "epoch", "batch"],
        default="batch",
        help="Progress verbosity. 'epoch' prints per-epoch ETA; 'batch' also prints batch heartbeat.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="When --progress=batch, print at least every N batches.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=30.0,
        help="When --progress=batch, print at least every N seconds.",
    )
    parser.add_argument(
        "--progress-log",
        default="",
        help="Optional path to append progress messages, e.g. results/progress.log.",
    )
    parser.add_argument(
        "--progress-state",
        default="",
        help="Path for live JSON state. Defaults to <results_dir>/progress_state.json.",
    )
    parser.add_argument(
        "--dashboard-path",
        default="",
        help="Path for the live HTML dashboard. Defaults to <results_dir>/progress_dashboard.html.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not write the live HTML dashboard.",
    )
    parser.add_argument(
        "--existing",
        choices=["skip", "rerun-partial", "error", "archive", "overwrite"],
        default="error",
        help=(
            "Policy when result artifacts already exist for a method/seed. "
            "'rerun-partial' skips completed runs but archives incomplete outputs before rerunning. "
            "Default aborts to avoid overwrites and silent skips."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="Override cfg['results_dir']; useful for isolated experiment batches.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="",
        help="Optional prefix added to cfg['experiment_name'] before writing outputs.",
    )
    parser.add_argument(
        "--experiment-suffix",
        default="",
        help="Optional suffix added to cfg['experiment_name'] before writing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only audit planned runs and existing outputs; do not train.",
    )
    args = parser.parse_args()

    progress = ProgressLogger(
        mode=args.progress,
        interval=args.progress_interval,
        seconds=args.progress_seconds,
        log_path=args.progress_log or None,
    )
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg["experiment_type"] = str(cfg.get("experiment_type", "unspecified") or "unspecified")
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.experiment_prefix or args.experiment_suffix:
        cfg["experiment_name"] = f"{args.experiment_prefix}{cfg['experiment_name']}{args.experiment_suffix}"
    all_rows = []
    run_id = make_run_id()
    total_runs = len(cfg["methods"]) * len(cfg["seeds"])
    state_path = Path(args.progress_state) if args.progress_state else Path(cfg["results_dir"]) / "progress_state.json"
    dashboard_path = None
    if not args.no_dashboard:
        dashboard_path = Path(args.dashboard_path) if args.dashboard_path else Path(cfg["results_dir"]) / "progress_dashboard.html"
    progress.configure_state(state_path, dashboard_path)
    # 在启动首个训练前完成全矩阵碰撞审计，并同步展示到仪表盘。
    audit = planned_runs(cfg, args.existing)
    batch_note = ""
    if progress.batch_enabled():
        batch_note = f" | batch_interval={progress.interval} | seconds_interval={progress.seconds:g}"
    progress.update_state(
        status="dry_run" if args.dry_run else "running",
        run_id=run_id,
        config_path=args.config,
        experiment=cfg["experiment_name"],
        experiment_type=cfg["experiment_type"],
        existing_policy=args.existing,
        total_runs=total_runs,
        planned_runs=audit,
        completed_runs=0,
    )
    progress.log(
        f"[config] {args.config} | experiment={cfg['experiment_name']} | "
        f"methods={len(cfg['methods'])} | seeds={len(cfg['seeds'])} | total_runs={total_runs} | "
        f"epochs_per_run={int(cfg['epochs'])} | progress={args.progress}{batch_note} | "
        f"existing={args.existing} | run_id={run_id}"
    )
    if dashboard_path:
        progress.log(
            f"[dashboard] file={dashboard_path} | state={state_path} | "
            "serve with: python -m http.server 8000 -d results"
        )
    collisions = [r for r in audit if r["artifacts"]]
    if collisions:
        progress.log(
        f"[existing-results] found={len(collisions)} | policy={args.existing} | "
            "default error aborts before overwriting or skipping artifacts"
        )
        for row in collisions:
            progress.log(
                f"[existing-results] run={row['run_index']}/{total_runs} | "
                f"{row['experiment']} | {row['method']} | seed={row['seed']} | "
                f"status={row['status']} | artifacts={','.join(row['artifacts'])}"
            )
    else:
        progress.log("[existing-results] no existing result artifacts for this config")

    if args.dry_run:
        progress.update_state(status="completed")
        progress.close()
        return

    try:
        run_index = 0
        for method in cfg["methods"]:
            for seed in cfg["seeds"]:
                run_index += 1
                row = run_one(
                    cfg,
                    method,
                    int(seed),
                    progress,
                    run_index,
                    total_runs,
                    existing_policy=args.existing,
                    run_id=run_id,
                )
                if row is not None:
                    all_rows.append(row)

        # 每次运行先写独立目录；批次结束后再去重合并实验级和全局摘要。
        exp_dir = Path(cfg["results_dir"]) / cfg["experiment_name"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(all_rows)
        summary_by_seed_path = exp_dir / "summary_by_seed.csv"
        if summary_by_seed_path.exists() and not new_df.empty:
            old_summary = pd.read_csv(summary_by_seed_path)
            summary_by_seed = pd.concat([old_summary, new_df], ignore_index=True)
            keep = "last" if args.existing in ["rerun-partial", "archive", "overwrite"] else "first"
            summary_by_seed = summary_by_seed.drop_duplicates(["experiment", "method", "seed"], keep=keep)
        elif summary_by_seed_path.exists():
            summary_by_seed = pd.read_csv(summary_by_seed_path)
        else:
            summary_by_seed = new_df
        summary_by_seed.to_csv(summary_by_seed_path, index=False)

        master_path = Path(cfg["results_dir"]) / "master_summary.csv"
        if master_path.exists():
            old = pd.read_csv(master_path)
            if "experiment_type" not in old.columns and "experiment" in old.columns:
                old["experiment_type"] = old["experiment"].map(infer_experiment_type)
            elif "experiment_type" in old.columns and "experiment" in old.columns:
                missing = old["experiment_type"].isna() | (old["experiment_type"].astype(str).str.strip() == "")
                old.loc[missing, "experiment_type"] = old.loc[missing, "experiment"].map(infer_experiment_type)
            combined = pd.concat([old, new_df], ignore_index=True)
            keep = "last" if args.existing in ["rerun-partial", "archive", "overwrite"] else "first"
            combined = combined.drop_duplicates(["experiment", "method", "seed"], keep=keep)
        else:
            combined = new_df
        combined.to_csv(master_path, index=False)
        progress.update_state(status="completed")
    except Exception as exc:
        # 即使异常中止，也保留失败位置与异常文本，便于续跑审计定位。
        progress.update_state(status="error", error=str(exc))
        if "method" in locals() and "seed" in locals():
            failed_dir = planned_output_dir(cfg, method, int(seed))
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_dir / "run_metadata.json"
            try:
                failed = json.loads(failed_path.read_text(encoding="utf-8")) if failed_path.exists() else {}
            except Exception:
                failed = {}
            failed.update({
                "dataset": cfg.get("dataset", "UNRESOLVED"),
                "model": cfg.get("model", "UNRESOLVED"),
                "method": method,
                "seed": int(seed),
                "status": "FAILED",
                "exception": repr(exc),
                "last_completed_epoch": int(progress.state.get("current", {}).get("epoch", 0) or 0),
                "checkpoint_path": str(failed_dir / "best_checkpoint.pt"),
                "timestamp": datetime.now().astimezone().isoformat(),
            })
            failed_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()
