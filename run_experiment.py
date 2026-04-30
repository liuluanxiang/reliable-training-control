import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
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
    "history.csv",
    "summary.json",
    "test_logits.pt",
    "test_targets.pt",
    "calibration_bins.csv",
    "risk_coverage_curve.csv",
]


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds):
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
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def result_artifacts(out_dir):
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
    path = Path(out_dir) / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def planned_output_dir(cfg, method, seed):
    return Path(cfg["results_dir"]) / cfg["experiment_name"] / method / f"seed_{seed}"


def planned_runs(cfg, existing_policy):
    runs = []
    run_index = 0
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            run_index += 1
            out_dir = planned_output_dir(cfg, method, int(seed))
            artifacts = result_artifacts(out_dir)
            has_summary = "summary.json" in artifacts
            if not artifacts:
                status = "will_run"
            elif existing_policy == "skip":
                status = "skip_completed" if has_summary else "skip_partial"
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
    dashboard_path = Path(dashboard_path)
    state_path = Path(state_path)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    rel_state = os.path.relpath(state_path, dashboard_path.parent).replace("\\", "/")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment Progress</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #637083;
      --line: #d8dee8;
      --accent: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --good: #047857;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111417;
        --panel: #1a2027;
        --ink: #eef2f7;
        --muted: #a8b3c2;
        --line: #334155;
        --accent: #2dd4bf;
        --warn: #f59e0b;
        --bad: #f87171;
        --good: #34d399;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 24px;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      font-size: 20px;
      margin: 0 0 6px;
      font-weight: 700;
    }}
    main {{ padding: 20px 24px 32px; max-width: 1440px; margin: 0 auto; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .wide {{ grid-column: span 2; }}
    .full {{ grid-column: 1 / -1; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .value {{ font-size: 22px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .bar {{
      height: 12px;
      background: color-mix(in srgb, var(--line) 75%, transparent);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }}
    .fill {{ height: 100%; width: 0; background: var(--accent); transition: width .25s ease; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }}
    .status-run {{ color: var(--accent); font-weight: 700; }}
    .status-done {{ color: var(--good); font-weight: 700; }}
    .status-skip {{ color: var(--warn); font-weight: 700; }}
    .status-error {{ color: var(--bad); font-weight: 700; }}
    .messages {{
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
      background: color-mix(in srgb, var(--panel) 80%, var(--bg));
    }}
    @media (max-width: 900px) {{
      main {{ padding: 14px; }}
      .grid {{ grid-template-columns: 1fr 1fr; }}
      .wide {{ grid-column: 1 / -1; }}
    }}
    @media (max-width: 560px) {{
      .grid {{ grid-template-columns: 1fr; }}
      header {{ padding: 12px 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Experiment Progress</h1>
    <div class="muted" id="subtitle">Waiting for progress_state.json</div>
  </header>
  <main>
    <section class="grid">
      <div class="panel"><div class="label">Status</div><div class="value" id="status">-</div></div>
      <div class="panel"><div class="label">Run</div><div class="value" id="run">-</div><div class="bar"><div class="fill" id="runBar"></div></div></div>
      <div class="panel"><div class="label">Epoch</div><div class="value" id="epoch">-</div><div class="bar"><div class="fill" id="epochBar"></div></div></div>
      <div class="panel"><div class="label">Batch</div><div class="value" id="batch">-</div><div class="bar"><div class="fill" id="batchBar"></div></div></div>
      <div class="panel"><div class="label">Epoch Time</div><div class="value" id="epochTime">-</div></div>
      <div class="panel"><div class="label">Average Epoch</div><div class="value" id="avgEpoch">-</div></div>
      <div class="panel"><div class="label">ETA Current Run</div><div class="value" id="etaRun">-</div></div>
      <div class="panel"><div class="label">Best Validation</div><div class="value" id="bestVal">-</div></div>
      <div class="panel wide"><div class="label">Current</div><div class="value" id="current">-</div></div>
      <div class="panel wide"><div class="label">Last Update</div><div class="value" id="updated">-</div></div>
      <div class="panel full">
        <div class="label">Planned Runs and Collision Policy</div>
        <div id="plan"></div>
      </div>
      <div class="panel full">
        <div class="label">Recent Messages</div>
        <div class="messages" id="messages"></div>
      </div>
    </section>
  </main>
  <script>
    const stateUrl = "{rel_state}";
    const fmt = (v) => (v === undefined || v === null || v === "" ? "-" : v);
    const pct = (a, b) => b ? Math.max(0, Math.min(100, 100 * a / b)) : 0;
    function setText(id, value) {{ document.getElementById(id).textContent = fmt(value); }}
    function setBar(id, value) {{ document.getElementById(id).style.width = value.toFixed(1) + "%"; }}
    function statusClass(status) {{
      status = status || "";
      if (status === "completed" || status.includes("done")) return "status-done";
      if (status.includes("error")) return "status-error";
      if (status.includes("skip") || status.includes("archive") || status.includes("overwrite")) return "status-skip";
      return "status-run";
    }}
    function render(data) {{
      const current = data.current || {{}};
      const runIndex = current.run_index || data.completed_runs || 0;
      const totalRuns = data.total_runs || 0;
      setText("subtitle", `${{data.experiment || "-"}} | config=${{data.config_path || "-"}} | policy=${{data.existing_policy || "-"}}`);
      setText("status", data.status || "-");
      document.getElementById("status").className = "value " + statusClass(data.status);
      setText("run", `${{runIndex}}/${{totalRuns}}`);
      setBar("runBar", pct(runIndex, totalRuns));
      setText("epoch", current.epoch && current.total_epochs ? `${{current.epoch}}/${{current.total_epochs}}` : "-");
      setBar("epochBar", pct(current.epoch || 0, current.total_epochs || 0));
      setText("batch", current.batch && current.total_batches ? `${{current.batch}}/${{current.total_batches}}` : fmt(current.phase));
      setBar("batchBar", pct(current.batch || 0, current.total_batches || 0));
      setText("epochTime", current.epoch_time || "-");
      setText("avgEpoch", current.avg_epoch_time || "-");
      setText("etaRun", current.eta_run || "-");
      setText("bestVal", current.best_val_acc !== undefined ? `${{Number(current.best_val_acc).toFixed(4)}} @ ${{current.best_epoch || "-"}}` : "-");
      setText("current", current.experiment ? `${{current.experiment}} | ${{current.method}} | seed=${{current.seed}} | ${{current.phase || "-"}}` : "-");
      setText("updated", data.updated_at || "-");
      const planned = data.planned_runs || [];
      document.getElementById("plan").innerHTML = `<table><thead><tr><th>#</th><th>Experiment</th><th>Method</th><th>Seed</th><th>Status</th><th>Artifacts</th></tr></thead><tbody>${{planned.map(r => `<tr><td>${{r.run_index}}</td><td><code>${{r.experiment}}</code></td><td>${{r.method}}</td><td>${{r.seed}}</td><td class="${{statusClass(r.status)}}">${{r.status}}</td><td>${{(r.artifacts || []).join(", ") || "-"}}</td></tr>`).join("")}}</tbody></table>`;
      document.getElementById("messages").textContent = (data.messages || []).join("\\n");
    }}
    async function tick() {{
      try {{
        const response = await fetch(stateUrl + "?t=" + Date.now(), {{ cache: "no-store" }});
        render(await response.json());
      }} catch (err) {{
        setText("status", "waiting");
        setText("messages", "Serve the results directory to enable live refresh, for example: python -m http.server 8000 -d results");
      }}
    }}
    tick();
    setInterval(tick, 2000);
  </script>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")


class ProgressLogger:
    def __init__(self, mode="epoch", interval=50, seconds=30.0, log_path=None):
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
        return self.mode != "none"

    def batch_enabled(self):
        return self.mode == "batch"

    def configure_state(self, state_path, dashboard_path=None):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if dashboard_path:
            self.dashboard_path = Path(dashboard_path)
            write_dashboard(self.dashboard_path, self.state_path)
        self.write_state()

    def update_state(self, **kwargs):
        self.state.update(kwargs)
        self.state["updated_at"] = now_text()
        self.write_state()

    def update_current(self, **kwargs):
        current = dict(self.state.get("current", {}))
        current.update(kwargs)
        self.state["current"] = current
        self.state["updated_at"] = now_text()
        self.write_state()

    def record_finished(self, row):
        finished = self.state.setdefault("finished_runs", [])
        finished.append(row)
        self.state["completed_runs"] = len(finished)
        self.state["updated_at"] = now_text()
        self.write_state()

    def write_state(self):
        if self.state_path is None:
            return
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def log(self, message):
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
        if self._fh is not None:
            self._fh.close()
            self._fh = None


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


def is_reliability_controller_method(method):
    return method == "reliability" or str(method).startswith("reliability_")


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


def train_one_epoch(model, loader, optimizer, criterion, device, signals, progress=None, context=""):
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

    if artifacts:
        if existing_policy == "skip":
            existing_summary = load_existing_summary(out_dir)
            if existing_summary is not None:
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

            progress.log(
                f"[run {run_index}/{total_runs}] skip partial existing output | "
                f"experiment={exp_name} | method={method} | seed={seed} | "
                f"artifacts={','.join(artifacts)} | use --existing archive to preserve and rerun"
            )
            progress.record_finished({
                "run_index": run_index,
                "experiment": exp_name,
                "method": method,
                "seed": seed,
                "status": "skipped_partial",
            })
            return None

        if existing_policy == "error":
            raise FileExistsError(
                f"Existing result artifacts found in {out_dir}: {artifacts}. "
                "Use --existing skip, --existing archive, or --existing overwrite."
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
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = resolve_device(cfg)

    train_loader, val_loader, test_loader, num_classes = build_loaders(
        cfg["dataset"],
        cfg["data_root"],
        int(cfg["batch_size"]),
        int(cfg["num_workers"]),
        float(cfg["val_ratio"]),
        seed,
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
    epoch_times = []

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
            "bad_val": 0,
            "first_lr_drop_epoch": -1,
            "lr_reductions": 0,
        }

        if is_reliability_controller_method(method):
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
            "experiment_type": exp_type,
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

        if action["early_stop"]:
            progress.log(
                f"[run {run_index}/{total_runs}] early stop | experiment={exp_name} | "
                f"method={method} | seed={seed} | epoch={epoch} | reason={action['stop_reason']}"
            )
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    progress.log(
        f"[run {run_index}/{total_runs}] test start | experiment={exp_name} | method={method} | seed={seed}"
    )
    progress.update_current(
        phase="testing",
        batch=0,
        total_batches=len(test_loader),
    )
    test_stats = evaluate(
        model, test_loader, criterion, device, return_logits=True,
        progress=progress,
        context=f"[run {run_index}/{total_runs}] {exp_name} | {method} | seed={seed} | test",
    )
    logits = test_stats.pop("logits")
    targets = test_stats.pop("targets")

    torch.save(logits, out_dir / "test_logits.pt")
    torch.save(targets, out_dir / "test_targets.pt")

    pd.DataFrame(reliability_bins(logits, targets)).to_csv(out_dir / "calibration_bins.csv", index=False)
    cov, risk = risk_coverage_curve(logits, targets)
    pd.DataFrame({"coverage": cov, "risk": risk}).to_csv(out_dir / "risk_coverage_curve.csv", index=False)

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
        "run_id": run_id or "",
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
        choices=["skip", "error", "archive", "overwrite"],
        default="error",
        help="Policy when result artifacts already exist for a method/seed. Default aborts to avoid overwrites and silent skips.",
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

        exp_dir = Path(cfg["results_dir"]) / cfg["experiment_name"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(all_rows)
        summary_by_seed_path = exp_dir / "summary_by_seed.csv"
        if summary_by_seed_path.exists() and not new_df.empty:
            old_summary = pd.read_csv(summary_by_seed_path)
            summary_by_seed = pd.concat([old_summary, new_df], ignore_index=True)
            keep = "last" if args.existing in ["archive", "overwrite"] else "first"
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
            keep = "last" if args.existing in ["archive", "overwrite"] else "first"
            combined = combined.drop_duplicates(["experiment", "method", "seed"], keep=keep)
        else:
            combined = new_df
        combined.to_csv(master_path, index=False)
        progress.update_state(status="completed")
    except Exception as exc:
        progress.update_state(status="error", error=str(exc))
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    main()
