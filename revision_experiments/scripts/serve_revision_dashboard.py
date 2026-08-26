"""提供补充实验实时仪表盘与状态 API。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision_experiments"
RESULTS = REVISION / "results"
DASHBOARD = REVISION / "dashboard" / "index.html"
GPU_CACHE = {"at": 0.0, "value": {}}
GPU_LOCK = threading.Lock()
MANIFEST_CACHE = {"mtime": None, "rows": []}


def read_json(path: Path, default):
    """容错读取 JSON 状态，失败时返回调用方默认值。"""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_csv(path: Path):
    """容错读取 CSV 为字典列表。"""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def tail(path: Path, count: int = 160):
    """读取日志末尾有限行，控制 API 响应大小。"""

    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]
    except OSError:
        return []


def manifest_rows():
    """按文件修改时间缓存并返回正式队列清单。"""

    path = RESULTS / "run_manifest.csv"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return []
    if MANIFEST_CACHE["mtime"] == mtime:
        return MANIFEST_CACHE["rows"]
    rows = read_csv(path)
    for row in rows:
        try:
            cfg = read_json(ROOT / row["config"], {})
            row["output_dir"] = str(
                ROOT / cfg["results_dir"] / cfg["experiment_name"]
                / row["method"] / f"seed_{row['seed']}"
            )
        except (KeyError, TypeError):
            row["output_dir"] = ""
    MANIFEST_CACHE.update({"mtime": mtime, "rows": rows})
    return rows


def gpu_snapshot():
    """读取并短时缓存 GPU 利用率、显存、温度和功耗。"""

    with GPU_LOCK:
        now = time.monotonic()
        if now - GPU_CACHE["at"] < 4:
            return GPU_CACHE["value"]
        command = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
            fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            value = {
                "name": fields[0], "utilization_percent": float(fields[1]),
                "memory_used_mb": float(fields[2]), "memory_total_mb": float(fields[3]),
                "temperature_c": float(fields[4]), "power_w": float(fields[5]),
                "power_limit_w": float(fields[6]),
            }
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            value = {"error": "GPU telemetry unavailable"}
        GPU_CACHE.update({"at": now, "value": value})
        return value


def runtime_snapshot(manifest):
    """根据清单和最新工期估算表计算队列完成度。"""

    rows = read_csv(RESULTS / "runtime_estimate.csv")
    remaining = 0.0
    resolved = True
    for row in rows:
        try:
            remaining += float(row.get("remaining_gpu_hours", ""))
        except ValueError:
            resolved = False
    active_elapsed = 0.0
    active = next((row for row in manifest if row.get("status") == "RUNNING"), None)
    if active and active.get("started_at"):
        try:
            active_elapsed = max(
                0.0,
                (datetime.now().astimezone() - datetime.fromisoformat(active["started_at"])).total_seconds()
                / 3600.0,
            )
        except ValueError:
            active_elapsed = 0.0
    adjusted_remaining = max(0.0, remaining - active_elapsed) if resolved else None
    completion = (
        datetime.now().astimezone() + timedelta(hours=adjusted_remaining)
        if adjusted_remaining is not None else None
    )
    return {
        "groups": rows,
        "remaining_gpu_hours": adjusted_remaining,
        "active_elapsed_gpu_hours": active_elapsed,
        "continuous_completion": completion.isoformat() if completion else None,
    }


def build_status():
    """合并训练状态、队列、日志尾部和硬件状态为 API 响应。"""

    rows = manifest_rows()
    current_state = read_json(RESULTS / "current_run_state.json", {})
    active = next((row for row in rows if row.get("status") == "RUNNING"), None)
    counts = {status: sum(row.get("status") == status for row in rows)
              for status in ["PENDING", "RUNNING", "COMPLETED", "FAILED"]}
    groups = []
    for name in ["main", "pd", "budget", "sigma", "tiny_imagenet", "deit_cifar100"]:
        selected = [row for row in rows if row.get("group") == name]
        groups.append({
            "name": name, "total": len(selected),
            "pending": sum(row.get("status") == "PENDING" for row in selected),
            "running": sum(row.get("status") == "RUNNING" for row in selected),
            "completed": sum(row.get("status") == "COMPLETED" for row in selected),
            "failed": sum(row.get("status") == "FAILED" for row in selected),
        })
    current = dict(current_state.get("current") or {})
    if active:
        current.update({
            "global_run_index": int(active["run_index"]), "global_total_runs": len(rows),
            "group": active["group"], "dataset": active["dataset"],
            "model": active["model"], "method": active["method"],
            "seed": int(active["seed"]), "output_dir": active.get("output_dir", ""),
        })
    epoch_fraction = 0.0
    try:
        epoch_fraction = float(current.get("epoch", 0)) / max(float(current.get("total_epochs", 1)), 1)
    except (TypeError, ValueError):
        pass
    total = max(len(rows), 1)
    overall = (counts["COMPLETED"] + (epoch_fraction if active else 0.0)) / total
    status = "running" if active else ("completed" if rows and counts["COMPLETED"] == len(rows) else "idle")
    if counts["FAILED"] and not active:
        status = "attention"
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status, "counts": counts, "total": len(rows),
        "overall_fraction": overall, "groups": groups, "current": current,
        "current_state": current_state, "manifest": rows,
        "runtime": runtime_snapshot(rows), "gpu": gpu_snapshot(),
        "queue_log": tail(RESULTS / "revision_queue.out.log"),
        "queue_errors": tail(RESULTS / "revision_queue.err.log", 80),
        "current_log": tail(RESULTS / "current_run.log"),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    """同时提供静态前端和动态 ``/api/status`` JSON。"""

    def do_GET(self):
        """返回实时状态 API 和仪表盘首页，拒绝其他路径。"""

        route = urlparse(self.path).path
        if route == "/api/status":
            payload = json.dumps(build_status(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if route in {"/", "/index.html"}:
            payload = DASHBOARD.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def log_message(self, _format, *_args):
        """抑制高频轮询访问日志，保持实验日志可读。"""

        return


def main():
    """启动线程化本地 HTTP 服务。"""

    parser = argparse.ArgumentParser(description="Serve the global Paper 1 revision dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not DASHBOARD.exists():
        raise FileNotFoundError(DASHBOARD)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
