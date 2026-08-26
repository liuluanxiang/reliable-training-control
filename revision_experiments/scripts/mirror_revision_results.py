"""将旧补充实验目录中的完整运行迁移或复制到统一 ``results_v2``。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "revision_experiments" / "results"
MIRROR_ROOT = ROOT / "results_v2"
STATE_PATH = SOURCE_ROOT / "results_v2_mirror_state.json"
REQUIRED_ARTIFACTS = {
    "best_checkpoint.pt",
    "config_used.json",
    "run_metadata.json",
    "history.csv",
    "summary.json",
    "test_logits.pt",
    "test_targets.pt",
    "calibration_bins.csv",
    "risk_coverage_curve.csv",
}


def read_json(path: Path):
    """读取 UTF-8 JSON 摘要。"""

    return json.loads(path.read_text(encoding="utf-8"))


def complete_artifacts(directory: Path):
    """检查运行目录是否包含协议要求的完整九项产物。"""

    return REQUIRED_ARTIFACTS.issubset(path.name for path in directory.iterdir() if path.is_file())


def same_result(destination: Path, scientific_hash: str):
    """通过科学配置哈希判断目标目录是否已含同一结果。"""

    if not destination.exists() or not complete_artifacts(destination):
        return False
    try:
        return read_json(destination / "summary.json").get("scientific_config_hash") == scientific_hash
    except (OSError, json.JSONDecodeError):
        return False


def write_state(payload):
    """写出迁移进度状态，供诊断与仪表盘使用。"""

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)


def mirror_once(move=False):
    """执行一轮完整结果扫描，并仅迁移可验证完成的运行。"""

    MIRROR_ROOT.mkdir(parents=True, exist_ok=True)
    mirrored = []
    already_present = []
    incomplete = []
    conflicts = []
    errors = []
    candidates = 0

    for summary_path in SOURCE_ROOT.rglob("summary.json"):
        if any(part in {"_archive", "smoke"} for part in summary_path.parts):
            continue
        source = summary_path.parent
        try:
            summary = read_json(summary_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"source": str(source), "error": str(exc)})
            continue
        if (
            summary.get("data_protocol_version") != "deterministic_validation_v2"
            or summary.get("status") != "COMPLETED"
        ):
            continue
        candidates += 1
        if not complete_artifacts(source):
            incomplete.append(str(source))
            continue
        experiment = str(summary.get("experiment", "")).strip()
        method = str(summary.get("method", "")).strip()
        seed = summary.get("seed")
        scientific_hash = str(summary.get("scientific_config_hash", "")).strip()
        if not experiment or not method or seed is None or not scientific_hash:
            errors.append({"source": str(source), "error": "missing result identity/hash"})
            continue
        destination = MIRROR_ROOT / experiment / method / f"seed_{int(seed)}"
        if same_result(destination, scientific_hash):
            already_present.append(str(destination))
            continue
        if destination.exists():
            conflicts.append({
                "source": str(source), "destination": str(destination),
                "error": "destination exists but is incomplete or has a different scientific hash",
            })
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if move:
                source.rename(destination)
            else:
                temporary = destination.with_name(f".{destination.name}.mirroring_{time.time_ns()}")
                shutil.copytree(source, temporary)
                if not complete_artifacts(temporary):
                    raise RuntimeError("temporary mirror failed the nine-artifact completeness check")
                temporary.rename(destination)
            mirrored.append(str(destination))
        except Exception as exc:
            errors.append({"source": str(source), "destination": str(destination), "error": str(exc)})

    state = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "source_root": str(SOURCE_ROOT),
        "mirror_root": str(MIRROR_ROOT),
        "operation": "move" if move else "copy",
        "candidates": candidates,
        "mirrored_this_pass": len(mirrored),
        "complete_mirrors": len(mirrored) + len(already_present),
        "incomplete_sources": incomplete,
        "conflicts": conflicts,
        "errors": errors,
        "latest_mirrored": mirrored[-10:],
        "status": "ok" if not conflicts and not errors else "attention",
    }
    write_state(state)
    return state


def main():
    """解析单次/持续模式以及复制/移动策略。"""

    parser = argparse.ArgumentParser(description="Atomically mirror complete protocol-v2 runs to results_v2.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--move", action="store_true", help="Move complete source directories instead of copying.")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    if not args.once and not args.watch:
        args.once = True
    while True:
        state = mirror_once(move=args.move)
        print(json.dumps({
            "updated_at": state["updated_at"], "status": state["status"],
            "candidates": state["candidates"], "complete_mirrors": state["complete_mirrors"],
            "mirrored_this_pass": state["mirrored_this_pass"],
            "conflicts": len(state["conflicts"]), "errors": len(state["errors"]),
        }), flush=True)
        if args.once:
            break
        time.sleep(max(2.0, args.interval))


if __name__ == "__main__":
    main()
