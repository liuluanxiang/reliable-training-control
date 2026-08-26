"""审计历史实验可复用性、复现哈希、协议缺口并生成补跑配置。"""

import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from src.models import build_model
from src.reproducibility import set_seed


REVISION = ROOT / "revision_experiments"
AUDIT = REVISION / "audit"
CONFIGS = REVISION / "configs"
RESULTS = REVISION / "results"
MAIN_CONFIGS = {
    ("cifar10", "resnet18"): "configs/cifar10_resnet18_main.json",
    ("cifar100", "resnet18"): "configs/cifar100_resnet18_main.json",
    ("cifar10", "resnet50"): "configs/cifar10_resnet50_supplement.json",
    ("cifar100", "resnet50"): "configs/cifar100_resnet50_supplement.json",
    ("cifar10", "vgg16"): "configs/cifar10_vgg16_supplement.json",
    ("cifar100", "vgg16"): "configs/cifar100_vgg16_supplement.json",
}
METHODS = ["step", "cosine", "plateau", "reliability_val_loss_only", "reliability"]
SEEDS = [1, 2, 3, 4, 5]
HISTORICAL_ARTIFACTS = [
    "history.csv", "summary.json", "test_logits.pt", "test_targets.pt",
    "calibration_bins.csv", "risk_coverage_curve.csv",
]


def read_json(path):
    """读取 UTF-8 JSON 配置或结果文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def controller_template():
    """返回审计配置共用的控制器参数模板。"""

    return read_json(ROOT / "configs/cifar10_resnet18_main.json")["controller"]


def sha256_json(value):
    """对规范序列化的 JSON 对象计算 SHA-256。"""

    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def tensor_state_hash(state):
    """计算包含参数结构与字节内容的模型状态哈希。"""

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def reconstructed_hashes():
    """按固定种子重建模型初始化哈希，用于识别历史运行。"""

    split_cache = {}
    initialization_cache = {}
    for dataset, model in MAIN_CONFIGS:
        num_classes = 10 if dataset == "cifar10" else 100
        for seed in SEEDS:
            split_key = (dataset, seed)
            if split_key not in split_cache:
                indices = torch.randperm(50000, generator=torch.Generator().manual_seed(seed)).tolist()
                split_cache[split_key] = sha256_json({
                    "train": indices[:45000], "validation": indices[45000:]
                })
            initialization_key = (dataset, model, seed)
            if initialization_key not in initialization_cache:
                set_seed(seed, deterministic=False)
                instance = build_model(model, num_classes)
                initialization_cache[initialization_key] = tensor_state_hash(instance.state_dict())
                del instance
    return split_cache, initialization_cache


def scientific_config(cfg, method, seed, budget_mode="adaptive"):
    """构造用于匹配历史运行的科学配置投影。"""

    ignored = {"experiment_name", "experiment_type", "methods", "seeds", "results_dir"}
    payload = {key: value for key, value in cfg.items() if key not in ignored}
    payload.update({"method": method, "seed": seed, "budget_mode": budget_mode})
    return payload


def iter_summaries():
    """遍历现有结果根目录中的全部逐运行摘要。"""

    for root_name in ["results_v2", "results"]:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("summary.json"):
            try:
                row = read_json(path)
            except Exception:
                continue
            row["_path"] = path
            row["_root_rank"] = 0 if root_name == "results_v2" else 1
            yield row


def main_candidates(rows):
    """从历史摘要中筛出主实验候选运行。"""

    candidates = {}
    for row in rows:
        key = (row.get("dataset"), row.get("model"), row.get("method"), row.get("seed"))
        if key[:2] not in MAIN_CONFIGS or key[2] not in METHODS or key[3] not in SEEDS:
            continue
        if row.get("experiment_type") not in {"main", "supplement"}:
            continue
        path = row["_path"].parent
        row["_artifact_count"] = sum((path / name).exists() for name in HISTORICAL_ARTIFACTS)
        previous = candidates.get(key)
        score = (row["_artifact_count"], -row["_root_rank"])
        previous_score = ((previous or {}).get("_artifact_count", -1), -(previous or {}).get("_root_rank", 99))
        if previous is None or score > previous_score:
            candidates[key] = row
    return candidates


def write_csv(path, rows, fieldnames):
    """按固定列顺序写出审计 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_reusable(rows):
    """筛选满足完整产物和审计字段要求的可复用运行。"""

    candidates = main_candidates(rows)
    split_hashes, initialization_hashes = reconstructed_hashes()
    reusable = []
    missing = []
    for (dataset, model), config_rel in MAIN_CONFIGS.items():
        cfg = read_json(ROOT / config_rel)
        for method in METHODS:
            for seed in SEEDS:
                key = (dataset, model, method, seed)
                candidate = candidates.get(key)
                path = candidate.get("_path").parent if candidate else None
                missing_artifacts = [name for name in HISTORICAL_ARTIFACTS if path is None or not (path / name).exists()]
                config_hash = sha256_json(scientific_config(cfg, method, seed))
                is_reusable = False
                historical_state = (
                    "no completed historical run" if not candidate else
                    f"historical artifacts missing: {','.join(missing_artifacts)}" if missing_artifacts else
                    "historical artifacts complete"
                )
                reason = (
                    "Invalidated by corrected protocol: historical validation inherited random training "
                    "augmentation, and the shared reliability counter made reliability-based stopping "
                    f"unreachable ({historical_state})."
                )
                reusable.append({
                    "dataset": dataset, "model": model, "method": method, "seed": seed,
                    "config_hash": config_hash,
                    "split_hash": candidate.get("split_hash", split_hashes[(dataset, seed)]) if candidate else split_hashes[(dataset, seed)],
                    "initialization_hash": candidate.get("initialization_hash", initialization_hashes[(dataset, model, seed)]) if candidate else initialization_hashes[(dataset, model, seed)],
                    "checkpoint_path": str(path / "best_checkpoint.pt") if path and (path / "best_checkpoint.pt").exists() else "UNRESOLVED",
                    "metrics_path": str(candidate.get("_path")) if candidate else "UNRESOLVED",
                    "reusable": str(is_reusable).lower(), "reuse_scope": "endpoint_metrics_and_trajectories" if is_reusable else "none",
                    "reason": reason,
                })
                if not is_reusable:
                    missing.append({"experiment_group": "main_matrix", "dataset": dataset, "model": model,
                                    "method": method, "seed": seed, "reason": reason})
    return reusable, missing


def materialize_configs():
    """写出主补跑、消融、敏感性、预算与外部有效性配置。"""

    for (dataset, model), source in MAIN_CONFIGS.items():
        cfg = read_json(ROOT / source)
        cfg.update({
            "experiment_name": f"revision_main_{dataset}_{model}", "experiment_type": "revision_main",
            "methods": METHODS, "seeds": SEEDS, "results_dir": "results_v2",
            "budget_mode": "adaptive", "data_protocol_version": "deterministic_validation_v2",
        })
        path = CONFIGS / "main_5seed" / f"{dataset}_{model}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    base = read_json(ROOT / "configs/cifar10_resnet18_main.json")
    base.update({
        "experiment_name": "revision_pd_component_ablation", "experiment_type": "pd_component_ablation",
        "methods": ["reliability_pd_full", "reliability_pd_v_only", "reliability_pd_v_h",
                    "reliability_pd_v_u", "reliability_pd_h_u"],
        "seeds": SEEDS, "results_dir": "results_v2",
        "budget_mode": "adaptive", "data_protocol_version": "deterministic_validation_v2",
    })
    path = CONFIGS / "pd_ablation" / "cifar10_resnet18.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=2), encoding="utf-8")

    for multiplier in [0.5, 1.0, 2.0, 4.0]:
        cfg = json.loads(json.dumps(base))
        cfg.update({"experiment_name": f"revision_sigma_{str(multiplier).replace('.', 'p')}",
                    "experiment_type": "sigma_sensitivity", "methods": ["reliability"],
                    "results_dir": "results_v2",
                    "data_protocol_version": "deterministic_validation_v2"})
        cfg["controller"]["prior_sigma"] = 0.05 * multiplier
        path = CONFIGS / "sigma_sensitivity" / f"sigma_{str(multiplier).replace('.', 'p')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    main = read_json(ROOT / "configs/cifar10_resnet18_main.json")
    for mode in ["fixed", "adaptive"]:
        cfg = json.loads(json.dumps(main))
        cfg.update({"experiment_name": f"revision_budget_{mode}_cifar10_resnet18",
                    "experiment_type": "budget_comparison", "methods": METHODS, "seeds": SEEDS,
                    "results_dir": "results_v2", "budget_mode": mode,
                    "data_protocol_version": "deterministic_validation_v2"})
        path = CONFIGS / "budget_comparison" / f"cifar10_resnet18_{mode}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    external_specs = [
        {
            "directory": "tiny_imagenet", "experiment_name": "revision_tiny_imagenet_resnet18",
            "experiment_type": "external_tiny_imagenet", "dataset": "tiny_imagenet",
            "model": "resnet18", "epochs": 120, "batch_size": 128, "num_workers": 8,
            "optimizer": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005,
            "image_size": 64, "normalization": "imagenet",
        },
        {
            "directory": "deit_cifar100", "experiment_name": "revision_cifar100_deit_tiny",
            "experiment_type": "external_architecture", "dataset": "cifar100",
            "model": "deit_tiny_patch16_224", "epochs": 100, "batch_size": 128,
            "num_workers": 8, "optimizer": "adamw", "lr": 0.0005, "momentum": 0.9,
            "weight_decay": 0.05, "image_size": 224, "normalization": "imagenet",
            "pretrained_checkpoint": "data/pretrained/deit_tiny_patch16_224-a1311bcf.pth",
        },
    ]
    for spec in external_specs:
        cfg = {
            **{key: value for key, value in spec.items() if key != "directory"},
            "methods": METHODS, "seeds": SEEDS, "val_ratio": 0.1, "data_root": "./data",
            "results_dir": "results_v2", "device": "cuda",
            "deterministic": False, "budget_mode": "adaptive",
            "data_protocol_version": "deterministic_validation_v2", "controller": controller_template(),
        }
        directory = CONFIGS / spec["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "full.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        smoke = json.loads(json.dumps(cfg))
        smoke.update({"experiment_name": f"{cfg['experiment_name']}_smoke", "epochs": 5,
                      "seeds": [1], "results_dir": f"revision_experiments/results/smoke/{spec['directory']}"})
        (directory / "smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")


def write_audit_documents(reusable_count, missing_count):
    """生成仓库审计结论、缺口与后续操作说明。"""

    cfg = read_json(ROOT / "configs/cifar10_resnet18_main.json")
    controller = cfg["controller"]
    (AUDIT / "current_controller_config.json").write_text(json.dumps(controller, indent=2), encoding="utf-8")
    params = [
        ("alpha", controller["alpha"], "global", "configs/*.json; src/signals.py", "historical equal weighting; selection record absent", "unknown"),
        ("beta", controller["beta"], "global", "configs/*.json; src/signals.py", "historical default; 3x3 sensitivity exists", "unknown"),
        ("gamma", controller["gamma"], "global", "configs/*.json; src/controller.py", "historical controller setting", "unknown"),
        ("delta", controller["delta"], "global", "configs/*.json; src/signals.py", "historical PB-inspired formula setting", "unknown"),
        ("sigma", controller["prior_sigma"], "global", "configs/*.json; src/signals.py", "historical default; justification absent", "unknown"),
        ("lambda_g", controller["lambda_g"], "global", "configs/*.json; src/signals.py", "historical scaling; justification absent", "unknown"),
        ("lambda_u", controller["lambda_u"], "global", "configs/*.json; src/signals.py", "historical scaling; justification absent", "unknown"),
        ("min_lr", controller["min_lr"], "global", "configs/*.json; src/controller.py", "historical controller setting", "unknown"),
        ("lr_patience", controller["p_lr"], "global", "configs/*.json; src/controller.py", "historical controller setting", "unknown"),
        ("stopping_patience", controller["p_stop"], "global", "configs/*.json; src/controller.py", "historical controller setting", "unknown"),
        ("epsilon", controller["epsilon"], "global", "configs/*.json; src/controller.py", "historical controller setting", "unknown"),
        ("optimizer", cfg["optimizer"], "global", "configs/*.json", "existing recipe", "unknown"),
        ("initial_lr", cfg["lr"], "global", "configs/*.json", "existing recipe", "unknown"),
        ("weight_decay", cfg["weight_decay"], "global", "configs/*.json", "existing recipe", "unknown"),
        ("batch_size", cfg["batch_size"], "global", "configs/*.json", "existing recipe", "unknown"),
    ]
    write_csv(AUDIT / "controller_hyperparameters.csv", [
        {"parameter": p, "value": v, "scope": scope, "where_defined": where,
         "how_selected": selected, "tuned": tuned, "notes": "No value invented."}
        for p, v, scope, where, selected, tuned in params
    ], ["parameter", "value", "scope", "where_defined", "how_selected", "tuned", "notes"])

    (AUDIT / "current_implementation_audit.md").write_text(f"""# Current Implementation Audit

Generated: {datetime.now().astimezone().isoformat()}

## Controller

- PB signal: `validation_NLL + sqrt((C_t + log(2*sqrt(n)/delta))/(2*n))`, where `C_t = ||theta_t-theta_0||^2/(2*sigma^2)`.
- PD signal: `max(0, validation_NLL-train_NLL) + lambda_g*gradient_norm + lambda_u*update_norm`; gradient norm is sampled on the final training minibatch and update norm is the full epoch parameter displacement.
- Both signals use online sample z-standardization (Welford update), then alpha weighting and beta exponential smoothing.
- LR is multiplied by gamma after `p_lr` non-improving reliability-score epochs. A separate non-resetting reliability-stagnation counter and the validation-stagnation counter independently trigger stopping at `p_stop`.
- Checkpoint selection is maximum validation accuracy; test evaluation occurs once after checkpoint selection.
- Optimizer/config: SGD, lr=0.1, momentum=0.9, weight decay=0.0005, batch=128; max epochs=120 for main runs and 90 for historical ablation/sensitivity.
- Seeds: 1..5 for ResNet-18 main; 1..3 historically for ResNet-50/VGG-16.
- Metrics: ECE uses 15 equal-width confidence bins; Brier is mean multiclass squared probability error; AURC integrates selective risk over coverage.

## Audit Outcome

- Historical main candidates reusable under corrected protocol: {reusable_count}/150.
- Required corrected-protocol main rows: {missing_count}/150.
- All historical results are invalidated for revision inference: validation subsets inherited random crop/flip and the shared reliability counter made reliability-based stopping unreachable. Files remain untouched for provenance.
- New CIFAR and Tiny ImageNet validation subsets use separate dataset objects with deterministic evaluation transforms and seed-reproducible, disjoint indices.
- Baseline runs also compute diagnostic PB/PD trajectories; their recorded wall-clock therefore includes monitoring work.
- The legacy loop redundantly computed gradient norm on every minibatch while retaining only the final value. Protocol v2 computes it only on the final minibatch, preserving the signal definition and removing overwritten work.
- Tiny ImageNet support and a ResNet-18 recipe are present. The official training set supplies optimization/internal validation; the official labeled validation split is reserved for one final test evaluation.
- DeiT-Tiny/16 at 224 pixels is implemented with a local official ImageNet-1k checkpoint and a seed-specific CIFAR-100 classification head.
""", encoding="utf-8")

    (AUDIT / "data_leakage_protocol.md").write_text("""# Data Leakage Protocol

Training data is used for optimization only. Validation data drives PB/PD observations, LR control, stopping, and maximum-validation-accuracy checkpoint selection. Test data is evaluated exactly once after the checkpoint is selected. New runs persist `test_evaluation_count`, `test_evaluation_timestamp`, `checkpoint_selected_before_test`, and `test_used_during_training`.

Automatic exclusion rule: any run with `test_used_during_training=true`, a test count other than one, or checkpoint selection after the test timestamp is invalid. Historical code follows the intended call order but did not persist these fields, so this is verified from code provenance rather than run metadata.

Protocol v2 correction: training and validation use separate dataset objects over disjoint seed-fixed indices. Validation uses only deterministic tensor conversion/normalization for native-size CNN runs and deterministic resize/center-crop for DeiT. Historical v1 runs are excluded from revision inference. Tiny ImageNet uses 90,000 official training images for optimization, 10,000 held-out official training images for controller/checkpoint decisions, and the 10,000 labeled official validation images for exactly one final test evaluation.
""", encoding="utf-8")

    (AUDIT / "early_stopping_logic.md").write_text("""# Early-Stopping Logic

- Reliability improvement: `rbar < best_rbar - epsilon`.
- Validation improvement: `val_loss < best_val_loss - epsilon`.
- Reliability improvement resets both `bad_rel_lr` and `bad_rel_stop`; otherwise both increment.
- When `bad_rel_lr >= p_lr`, LR becomes `max(lr*gamma, min_lr)` and only `bad_rel_lr` resets.
- Stop condition: `bad_val >= p_stop OR bad_rel_stop >= p_stop`.
- This split-counter implementation makes reliability stopping reachable while retaining periodic LR reductions. `bad_rel` remains a history alias for `bad_rel_stop` for compatibility.
- Checkpoint retention: the epoch with strictly highest validation accuracy.

```text
update bad_rel_lr and bad_rel_stop using smoothed reliability score and epsilon
update bad_val using validation NLL and epsilon
if bad_rel_lr >= p_lr:
    reduce LR; bad_rel_lr = 0
if bad_val >= p_stop or bad_rel_stop >= p_stop:
    stop
retain model state when validation accuracy strictly improves
```
""", encoding="utf-8")

    (AUDIT / "generalization_gap_definition.md").write_text("""# Generalization-Gap Definition

Primary stored loss gap: `validation_NLL - training_NLL` at each epoch; signed, not absolute. The final reported value uses the final training epoch, not the selected checkpoint epoch. An additional accuracy gap is stored as `training_accuracy - validation_accuracy`. The controller's `V_t` is the positive part of the loss gap: `max(0, validation_NLL - training_NLL)`.
""", encoding="utf-8")
    (AUDIT / "sigma_audit.md").write_text("""# Sigma Audit

- `sigma_0 = 0.05` (`controller.prior_sigma` in every current JSON config; default also in `src/signals.py`).
- Fixed across current datasets/models: yes.
- Historical selection rationale: UNRESOLVED.
- Planned grid: 0.025, 0.05, 0.10, 0.20 with all other controller settings fixed.
""", encoding="utf-8")
    (AUDIT / "pd_parameter_audit.md").write_text("""# PD Parameter Audit

- `lambda_g = 0.0001`.
- `lambda_u = 0.001`.
- Defined in every current JSON config and as defaults in `src/signals.py`; fixed across datasets/models.
- Whether systematically tuned and why these scales were selected: UNRESOLVED.
- Explicit variants added without changing the default: FULL, V only, V+H, V+U, and H+U.
""", encoding="utf-8")
    (AUDIT / "tiny_imagenet_config.md").write_text("""# Tiny ImageNet Configuration

- Data: Tiny ImageNet-200, 200 classes, 64x64 RGB.
- Partitioning: seed-fixed 90/10 split of the 100,000 official training images; the 10,000 labeled official validation images are isolated as the final test set.
- Model: CIFAR-style ResNet-18, trained from scratch.
- Augmentation: random crop with 8-pixel padding and horizontal flip for training; deterministic evaluation transform for validation/test.
- Normalization: ImageNet mean/std.
- Optimization: SGD, lr=0.1, momentum=0.9, weight decay=0.0005, batch=128, maximum 120 epochs.
- Comparison: five methods, seeds 1..5, equal data/model/optimizer/budget.
- Configs: `revision_experiments/configs/tiny_imagenet/{smoke,full}.json`.
""", encoding="utf-8")
    (AUDIT / "vit_selection.md").write_text("""# Selected External Architecture

- Model: DeiT-Tiny/16 (`deit_tiny_patch16_224`), approximately 5M parameters.
- Initialization: official ImageNet-1k pretrained checkpoint `deit_tiny_patch16_224-a1311bcf.pth`; the 1,000-class head is discarded and a seed-specific 100-class head is initialized.
- Data: CIFAR-100 resized to 224; ImageNet normalization; random resized crop/flip for training and deterministic resize/center-crop for validation/test.
- Optimization: AdamW, lr=0.0005, weight decay=0.05, batch=128, maximum 100 epochs.
- Comparison: five methods, seeds 1..5, equal data/model/optimizer/budget.
- Configs: `revision_experiments/configs/deit_cifar100/{smoke,full}.json`.
""", encoding="utf-8")
    (AUDIT / "corrective_logic_audit.md").write_text("""# Corrective Logic Audit

## Fixed defects

1. **Stochastic validation observations:** CIFAR validation previously reused the augmented training dataset object. Protocol v2 uses a second dataset object with deterministic evaluation transforms over the same seed-fixed held-out indices.
2. **Unreachable reliability early stopping:** the LR patience reset prevented the reliability counter from reaching stop patience. Protocol v2 uses independent `bad_rel_lr` and `bad_rel_stop` counters.
3. **Redundant gradient-norm work:** the loop computed a full gradient norm for every minibatch but retained only the final value. Protocol v2 computes exactly the retained final-minibatch norm once per epoch.
4. **Invalid CUDA overhead attribution:** parameter-level `.item()` calls synchronized pending training kernels and counted that wait as controller work. Norm reductions now stay on-device, synchronize once, and use explicit timing boundaries; the initial PB reference is retained on-device to avoid repeated host-to-device copies.
5. **Unsafe completed-result reuse:** a `summary.json` at the expected path was previously treated as complete without checking scientific configuration or the remaining output files. Reuse now requires all nine artifacts and a matching scientific-run hash; partial outputs or mismatches are archived and rerun.
6. **Broken seed grouping:** scheduled single-run configs previously changed `experiment_name` for every method/seed, preventing five-seed statistical grouping. Temporary config filenames remain unique while the scientific experiment name remains stable.
7. **Missing fourth baseline in statistics:** Validation-Loss-Only is now included in the baseline set and paired comparisons.
8. **Missing revision postprocessing:** `analyze_revision.py` now materializes documented main-run sharing, summaries, paired bootstrap CIs, paired Cohen's d_z, raw tests, and Holm-corrected tests.
9. **Unrecorded queue failures:** `run_manifest.csv` is updated before and after every run; failures are retained and the queue can continue unless `--fail-fast` is requested.

## Verified invariants

- Training and internal validation indices are disjoint and reproducible by seed.
- Test evaluation occurs only after validation checkpoint selection.
- All five compared methods share data, model initialization, optimizer recipe, epoch ceiling, and checkpoint rule within a setting.
- Tiny ImageNet's labeled official validation split is isolated as final test data; its unlabeled official test split is unused.
- DeiT loads the official ImageNet-1k backbone while discarding only the 1,000-class head.
- Headline paired statistics are labeled headline-only at exactly five matched seeds.

No additional known core training/control logic gap remains after the regression suite and smoke tests. Long-run scientific conclusions remain pending the 260 corrected-protocol physical runs.
""", encoding="utf-8")


def write_alpha_beta(rows):
    """写出 alpha/beta 预注册敏感性网格。"""

    sensitivity = [row for row in rows if row.get("experiment_type") == "sensitivity"]
    grouped = {}
    for row in sensitivity:
        key = (row.get("controller_alpha"), row.get("controller_beta"))
        grouped.setdefault(key, []).append(row)
    output = []
    for (alpha, beta), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        output.append({
            "alpha": alpha, "beta": beta, "n": len(group),
            "mean_accuracy": sum(x["test_acc"] for x in group) / len(group),
            "mean_nll": sum(x["test_nll"] for x in group) / len(group),
            "mean_ece": sum(x["test_ece"] for x in group) / len(group),
            "mean_brier": sum(x["test_brier"] for x in group) / len(group),
            "mean_generalization_gap": sum(x["final_gen_gap_loss"] for x in group) / len(group),
            "mean_epochs": sum(x["epochs_run"] for x in group) / len(group),
            "selection_history": "UNRESOLVED; alpha=0.5 is consistent with neutral/equal weighting but no a-priori record was found",
        })
    write_csv(RESULTS / "alpha_beta_existing_analysis.csv", output,
              ["alpha", "beta", "n", "mean_accuracy", "mean_nll", "mean_ece", "mean_brier",
               "mean_generalization_gap", "mean_epochs", "selection_history"])


def main():
    """执行仓库审计并输出 CSV、配置和可读审计文档。"""

    for directory in [AUDIT, RESULTS, REVISION / "figures", REVISION / "logs"]:
        directory.mkdir(parents=True, exist_ok=True)
    rows = list(iter_summaries())
    reusable, missing = build_reusable(rows)
    fields = ["dataset", "model", "method", "seed", "config_hash", "split_hash", "initialization_hash",
              "checkpoint_path", "metrics_path", "reusable", "reuse_scope", "reason"]
    write_csv(AUDIT / "reusable_runs.csv", reusable, fields)
    write_csv(AUDIT / "missing_main_runs.csv", missing,
              ["experiment_group", "dataset", "model", "method", "seed", "reason"])
    materialize_configs()
    reusable_count = sum(row["reusable"] == "true" for row in reusable)
    write_audit_documents(reusable_count, len(missing))
    write_alpha_beta(rows)
    print(f"Historical summaries scanned: {len(rows)}")
    print(f"Reusable main rows: {reusable_count}/150")
    print(f"Missing main rows: {len(missing)}/150")


if __name__ == "__main__":
    main()
