# Paper 1 Revision Experiment Report

## Current Stage

Corrective implementation, external asset acquisition, runtime benchmarking, and five-method smoke verification are complete. The 260-run formal queue was launched at 2026-08-26 00:49:04 +08:00 and is running sequentially on the RTX 5070 Ti.

## Corrective Audit

- Historical protocol-v1 validation observations used random training augmentation, and the reliability LR reset made reliability-driven early stopping unreachable.
- Because both defects change controller trajectories, all 150 historical main-matrix slots are excluded from protocol-v2 inference.
- Protocol v2 uses deterministic validation transforms, disjoint seed-reproducible indices, and independent LR/stopping patience counters.
- Result reuse now requires all nine output artifacts and a scientific-config hash; queue failures and test-leakage metadata are persisted.
- Gradient-norm work and CUDA timing boundaries were corrected before benchmarking.
- Validation-Loss-Only is the fourth formal baseline; five-seed paired inference includes bootstrap CIs, paired effect sizes, raw tests, and Holm correction.
- Twelve protocol regression tests, static compilation, both external smoke suites, and artifact checks pass.

## External Validation

- Tiny ImageNet-200 is downloaded and MD5-verified (`90528d7ca1a48142e341f4ef8d21d0de`): 200 classes and 100,000/10,000/10,000 official train/labeled-validation/unlabeled-test images.
- Protocol v2 splits official training data into 90,000 optimization and 10,000 internal-validation images; labeled official validation is isolated as final test, and unlabeled official test is unused.
- Tiny ImageNet + ResNet-18 completed all five 5-epoch smoke runs with test accuracies from 0.2712 to 0.3947 and nine artifacts per run.
- DeiT-Tiny/16 uses the verified official ImageNet-1k checkpoint, replacing only the classification head; its five-method smoke suite also completed.

## Inventory and Runtime

- Final analysis inventory: 295 slots.
- Historical protocol-v2 reusable slots: 0.
- Corrected main runs deliberately supply 35 secondary-analysis slots without duplicate training.
- Unique formal physical runs: 260; completed 0; remaining 260; failed 0.
- All six experiment groups have measured RTX 5070 Ti throughput; none is awaiting a benchmark.
- Conservative remaining time: 172.6 GPU-hours, 7.19 days at 24 GPU-hours/day, or 8.63 days at 20 GPU-hours/day.
- From the 2026-08-26 00:32:09 +08:00 estimate, expected completion is 2026-09-02 05:08:15 +08:00 under continuous execution or 2026-09-03 15:39:28 +08:00 at 20 GPU-hours/day.
- The estimate includes first-epoch warm-up and assumes maximum epochs; it does not speculate about future early stopping.

## Live Operation

- Dashboard: `http://127.0.0.1:8000/`, refreshed every two seconds.
- The global view combines the 260-row manifest, current batch/epoch state, training and validation metrics, reliability/LR signals, GPU telemetry, ETA, output paths, and queue/current-run logs.
- C5O2 campaign `20260825T095903Z_resume` was paused before this queue started. Six complete C5O2 jobs remain intact; its interrupted `cs9_cifar100_resnet18_cifar_safe_no_postft_check_s70_seed44` job must restart when C5O2 resumes.
- Formal protocol-v2 run directories use `results_v2/<experiment>/<method>/seed_<n>/` as the authoritative output location alongside earlier experiment results. `revision_experiments/results/` contains orchestration state, logs, ETA files, smoke outputs, and revision aggregates rather than formal per-run artifacts.

See `audit/corrective_logic_audit.md`, `DATA_MODEL_PROVENANCE.md`, `RUNTIME_ESTIMATE.md`, `REVISION_STATUS.md`, `results/run_manifest.csv`, and `results/runtime_benchmarks.csv` for auditable details.
